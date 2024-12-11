
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pygame
import mysql.connector

# Load images and class names
path = 'photos'
images = [face_recognition.load_image_file(f'{path}/{img}') for img in os.listdir(path)]
classNames = [os.path.splitext(img)[0] for img in os.listdir(path)]
encodeListKnown = [face_recognition.face_encodings(img)[0] for img in images]

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize variables for attendance marking and database connection
attendance_recorded = set()
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Poojitha@143",
    database="d1"
)
mycursor = mydb.cursor()

# Play audio file
pygame.mixer.init()
pygame.mixer.music.load(r"C:\Users\SHYAMALA\Desktop\mini project\relaxing-guitar-loop-v5-245859.mp3")
pygame.mixer.music.play()

# Load Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml')

# Set threshold for similarity score
threshold = 0.6

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Find all face locations and encodings in the current frame
    faceLocsCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, faceLocsCurFrame)

    # Keep track of attendance marked for this session
    attendance_marked_session = set()

    for encodeFace, faceLoc in zip(encodesCurFrame, faceLocsCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        
        # Check if the face matches any known face above the threshold
        if matches[matchIndex] and faceDis[matchIndex] < threshold:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            
            if name not in attendance_recorded and name not in attendance_marked_session:
                # Detect eyes, nose, and mouth
                roi_gray = img[y1:y2, x1:x2]
                eyes = eye_cascade.detectMultiScale(roi_gray)

                if len(eyes) == 0:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(img, 'Please keep all facial features visible for attendance', (x1 + 6, y2 + 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                else:
                    # Draw bounding box around the face
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                    # Write name, date, and time to MySQL database
                    now = datetime.now()
                    date_str = now.strftime('%Y-%m-%d')  # Convert date object to string
                    time_str = now.strftime('%H:%M:%S')   # Convert time object to string
                    sql = "INSERT INTO std (name, date, time) VALUES (%s, %s, %s)"
                    val = (name, date_str, time_str)
                    mycursor.execute(sql, val)
                    mydb.commit()

                    # Add the name to the set of recorded attendees
                    attendance_recorded.add(name)
                    attendance_marked_session.add(name)

                    # Display 'Attendance Marked' on webcam
                    cv2.putText(img, f'Attendance Marked for {name}', (x1 + 6, y2 + 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                # Display 'Attendance Already Marked' on webcam
                cv2.putText(img, f'Attendance Already Marked for {name}', (x1 + 6, y2 + 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

        else:
            # If no match found or below threshold, display "Unknown"
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, 'Unknown', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, 'Please keep all facial features visible for attendance', (x1 + 6, y2 + 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

    # Get current date
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Query the database to count the number of persons for the current date
    mycursor.execute("SELECT COUNT(DISTINCT name) FROM std WHERE date = %s", (today,))
    count = mycursor.fetchone()[0]
    count_text = f"Total Number of Attendence Taken ON {today}: {count} PERSONS"

    # Display the count of persons on the webcam feed
    cv2.putText(img, count_text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)

    # Show the image with detected faces
    cv2.imshow('Webcam', img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()