import cv2
import pygame
import numpy as np
import os
import mysql.connector

# Initialize pygame mixer
pygame.mixer.init()

# Use raw string for the file path or escape backslashes
audio_file_path = r"C:\Users\SHYAMALA\Desktop\mini project\relaxing-guitar-loop-v5-245859.mp3"

try:
    pygame.mixer.music.load(audio_file_path)
    pygame.mixer.music.play()
    print("Audio is playing...")
except pygame.error as e:
    print(f"Error loading or playing audio file: {e}")

# MySQL connection setup
try:
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Poojitha@143",  # Add your password here
        database="d1"
    )
    cursor = mydb.cursor()
    print("Database connected successfully.")
except mysql.connector.Error as err:
    print(f"Error: {err}")
    exit()

# Specify the paths for Haar Cascades
nose_cascade_path = cv2.data.haarcascades + 'haarcascade_mcs_nose.xml'
mouth_cascade_path = cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml'

# Load Haar Cascades if available
if os.path.exists(nose_cascade_path) and os.path.exists(mouth_cascade_path):
    nose_cascade = cv2.CascadeClassifier(nose_cascade_path)
    mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)
else:
    print("Warning: Haar Cascade files for nose and mouth not found!")
    nose_cascade = None
    mouth_cascade = None

# Dummy faceDis array for example (replace with your face recognition logic)
faceDis = []  # Example face distances, populate this properly

# Check if faceDis contains values before calling np.argmin
if len(faceDis) > 0:
    matchIndex = np.argmin(faceDis)
    print("Match found with index:", matchIndex)
else:
    print("No faces detected. Cannot determine the closest match.")
