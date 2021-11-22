import cv2
import mediapipe as mp
import time                     # used for frame rate

cap = cv2.VideoCapture(0)       # open camera for video capture

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    #print(results.multi_hand_landmarks)

    # check for number of hands
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms)

    cv2.imshow("Image", img)    # display captured video on screen
