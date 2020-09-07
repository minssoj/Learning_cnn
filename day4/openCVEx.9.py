import cv2
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('../datasets/opencv/fish.mp4')

while True:
    _ret, frame = cap.read()
    frame = cv2.resize(frame, (500,400))
    cv2.imshow('opencv camera', frame)
    k = cv2.waitKey(1) #1msec 대기
    if k==27 or k==13 : break
cap.release()
cv2.destroyAllWindows()

import numpy as np
while True:
    _ret, frame = cap.read()
    frame = cv2.resize(frame, (500,400))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0] #hue
    s = hsv[:, :, 1] #saturation
    v = hsv[:, :, 2] # value brighthness
    img = np.zeros(h.shape, dtype=np.uint8)
    img[((h < 50) | (h > 200)) & (s > 100)] = 255
    cv2.imshow('opencv camera', img)
    k = cv2.waitKey(1) #1msec 대기
    if k==27 or k==13 : break
cap.release()
cv2.destroyAllWindows()
