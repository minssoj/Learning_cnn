import cv2

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('../datasets/opencv/fish.mp4')
img_last = None
green = (0,255,0)

while True:
    _ret, frame = cap.read()
    frame = cv2.resize(frame, (500,400))
    gimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gimg = cv2.GaussianBlur(gimg, (9,9), 0)
    img_b = cv2.threshold(gimg, 100, 255, cv2.THRESH_BINARY_INV)[1]
    if img_last is None:
        img_last = img_b
        continue
    frame_diff = cv2.absdiff(img_last, img_b)
    cnts, hierachy = cv2.findContours(frame_diff.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for pt in cnts:
        x,y,w,h = cv2.boundingRect(pt)
        if 30 < w : continue
        cv2.rectangle(frame, (x, y), (x+w, y+h), green, 2)
    img_last = img_b
    cv2.imshow('diff camera', frame)
    cv2.imshow('diff data', frame_diff)
    k = cv2.waitKey(1) #1msec 대기
    if k==27 or k==13 : break
cap.release()
cv2.destroyAllWindows()
