import cv2
import matplotlib.pyplot as plt

# 참고 : https://github.com/opencv/opencv/tree/master/data/haarcascades
cascade_file = '../datasets/opencv/haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_file)

img = cv2.imread('../datasets/opencv/girl.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_list = cascade.detectMultiScale(img_gray, minSize=(150,150))

if len(face_list) == 0:
    print('검출 실패')
    quit()

for (x,y,w,h) in face_list:
    print('인식좌표:',x,y,w,h)
    red = (0,0,255)         #(B,R,G)
    cv2.rectangle(img, (x,y), (x+w, y+h), red, thickness=15)

cv2.imwrite('../datasets/opencv/result/girl_face_detect.png', img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

