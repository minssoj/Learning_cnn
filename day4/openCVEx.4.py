# mini assignment : 모자이크 처리
import cv2
import matplotlib.pyplot as plt

# 모자이크 함수
def mosaic(img, rect, size):
    (x1,y1,x2,y2) = rect
    w = x2-x1
    h = y2-y1

    i_rect = img[y1:y2, x1:x2]
    i_small = cv2.resize(i_rect, (size, size))
    i_mos = cv2.resize(i_small, (w,h), interpolation=cv2.INTER_AREA)
    img2 = img.copy()
    img2[y1:y2, x1:x2] = i_mos
    return img2

# 얼굴인식 정보, 참고 : https://github.com/opencv/opencv/tree/master/data/haarcascades
cascade_file = '../datasets/opencv/haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_file)

img = cv2.imread('../datasets/opencv/family.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_list = cascade.detectMultiScale(img_gray, minSize=(150,150))

if len(face_list) == 0:
    print('검출 실패')
    quit()

img2 = img.copy()
img_mos = img.copy()
for (x,y,w,h) in face_list:
    print('인식좌표:',x,y,w,h)
    red = (0,0,255)         #(B,R,G)
    cv2.rectangle(img2, (x,y), (x+w, y+h), red, thickness=15)
    img_mos = mosaic(img_mos, (x,y, (x+w), (y+h)), 10)


plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

plt.subplot(1,2,2)
cv2.imwrite('../datasets/opencv/result/family_mosaic.jpg', img_mos)
plt.imshow(cv2.cvtColor(img_mos, cv2.COLOR_BGR2RGB))
plt.show()

