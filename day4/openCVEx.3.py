# 모자이크 처리
import cv2
import matplotlib.pyplot as plt

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

img = cv2.imread('../datasets/opencv/cat.jpg')
mos = mosaic(img, (10,80,210,320), 10)

cv2.imwrite('../datasets/opencv/reslut/cat_mosaic.jpg', mos)
plt.imshow(cv2.cvtColor(mos, cv2.COLOR_BGR2RGB))
plt.show()


