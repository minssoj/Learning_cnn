import cv2
import matplotlib.pyplot as plt

def detect_zipno(img):
    h, w = img.shape[:2]
    img = img[0:h//2, w//3:]
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gimg = cv2.GaussianBlur(gimg, (3,3), 0)
    im2 = cv2.threshold(gimg, 140, 255, cv2.THRESH_BINARY_INV)[1]
    cnts, hierachy = cv2.findContours(im2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for pt in cnts:
        x,y,w,h = cv2.boundingRect(pt)
        if not(50 < w < 70) : continue
        result.append([x,y,w,h])
    result = sorted(result, key=lambda x:x[0])
    result2 = []
    lastx = -100
    for x, y, w, h in result:
        if(x-lastx) < 10: continue
        result2.append([x,y,w,h])
    for x,y,w,h in result2:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
    return result2, img

input_img = cv2.imread('../datasets/opencv/hagaki1.png')
cnts, img = detect_zipno(input_img)
print('zip의 개수',len(cnts))
print(cnts)
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.savefig('../datasets/opencv/result/hagaki1_contuous_zip.png', dpi=200)
plt.show()