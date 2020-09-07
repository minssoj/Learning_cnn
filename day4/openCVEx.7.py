import cv2
import matplotlib.pyplot as plt
img = cv2. imread('../datasets/opencv/flower.jpg')
img = cv2.resize(img, (300, 170))
gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gimg = cv2.GaussianBlur(gimg, (7,7), 0)
im2 = cv2.threshold(gimg, 140, 240, cv2.THRESH_BINARY_INV)[1]
plt.subplot(1,2,1)
plt.imshow(im2, cmap='gray')
'''
mode:
    cv2.RETR_LIST : 단순한 윤곽 검출
    cv2.EXTERNAL: 가장 외곽에 있는 윤곽만 검출
    cv2.RETR_TREE: 모든 윤곽 검출하고 계층 구조로 저장
method:
    cv2.CHAIN_APPROX_NONE : 모든 윤곽에 있는 모든 점을 반환
    cv2.CHAIN_APPROX_SIMPLE : 의미없는 정보를 제거하고 점을 반환
    cv2.CHAIN_APPROX_TC89_L1 : Ten_chin 연결 근사 알고리즘을 적용하여 컨투어 포인트를 줄임
'''
cnts, hirachy = cv2.findContours(im2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
for pt in cnts:
    x, y, w, h = cv2.boundingRect(pt)
    if w < 30 or w > 200: continue
    cv2.rectangle(img, (x,y),(x+w, y+h), (0,255,0), 2)
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.savefig('../datasets/opencv/result/find_contuous.png', dpi=200)
plt.show()