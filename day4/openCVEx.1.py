import cv2
import urllib.request as req

url = 'http://uta.pw/shodou/img/28/214.png'
req.urlretrieve(url, '../datasets/opencv/downimage.png')

img = cv2.imread('../datasets/opencv/downimage.png')
print(img)

import matplotlib.pyplot as plt
img = cv2.imread('../datasets/opencv/test.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite('../datasets/opencv/result/test.png',img)

# img_resize
img2 = cv2.resize(img, (600,300))
cv2.imwrite('../datasets/opencv/result/test_resize.png', img2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.show()

# img_crop
img3 = img[150:450, 150:450]
cv2.imwrite('../datasets/opencv/result/test_crop.png', img3)
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
plt.show()
