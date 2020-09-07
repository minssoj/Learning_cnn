import cv2

img = cv2.imread('../datasets/opencv/test.png', cv2.IMREAD_GRAYSCALE)

ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)#픽셀 값이 threshold_value보다 작으면 0, 크면 value
ret, th2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)#픽셀 값이 threshold_value보다 크면 0, 작으면 value
ret, th3 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)#픽셀 값이 threshold_value보다 작으면 0, 크면 픽셀 값
ret, th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)#픽셀 값이 threshold_value보다 작으면 0, 크면 픽셀 값

cv2.imshow('orginal', img)
cv2.imshow('BINARY', th1)
cv2.imshow('BINARY_IVY', th2)
cv2.imshow('TOZERO', th3)
cv2.imshow('TOZERO_INV', th4)
cv2.waitKey(0)
cv2.destroyAllWindows()