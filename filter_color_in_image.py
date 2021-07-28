import cv2
import numpy as np

img = cv2.imread('test.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_color = np.array([60,0,130])
upper_color = np.array([80,25,180])

mask = cv2.inRange(hsv, lower_color, upper_color)
result = cv2.bitwise_and(img, img, mask= mask)

cv2.imshow('ori',img)
cv2.imshow('mask',mask)
cv2.imshow('result',result)

cv2.waitKey(0)
cv2.destroyAllWindows()
