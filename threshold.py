import cv2
import numpy as np

img = cv2.imread('gray.png')
ret,thresh = cv2.threshold(img,10,255,cv2.THRESH_BINARY)



cv2.imshow('thresh',thresh)
# cv2.imwrite("80_gray_threshold.png", thresh)




cv2.waitKey(0)
cv2.destroyAllWindows()
