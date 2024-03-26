# coding: utf-8

import cv2
import numpy as np

img = cv2.imread("80_gray.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_show = img.copy()

min_h = min_s = min_v = 255
max_h = max_s = max_v = 0


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (y, x)

        cv2.circle(img_show, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img_show, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img_show)

        point_hsv = hsv[y, x]
        h = point_hsv[0]
        s = point_hsv[1]
        v = point_hsv[2]
        print("hsv: (%d,%d,%d)" % (h, s, v))
        
        point_rgb = img[y, x]
        r = point_rgb[0]
        g = point_rgb[1]
        b = point_rgb[2]
        print("rgb: (%d,%d,%d)" % (r, g, b))

        global min_h, min_s, min_v, max_h, max_s, max_v
        min_h = min(min_h, h)
        min_s = min(min_s, s)
        min_v = min(min_v, v)
        max_h = max(max_h, h)
        max_s = max(max_s, s)
        max_v = max(max_v, v)


cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img_show)

while (True):
    if cv2.waitKey(100) == ord('q'):
        break

cv2.destroyAllWindows()


print("min hsv: (%d,%d,%d)" % (min_h, min_s, min_v))
print("max hsv: (%d,%d,%d)" % (max_h, max_s, max_v))

lower_color = np.array([min_h - 10, min_s - 10, min_v - 10])
upper_color = np.array([max_h + 10, max_s + 10, max_v + 10])

mask = cv2.inRange(hsv, lower_color, upper_color)
result = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow('ori', img)
cv2.imshow('mask', mask)
cv2.imshow('result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
