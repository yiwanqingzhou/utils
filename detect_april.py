import numpy as np
import cv2
import apriltag

detector = apriltag.Detector()
img = cv2.imread("1.png")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
detections, dimg = detector.detect(gray, True)

if len(detections) == 0:
  print("No detetions")
else:
  print("success")
  for detection in detections:
    points_2d = np.round(detection.corners).astype(int)
    for point_2d in points_2d:
      x, y = point_2d
      print(x,y)
