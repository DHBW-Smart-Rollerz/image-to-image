import cv2

img = cv2.imread("data/1_dashcam/0001.jpg")

# Beispiel-Autobox (x1, y1, x2, y2)
x1, y1, x2, y2 = 100, 300, 400, 480

roi = img[y1:y2, x1:x2]
blurred = cv2.GaussianBlur(roi, (51, 51), 0)
img[y1:y2, x1:x2] = blurred

cv2.imwrite("frame_blur.png", img)
