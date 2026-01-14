
import cv2, os
def gen_canny(in_path, out_path):
  img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
  edges = cv2.Canny(img, 100, 200)
  cv2.imwrite(out_path, edges)