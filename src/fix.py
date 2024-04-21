import cv2
import numpy as np
import tkinter as tk
height, width = tk.Tk().winfo_screenwidth(), tk.Tk().winfo_screenheight()
img = cv2.imread("/home/gilad/dev_ws/RobeeVisionDevel/images/21.4/0_l_11_4k.png", cv2.IMREAD_COLOR)
height, width, channels = img.shape
rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
cv2.namedWindow("Rotated Image", cv2.WINDOW_NORMAL)
cv2.imshow("Rotated Image", rotated_img)
cv2.resizeWindow("Rotated Image", height, width)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("/home/gilad/dev_ws/RobeeVisionDevel/images/21.4/0_l_11_4k_rotated.png", rotated_img)
