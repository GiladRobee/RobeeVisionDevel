import cv2

img = cv2.imread('/home/gilad/dev_ws/RobeeVisionDevel/images/14.4.24/1_Color.png')
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.imshow("output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
