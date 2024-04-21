import cv2
import os
import numpy as np
import tkinter as tk
import cv2
height, width = tk.Tk().winfo_screenwidth(), tk.Tk().winfo_screenheight()
class DrawLineWidget(object):
    def __init__(self, image_path):
        self.original_image = cv2.imread(path, cv2.IMREAD_COLOR)
        self.clone = self.original_image.copy()
        self.lines = []
        self.line_counter = 0
        self.slopes = []
        cv2.namedWindow('image')
        cv2.resizeWindow('image', height, width)
        cv2.setMouseCallback('image', self.extract_coordinates)

        # List to store start/end points


    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates_s = (x,y)
            print('Starting: ', self.image_coordinates_s)

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.image_coordinates_e = (x,y)
            print('Starting: {}, Ending: {}'.format(self.image_coordinates_s, self.image_coordinates_e))
            cv2.line(self.clone, self.image_coordinates_s, self.image_coordinates_e, (36,255,12), 2)
 
            self.line_counter+=1
            self.slopes.append((self.image_coordinates_e[1] - self.image_coordinates_s[1])/(self.image_coordinates_e[0] - self.image_coordinates_s[0]))
            # Draw line
            cv2.resizeWindow('image', height, width)
            cv2.imshow("image", self.clone) 

        

    def show_image(self):
        return self.clone

if __name__ == '__main__':
    abs_path = os.path.abspath(os.path.dirname(__file__))
    print(abs_path)
    rel_path = "../images/21.4/0_l_11_4k_rotated.png"
    path = os.path.join(abs_path, rel_path)
    print(path)
    draw_line_widget = DrawLineWidget(path)
    looping =True
    cv2.namedWindow('image')
    cv2.resizeWindow('image', height, width)

    while looping:
        cv2.imshow('image', draw_line_widget.original_image)
        key = cv2.waitKey(1)

        # Close program with keyboard 'q'
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit(1)
        if draw_line_widget.line_counter == 2:
            looping = False
            cv2.destroyAllWindows()

    slope1 = draw_line_widget.slopes[0]
    slope2 = draw_line_widget.slopes[1]
    ang1 = np.arctan(slope1)
    ang2 = np.arctan(slope2)
    print(ang1, ang2)
    while ang1 < 0:
        ang1 += np.pi
    while ang2 < 0:
        ang2 += np.pi
    print(ang1, ang2)
    angle = ang1 - ang2
    print(angle)
    if angle < 0:
        angle += np.pi
    print(angle*180/np.pi," degrees")
