import cv2
import os
import numpy as np
import yaml
import tkinter as tk


points = []
# function to display the coordinates of 
# of the points clicked on the image  
def click_event(event, x, y, flags, params): 
  
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        print(x, ' ', y) 
        points.append([x, y])
        # displaying the coordinates 
        # on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font, 
                    1, (255, 0, 0), 2) 
        cv2.imshow('image', img) 
  
    # checking for right mouse clicks      
    if event==cv2.EVENT_RBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        print(x, ' ', y) 
  
        # displaying the coordinates 
        # on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        b = img[y, x, 0] 
        g = img[y, x, 1] 
        r = img[y, x, 2] 
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r), 
                    (x,y), font, 1, 
                    (255, 255, 0), 2) 
        cv2.imshow('image', img) 
  
# driver function 
if __name__=="__main__": 
    """
    This function is used to create a yaml file from the points clicked on the image
    Do 5 clicks per Mask
    """
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    abs_path = os.path.abspath(os.path.dirname(__file__))
    print(abs_path)
    rel_path = "../images/21.4/0_l_11_4k_rotated.png" #"../images/18.4/3_l_Color.png"
    path = os.path.join(abs_path, rel_path)
    print(path)
    # reading the image 
    img = cv2.imread(path, cv2.IMREAD_COLOR) 
    
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', screen_width, screen_height)

    # displaying the image 
    cv2.imshow('image', img) 
  
    # setting mouse handler for the image 
    # and calling the click_event() function 
    cv2.setMouseCallback('image', click_event) 
  
    # wait for a key to be pressed to exit 
    cv2.waitKey(0) 
  
    # close the window 
    cv2.destroyAllWindows() 
    goal = float(input("Enter the goal: "))
    NumOfMasks = len(points)//5
    Dtype = "np.int32"
    data = {"NumOfMasks": NumOfMasks}
    data["Goal"] = goal
    data["Dtype"] = Dtype
    for i in range(int(NumOfMasks)):
        data["Mask"+str(i+1)] = {"Points": points[i*5:i*5+5]}
    print(data)
    rel_path_yaml = "../config/"+rel_path.split("/images/")[-1].removesuffix(".png")+".yaml"
    temp_dir = rel_path.split("/images/")[-1].split("/")[0]
    if not os.path.exists(os.path.join(abs_path, "../config/"+temp_dir)):
        os.makedirs(os.path.join(abs_path, "../config/"+temp_dir))
    path_yml = os.path.join(abs_path, rel_path_yaml)
    with open(path_yml, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    print("yaml file created at: ", path_yml)
    print("yaml file content: ")
    with open(path_yml) as stream:
        try:
            data = yaml.safe_load(stream)
            print(data)
        except yaml.YAMLError as exc:
            print(exc)
