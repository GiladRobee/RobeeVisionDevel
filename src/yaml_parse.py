import yaml
import cv2
import os
import numpy as np
from pprint import pprint

if __name__ == '__main__':
    abs_path = os.path.abspath(os.path.dirname(__file__))
    print(abs_path)
    rel_path_im = "../images/18.4/3_l_Color.png"
    path_im = os.path.join(abs_path, rel_path_im)
    print(path_im)
    rel_path_yaml = "../config/"+rel_path_im.split("/images/")[-1].removesuffix(".png")+".yaml"
    path_yml = os.path.join(abs_path, rel_path_yaml)
    # reading the image 
    img = cv2.imread(path_im, cv2.IMREAD_COLOR) 
    
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image', img)
 
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    masks_points = []
    with open(path_yml) as stream:
        try:
            data = yaml.safe_load(stream)
            # print(data)
        except yaml.YAMLError as exc:
            print(exc)

    num_of_masks = data["NumOfMasks"]

    for i in range(num_of_masks):
        mask = data["Mask"+str(i+1)]
        # print("mask: ",i+1," = ", mask)
        masks_points.append(mask["Points"])
        # for j in mask["Points"]:
        #     print(j)
    
    match data["Dtype"]:
        case "np.int32":
            dtype = np.int32
        case "np.float32":
            dtype = np.float32
        case "np.float64":
            dtype = np.float64
        case "np.int64":
            dtype = np.int64
        case "np.int16":
            dtype = np.int16
        case _ :
            dtype = np.int32
    masks_points = np.array(masks_points, dtype=dtype)    
    pprint(masks_points)