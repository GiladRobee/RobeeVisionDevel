
import pyrealsense2 as rs
import numpy as np
import cv2
import sys
from segment_anything import sam_model_registry, SamPredictor
import torch
import torchvision
import matplotlib.pyplot as plt
import math
import signal
def importport():
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())

class predictorSegmentation:

    def __init__(self, model_name, model_path,device):
        """
        init model and predictor
        """
        print("start predictorSegmentation")
        self.model = model_name
        self.model_path = model_path
        self.device = device
        self.sam = sam_model_registry[self.model](checkpoint=self.model_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)

    def setImg(self, img):
        """
        set image to predict
        """
        self.img = cv2.imread(img)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(self.img)
        

    def setRef(self, ref): #array of tuples
        """
        set reference points to predict ([[x1,y1],[x2,y2],...])
        """
        self.ref = ref

    def setLabel(self, label):
        """
        set label for reference points ([1,0,1,0,...])
        """
        self.label = label

    def setRefLabel(self,ref,label):
        """
        set reference points and label
        """
        self.setRef(ref)
        self.setLabel(label)

    def showInputImage(self):
        """
        show input image with reference points
        """
        plt.figure(figsize=(10,10))
        plt.imshow(self.img)
        pos_points = self.ref[self.label==1]
        neg_points = self.ref[self.label==0]
        plt.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=255, edgecolor='white', linewidth=1.25)
        plt.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=255, edgecolor='white', linewidth=1.25)
        plt.axis('on')
        plt.show()
        plt.waitforbuttonpress()
        plt.close() 
        
    def show_mask(self,mask,h,w,z):
        """
        segmented mask
        """
        
        color = np.array([30/255, 144/255, 255/255, 0.2])
        # color = np.array([30/255, 144/255, 255/255])
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        plt.imshow(mask_image)
        
    
    def predict(self):
        """
        predict segmentation
        """
        masks,scores,logits = self.predictor.predict(
            point_coords=self.ref,
            point_labels=self.label,
            multimask_output=False,
        )
        self.mask=masks
        self.score = scores
        self.logits = logits
        [self.mask_shape_z,self.mask_shape_y,self.mask_shape_x] = self.mask.shape
        print(f"mask shape: {self.mask_shape_x}x{self.mask_shape_y}x{self.mask_shape_z}")
    
    def showSegmentResults(self):
        """
        show segment results
        """
        for i,(mask, score) in enumerate(zip(self.mask, self.score)):
            plt.figure(figsize=(10,10))
            plt.imshow(self.img)
            self.show_mask(mask,self.mask_shape_y,self.mask_shape_x,self.mask_shape_z)
            pos_points = self.ref[self.label==1]
            neg_points = self.ref[self.label==0]
            plt.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=255, edgecolor='white', linewidth=1.25)
            plt.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=255, edgecolor='white', linewidth=1.25)
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()  


class realSenseDevice:
    def __init__(self):
        """
        init realsense device
        """
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))
        print(self.device_product_line)
        for s in self.device.sensors:
            print(s.get_info(rs.camera_info.name))
           
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
    def getFrame(self):
        """
        get frame from realsense device
        """
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None
        self.depth_image = np.asanyarray(depth_frame.get_data())
        self.color_image = np.asanyarray(color_frame.get_data())
        self.depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_AUTUMN)
        self.depth_colormap_dim = self.depth_colormap.shape
        self.color_colormap_dim = self.color_image.shape
        if self.depth_colormap_dim != self.color_colormap_dim:
            self.color_image = cv2.resize(self.color_image, dsize=(self.depth_colormap_dim[1], self.depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
    
    def __del__(self):
        """
        close realsense device
        """
        # if self.pipeline.():
        #     self.pipeline.stop()
        # self.pipeline.stop()
        print("realsense device stopped")
    
    def showFrame(self):
        """
        show frame
        """
        images = np.hstack((self.color_image, self.depth_colormap))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        
    def convertAndSaveRGB(self,path,name):
        """
        convert and save RGB image
        """
        cv2.imwrite(path+name+".jpg",self.color_image)

class postProcess:
    def __init__(self):
        self.lineQue = list()
        self.lineEqQue = list()
    def setMask(self,mask):
        self.mask = mask
        min = np.min(mask)
        max = np.max(mask)
        div = max/float(255)
        self.maskgray = (mask/div).astype(np.uint8)
    def getMask(self):
        return self.maskgray
    def setImg(self,img):
        self.img = img
    def cannyEdge(self):
        self.dst = cv2.Canny(self.maskgray,10,200)
        self.cdst = cv2.cvtColor(self.dst, cv2.COLOR_GRAY2BGR)
    def gaussianBlur(self):
        self.blur = cv2.GaussianBlur(self.maskgray, (5, 5), 0)
    def HougeLines(self):
        lines = cv2.HoughLines(self.dst, 1, (np.pi / 180)/2, 140, None, 0, 0)
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                # print(pt1,pt2)
                
                self.lineQue.append([pt1,pt2])

    def HarrisCorner(self):
        self.maskgray32 = np.float32(self.maskgray)
        self.harrisdst = cv2.cornerHarris(self.maskgray32,5,3,0.1)
        self.harrisdst = cv2.dilate(self.harrisdst,None)
        self.inputcpy = self.img.copy()
        self.inputcpy[self.harrisdst>0.01*self.harrisdst.max()]=[0,0,255]

    def filterLines(self):
        print("image shape: ",self.img.shape)
        for line in self.lineQue:
            print(line)
            if line[0][0] == 999 or line[0][0]==-1000 or line[0][1] == 999 or line[0][1]==-1000 or line[0][1] == 0 or line[0][1]== 1:
                print("line is out of bound")
                self.lineQue.remove(line)
    def printLines(self):
        for line in self.lineQue:
            cv2.line(self.cdst, line[0], line[1], (0,0,255), 3, cv2.LINE_AA)  
            cv2.line(self.img, line[0], line[1], (0,0,255), 3, cv2.LINE_AA)  
    def line2equation(self):
        for line in self.lineQue:
            x1 = line[0][0]
            y1 = line[0][1]
            x2 = line[1][0]
            y2 = line[1][1]
            m = (y2-y1)/(x2-x1)
            n = y1 - m*x1
            self.lineEqQue.append([m,n])
    def computeAngle(self):
        for i in range(len(self.lineEqQue)):
            for j  in range(len(self.lineEqQue)):
                if j <= i:
                    continue
                else:
                    m1 = self.lineEqQue[i][0]
                    m2 = self.lineEqQue[j][0]
                    angle = math.atan((m2-m1)/(1+m1*m2))
                    print("angle between line ",i," and line ",j," is: ",angle,"[rad] or ",angle*180/np.pi,"[deg]")
    def showResults(self):
        cv2.imshow("Source", self.img)
        cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", self.cdst)

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)
def main():
    signal.signal(signal.SIGINT, signal_handler)
    rsd = realSenseDevice()
    rsd.getFrame()
    # rsd.showFrame()
    rsd.convertAndSaveRGB("/home/gilad/dev_ws/segment-anything/notebooks/images/","test")
    
    seg = predictorSegmentation("vit_h","/home/gilad/dev_ws/segment-anything/models/sam_vit_h_4b8939.pth","cpu")
    seg.setImg('/home/gilad/dev_ws/segment-anything/notebooks/images/test.jpg')
    seg.setRef(np.array([[100,100],[400,100],[600,200]]))
    seg.setLabel(np.array([1,1,1]))
    seg.showInputImage()
    seg.predict()
    seg.showSegmentResults()

    pp = postProcess()
    # pp.setMask(seg.mask[len(seg.mask)-1])
    pp.setMask(seg.mask[0])
    pp.setImg(rsd.color_image)
    pp.gaussianBlur()
    pp.cannyEdge()
    pp.HougeLines()
    pp.filterLines()
    pp.printLines()
    pp.showResults()
    pp.line2equation()
    pp.computeAngle()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()