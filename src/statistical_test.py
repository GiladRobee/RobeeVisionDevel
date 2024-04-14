import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import *
import operator
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from enum import Enum
import pyrealsense2 as rs

class readMethod(Enum):
    file_path = 1
    realsense = 2
    seg_anything = 3
class filterChoice(Enum):
    sobel = 1
    prewitt = 2
    laplacian = 3
    canny = 4
    roberts = 5
    custom = 6
class imageIO:
    def __init__(self) -> None:
        self.read_method_ = readMethod.file_path # default read method
        self.filterChoice = filterChoice.prewitt

    def __del__(self):
        if self.read_method_ == readMethod.realsense:
            self.stopRealsense()
            cv2.destroyAllWindows()

    def getReadMethod(self):
        return self.read_method_
    
    def startRealsense(self):
        self.rs_pipeline = rs.pipeline()
        self.rs_config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.rs_pipeline)
        pipeline_profile = self.rs_config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        print("Realsense device id: ",device_product_line)
        self.rs_config.enable_stream(rs.stream.depth, 1280, 800, rs.format.z16, 10)
        self.rs_config.enable_stream(rs.stream.color, 1280, 800, rs.format.bgr8, 10)
        self.rs_pipeline.start(self.rs_config)

    def stopRealsense(self):
        try:
            self.rs_pipeline.stop()
        except Exception as e:
            print("Error in stopping realsense pipeline: ",e)

    def getRealsenseImage(self):
        try:
            frames = self.rs_pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
            self.input_img = np.asanyarray(color_frame.get_data())
            return self.input_img, depth_image
        except Exception as e:
            print("Error in getting realsense image: ",e)
        finally:
            self.rs_pipeline.stop()
            cv2.destroyAllWindows()

    def setReadMethod(self, method: readMethod):
        self.read_method_ = method
        if method == readMethod.realsense:
            self.startRealsense()

    def readImage(self, path: str):
        match self.read_method_:
            case readMethod.file_path:
                self.input_img = cv2.imread(path)
                
            case readMethod.realsense:
                pass
            case _:
                raise ValueError("Invalid read method")
        
        return self.input_img
            
    def writeImage(self, path: str, image: cv2.Mat):
        cv2.imwrite(path, image)
    
    def showImage(self, image: cv2.Mat):
        cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
        cv2.imshow('output', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def setFilterChoice(self, choice: filterChoice):
        self.filterChoice = choice

    def getFilterChoice(self):
        return self.filterChoice

    def applySobel(self, image: cv2.Mat,ksize:int=3, scale:int=1):
        sx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)*scale
        sy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)*scale

        sob = np.hypot(sx, sy)
        return sob.astype(np.uint8)

    def applyPrewitt(self, image: cv2.Mat, ksize:int=3, scale:int=1,threshold:int=5):
        pew_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) * scale
        pew_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) * scale
        prewitx = cv2.filter2D(image, -1, pew_x)
        prewity = cv2.filter2D(image, -1, pew_y)
        prewit = np.hypot(prewitx, prewity)
        prewit[prewit < 5] = 0
        return prewit

    def applyLaplacian(self, image: cv2.Mat, scale:int=1):
        lap = cv2.Laplacian(image, cv2.CV_64F)*scale
        lap = np.uint8(np.absolute(lap))
        return lap

    def applyCanny(self, image: cv2.Mat, threshold1:int=50, threshold2:int=150):
        return cv2.Canny(image, threshold1=threshold1, threshold2=threshold2)

    def applyRoberts(self, image: cv2.Mat, scale:int=1):
        gx_c = np.array([[1,0],[0,-1]])*scale
        gy_c = np.array([[0,1],[-1,0]])*scale
        g_x = cv2.filter2D(image, -1, gx_c)
        g_y = cv2.filter2D(image, -1, gy_c)
        g = np.hypot(g_x, g_y)
        roberts = g.astype(np.uint8)
        return roberts

    def applyCustomFilter(self, image: cv2.Mat,filter: cv2.Mat, scale:int=1):
        if filter is None:
            filter = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) * scale
        filt = cv2.filter2D(image, -1, filter)*scale
        return filt.astype(np.uint8)


    def applyFilter(self, image: cv2.Mat,custom_filter: cv2.Mat = None):
        self.gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.blur_img = cv2.bilateralFilter(self.gray_img,9,75,75)
        self.canny_image = self.applyCanny(self.gray_img, 50, 150)
        self.sobel_image = self.applySobel(self.gray_img, 3, 1)
        self.prewitt_image = self.applyPrewitt(self.gray_img, 3, 1.5)
        self.laplacian_image = self.applyLaplacian(self.gray_img, 1)
        self.roberts_image = self.applyRoberts(self.gray_img, 1)
        self.custom_image = self.applyCustomFilter(self.gray_img, custom_filter, 1)

    def setBB(self,coordinates: np.array):
        self.bb = coordinates.reshape(-1,2).astype(np.int32)

    def getBB(self):
        return self.bb

    def generateMask(self, image: cv2.Mat):
        match self.filterChoice:
            case filterChoice.sobel:
                self.input_mask_img = self.sobel_image
            case filterChoice.prewitt:
                self.input_mask_img = self.prewitt_image
            case filterChoice.laplacian:
                self.input_mask_img = self.laplacian_image
            case filterChoice.canny:
                self.input_mask_img = self.canny_image
            case filterChoice.roberts:
                self.input_mask_img = self.roberts_image
            case filterChoice.custom:
                self.input_mask_img = self.custom_image
            case _:
                raise ValueError("Invalid filter choice")
        
        mask = np.zeros_like(self.input_mask_img[:,:], dtype=np.uint8)
        cv2.drawContours(mask, [self.bb], -1, (255), -1)
        self.masked_image_hold = cv2.bitwise_and(self.input_mask_img, self.input_mask_img, mask=mask)
        self.mask_image_ground = cv2.bitwise_xor(self.input_mask_img, self.masked_image_hold)
        mask_t = 60
        self.masked_image_hold[self.masked_image_hold < mask_t] = 0
        self.masked_image_hold[self.masked_image_hold > mask_t] = 255
        self.mask_image_ground[self.mask_image_ground < mask_t] = 0
        self.mask_image_ground[self.mask_image_ground > mask_t] = 255
        return self.masked_image_hold, self.mask_image_ground

    def applyThreshold(self, image: cv2.Mat, threshold:int):
        image[image < threshold] = 0
        image[image >= threshold] = 255
        return image
    
    def elements(self,array):
        return array.ndim and array.size
    


    def PCA(self,image: cv2.Mat,block_size: int,line_threshold: int = 0.3):
        # self.showImage(image.astype(np.uint8))
        shape_h, shape_w = image.shape
        self.block_size = block_size
        self.line_threshold = line_threshold
        self.block_h = shape_h // self.block_size #num of blocks in height
        self.block_w = shape_w // self.block_size #num of blocks in width
        self.block = np.zeros((self.block_h, self.block_w, self.block_size, self.block_size)) # block matrix
        self.lines = np.zeros((self.block_h, self.block_w, 2)) # line matrix
        for i in range(self.block_h):
            for j in range(self.block_w):
                self.block[i, j] = image[i*self.block_size:(i+1)*self.block_size, j*self.block_size:(j+1)*self.block_size]
        #print("block valid", np.isnan(self.block).any() == False)
        #print("block debug:" ,(np.argwhere(self.block > 100)).shape)
        self.block_cov = np.zeros((self.block_h, self.block_w,2,2)) # covariance matrix for all blocks
        for i in range(self.block_h):
            for j in range(self.block_w):
                x = np.argwhere(self.block[i, j,:,:] >100)

                block_weight = []
                for k in range(x.shape[0]):
                    block_weight.append((self.block[i, j,x[k,0],x[k,1]]) ** 2  / sum((self.block[i, j,:,:].flatten()) ** 2))

                if len(x) < sqrt(2)*self.block_size or len(x) == 0:
                    # #print("put nan in: ",(i,j))
                    self.block_cov[i,j] = [[nan,nan],[nan,nan]]
                else:
                    temp =  np.cov(x.T,aweights=block_weight)
                    # assert(x.shape[0] == len(block_weight))
                    cov_temp = temp 
                    # #print("cov_temp: ",cov_temp)
                    if np.isnan(cov_temp).any():
                        self.block_cov[i,j] = [[nan,nan],[nan,nan]]
                        #print("nan")
                    else:
                        self.block_cov[i,j] = cov_temp
                        # #print("put vals in: ",(i,j))

        self.lines = np.zeros((self.block_h, self.block_w,2))

        # calculate eigen values and eigen vectors for each block and sort them in descending order
        for i in range(self.block_h):
            for j in range(self.block_w):
                if np.isnan(self.block_cov[i,j]).any():
                    self.lines[i,j] = nan
                    continue
                eig_val,eig_vec = np.linalg.eig(self.block_cov[i,j])
                idx = eig_val.argsort()[::-1]
                eig_val,eig_vec = eig_val[idx],eig_vec[:,idx]
                if eig_val[0]  == 0.0:  
                    self.lines[i,j] = nan
                elif eig_val[1] / eig_val[0] < line_threshold:
                    self.lines[i,j]  = eig_vec[0]/np.linalg.norm(eig_vec[0])
                    # #print("line: ",self.lines[i,j])
                    # while atan2(lines[i,j,1],lines[i,j,0]) < 0:
                    #     lines[i,j] = np.matmul([[-1,0],[0,-1]],lines[i,j])
                else:
                    self.lines[i,j] = nan
        


    def retPoints(self,i,j):
        if np.isnan(self.lines[i,j,0]) or np.isnan(self.lines[i,j,1]):
            return None,None,None,None
        center = (j*self.block_size + self.block_size//2, i*self.block_size + self.block_size//2)
        if self.elements(self.lines[i,j]):
            [x1,y1] = [int(center[0] - self.block_size*self.lines[i,j,1]), int(center[1] -self.block_size*self.lines[i,j,0])]
            [x2,y2] = [int(center[0] + self.block_size*self.lines[i,j,1]), int(center[1] + self.block_size*self.lines[i,j,0])]
        return [x1,y1,x2,y2]

    def drawPCAResults(self):
        temp_image = self.input_img.copy()
        temp_filter = self.input_mask_img.copy()
        nan_count = 0
        for i in range(self.block_h):
            for j in range(self.block_w):
                if np.isnan(self.lines[i,j,0]) or np.isnan(self.lines[i,j,1]):
                   nan_count += 1
                   continue
                #print("line: ",self.lines[i,j])
                center = (j*self.block_size + self.block_size//2, i*self.block_size + self.block_size//2)
                if self.elements(self.lines[i,j]):
                    [x1,y1] = [int(center[0] - self.block_size*self.lines[i,j,1]), int(center[1] -self.block_size*self.lines[i,j,0])]
                    [x2,y2] = [int(center[0] + self.block_size*self.lines[i,j,1]), int(center[1] + self.block_size*self.lines[i,j,0])]
                if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                    cv2.line(
                        temp_image, (x1, y1), (x2, y2),
                                (255,0,0), 5
                        )
                    cv2.line(
                        temp_filter, (x1, y1), (x2, y2),
                                (255,0,0), 7
                        )
                else:
                    nan_count += 1
        #print("nan_count: ",nan_count)
        #print("valid: ",len(self.lines) - nan_count)

        self.showImage(temp_image)

    def outlierFilter(self):
        """
        This function filters out the outliers from the lines detected by PCA
        [[deprecated for now]]
        """
        self.angle_lines = []
        self.lines_to_hist = []
        indx =[]

        reshaped_lines = self.lines.reshape(-1,2)
        reshaped_angles = np.arctan2(reshaped_lines[:,1],reshaped_lines[:,0])
        reshaped_angles = reshaped_angles[~np.isnan(reshaped_angles)]
        # #print(reshaped_angles.shape)
        # #print(reshaped_angles)
        for ind,line in enumerate(reshaped_angles):
            if  not np.isnan(line):
                if line > np.pi/2:
                    line -= np.pi/2
                self.lines_to_hist.append(line)
                indx.append(ind)

        kmeans = KMeans(n_clusters=2)
        kmeans.fit(np.array(self.lines_to_hist).reshape(-1,1))
        y_kmeans = kmeans.predict(np.array(self.lines_to_hist).reshape(-1,1))
        self.centers = kmeans.cluster_centers_
        #print("centers before outliers removal",self.centers)
        #print("centers_diff ",self.centers[1]-self.centers[0],"rad, ",(self.centers[1]-self.centers[0])*180/pi,"deg")
        for line in self.lines_to_hist:
            if min(abs(line - self.centers)) > 0.1:
                self.lines_to_hist.remove(line)


        kmeans.fit(np.array(self.lines_to_hist).reshape(-1,1))
        y_kmeans = kmeans.predict(np.array(self.lines_to_hist).reshape(-1,1))
        self.centers = kmeans.cluster_centers_
        #print("centers after outlier removal",self.centers)
        #print("centers_diff ",self.centers[1]-self.centers[0],"rad, ",(self.centers[1]-self.centers[0])*180/pi,"deg")

    def splitLines(self):
        self.ang1_lines = []
        self.ang2_lines = []
        self.ang1_img = np.zeros_like(self.masked_image_hold)
        self.ang2_img = np.zeros_like(self.masked_image_hold)

        for i in range(self.lines.shape[0]):
            for j in range(self.lines.shape[1]):
                x1,y1,x2,y2 = self.retPoints(i,j)
                if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                    angle = atan2(self.lines[i,j,1],self.lines[i,j,0])
                    if abs(angle - self.centers[0]) < abs(angle - self.centers[1]):
                        self.ang1_lines.append(self.lines[i,j])
                        cv2.line(
                            self.ang1_img, (x1, y1), (x2, y2),
                                    (255,0,0), 5
                            )
                    else:
                        self.ang2_lines.append(self.lines[i,j])
                        cv2.line(
                            self.ang2_img, (x1, y1), (x2, y2),
                                    (255,0,0), 5
                            )
        self.showImage(self.ang1_img.astype(np.uint8))
        self.showImage(self.ang2_img.astype(np.uint8))

    def closestPointsFilter(self,Dots,rad: int = 100):
        
        kmeans_lin = KMeans(n_clusters=2)
        #print(len(Dots))
        tree = cKDTree(Dots)

        visited = set()
        components = []

        for i, point in enumerate(Dots):
            if i not in visited:
                component = set()
                stack = [i]

                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.add(current)
                        neighbors = tree.query_ball_point(Dots[current], 100)
                        stack.extend(neighbors)

                components.append(list(component))

        components.sort(key=len, reverse=True)
        comps = np.array(Dots[components[0]])
        return components,comps
    
    def linearFit(self,mat: cv2.Mat):
        Dots = np.argwhere(mat > 0)
        components,comps = self.closestPointsFilter(Dots,100)
        count = len(comps)
        #print(count)
        s_xy = 0
        s_x = 0
        s_xx = 0
        s_y = 0
        s_yy =0
        for line in comps:
            s_xy += line[0]*line[1]
            s_x += line[0]
            s_y += line[1]
            s_xx += line[0]**2
            s_yy += line[1]**2
        #print("sum1_xy: ",s_xy)
        #print("sum1_x: ",s_x)
        #print("sum1_y: ",s_y)
        #print("sum1_xx: ",s_xx)
        #print("sum1_yy: ",s_yy)
        slope = (count * s_xy - s_x * s_y) / (count * s_xx - s_x**2)
        Yint = (s_y - slope * s_x) / count
        #print("slope1: ",slope,"Yint1: ",Yint)
        # plt.scatter([line[0] for line in comps], [line[1] for line in comps], s=1)
        x_min = min([line[0] for line in comps])
        x_max = max([line[0] for line in comps])
        y_min = Yint + slope * x_min
        y_max = Yint + slope * x_max

        return slope,Yint
    
    def drawLinearFit(self,based_on: cv2.Mat,backround: cv2.Mat, slope: float):
        #print("debug 1")
        cords_init = np.argwhere(based_on > 0)[len(np.argwhere(based_on > 0))//2]
        #print("debug 2")
        temp_out = backround.copy()
        #print("debug 3")
        temp_out = cv2.circle(temp_out, (cords_init[1], cords_init[0]), 10, (255,0,0), -1)
        #print("debug 4")
        cv_slope = slope
        c2_x2 = int(cords_init[1] + 2000*np.sin(atan(cv_slope)))
        c2_y2 = int(cords_init[0] + 2000*np.cos(atan(cv_slope)))
        c3_x2 = int(cords_init[1] - 2000*np.sin(atan(cv_slope)))
        c3_y2 = int(cords_init[0] - 2000*np.cos(atan(cv_slope)))
        #print("debug 5")
        temp_out = cv2.line(temp_out, (c2_x2, c2_y2), (c3_x2, c3_y2), (255,0,0), 5)
        #print("debug 6")
        self.showImage(temp_out)
        #print("debug 7")
        return c2_x2,c2_y2,c3_x2,c3_y2
    
    def drawLinearFit2(self,background: cv2.Mat,l1,l2):
        temp_out = background.copy()
        temp_out = cv2.line(temp_out, (l1[0], l1[1]), (l1[2], l1[3]), (255,0,0), 5)
        temp_out = cv2.line(temp_out, (l2[0], l2[1]), (l2[2], l2[3]), (255,0,0), 5)
        self.showImage(temp_out)
        return temp_out



def main():
    io = imageIO()
    io.setReadMethod(readMethod.file_path)
    io.setFilterChoice(filterChoice.prewitt)
    input_img = io.readImage('/home/gilad/dev_ws/camera_dev/images/20240403_170839.jpg')
    io.applyFilter(input_img)
    io.setBB(np.array([0,0,0,1410,1750,1680,2025,0]))
    held,ground = io.generateMask(input_img)
    # io.showImage(held.astype(np.uint8))
    io.PCA(io.masked_image_hold,block_size=10,line_threshold=0.3)
    io.drawPCAResults()
    io.outlierFilter()
    io.splitLines()
    slope,yint = io.linearFit(io.ang1_img)
    l1 = io.drawLinearFit(io.ang1_img,io.input_img,slope)
    slope,yint = io.linearFit(io.ang2_img)
    l2 = io.drawLinearFit(io.ang2_img,input_img,slope)
    io.drawLinearFit2(io.input_img,l1,l2)

if __name__ == "__main__":
    main()

