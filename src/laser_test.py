import numpy as np
import cv2
import matplotlib.pyplot as plt 
import math
import sys



class lasersIO():
    def __init__(self):
        self.int_points = []
        pass
    def loadImage(self,path):
        self.input_image = cv2.imread(path)
        # self.input_image = cv2.cvtColor(r_image,cv2.COLOR_BGR2RGB)
    def showImage(self, image):
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)  
        cv2.imshow("output", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def setDividers(self,cords: np.array):
        self.cords = cords.reshape(-1,5,2)
    def getDividers(self):
        return self.cords
    def getDividerShape(self):
        return self.cords.shape
    def maskImage(self):
        self.masks = []
        numofmasks = (self.cords.shape[0])
        for i in range(numofmasks):
            bb = self.cords[i,:,:].reshape(-1,2).astype(np.int32)
            # print("bb for mask ",i,": ",bb," shape: ",bb.shape)
            mask = np.zeros_like(self.input_image[:,:,0],dtype=np.uint8)
            cv2.drawContours(mask, [bb], -1, (255), -1)
            mask_image_i = cv2.bitwise_and(self.input_image, self.input_image, mask=mask)
            self.masks.append(mask_image_i)
    def getMasks(self):
        return self.masks
    def getMask(self,i):
        return self.masks[i]
    def showMasks(self):
        temp_img = self.masks[0]
        for i in range(1,len(self.masks)):
            temp_img = cv2.hconcat([temp_img,self.masks[i]])
        self.showImage(temp_img)
    def showMask(self,i):
         self.showImage(self.masks[i])

    def findComponents(self,mask):
        visited = np.zeros(mask.shape,dtype=bool)
        components = []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i][j] >= 1 and not visited[i][j]:
                    # print('found new component origin at:',i,j,'value:',mask[i][j])
                    component = []
                    component = self.DFS(mask,i,j,visited,component)
                    components.append(component)
        self.components = components
        # print("There are: ",len(self.components)," components")
        return components
    
    def DFS(self,G, i, j, visited, component):
        stack = [(i, j)]
        while stack:
            i, j = stack.pop()
            if not visited[i][j]:
                visited[i][j] = True
                component.append((i, j))
                moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
                for move in moves:
                    i_new = i + move[0]
                    j_new = j + move[1]
                    if 0 <= i_new < G.shape[0] and 0 <= j_new < G.shape[1] and not visited[i_new][j_new] and G[i_new][j_new] >= 1:
                        stack.append([i_new, j_new])
        return component

    def getComponents(self):
        return self.components

    def sortComponents(self,components):
        comps = sorted(components,key=lambda x: len(x),reverse=True)
        if len(comps) >2:
            return comps[:2]
        return comps
    
    def getBiggestComponent(self,components):
        return self.sortComponents(components)[0]
    
    def genColorMask(self):
        self.color_masks = []
        for i in range(len(self.masks)):
            temp_mask = self.masks[i]
            color_mask_i = temp_mask[:,:,0] > 200
            color_mask_i = color_mask_i.astype(np.uint8)
            color_mask_i[color_mask_i > 0] = 255
            self.color_masks.append(color_mask_i)
    def getColormasks(self):
        return self.color_masks
    def showColorMasks(self):
        temp_img = self.color_masks[0]
        for i in range(1,len(self.color_masks)):
            temp_img = cv2.hconcat([temp_img,self.color_masks[i]])
        self.showImage(temp_img)

    def showColorMask(self,i):
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        temp_img = self.color_masks[i]
        temp_img = cv2.cvtColor(temp_img,cv2.COLOR_GRAY2BGR)
        cv2.imshow("output", self.color_masks[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def showImageAndColorMask(self,i):
        temp_img = cv2.hconcat([self.masks[i],self.color_masks[i]])
        self.showImage(temp_img)

    def filterComponents(self,components,based_on):
        for component in components:
            if len(component) < 70:
                components.remove(component)
            for point in component:
                i,j = point
                if i>= self.color_masks[based_on].shape[0] or j>=self.color_masks[based_on].shape[1]:
                    component.remove(point)
                    break
                if len(component) < 70:
                    self.color_masks[based_on][i,j] = 0
        return components   
        # print("there are ",len(self.components)," components")   
    def showComponents(self,components):
        temp_img = self.input_image.copy()
        for k in range(2):
            for point in components[k]:
                i,j = point
                temp_img[i,j] = [0,255 if k == 1 else 0 ,255 if k == 0 else 0]
        self.showImage(temp_img)

    def genComponentMask(self,component):
        mask = np.zeros_like(self.input_image[:,:,0],dtype=np.uint8)
        for point in component:
            i,j = point
            mask[i,j] = 255
        return mask

    def sortComponentsByFeatures(self,components):
        comp_features = []
        for component in components:
            masked = self.genComponentMask(component)
            cnts = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = cnts[0]
            c = max(c, key=cv2.contourArea)
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])
            show_img = self.input_image.copy()
            cv2.circle(show_img,extLeft,3,(0,255,255),-1)
            cv2.circle(show_img,extRight,3,(255,0,255),-1)
            cv2.circle(show_img,extTop,3,(255,255,0),-1)
            cv2.circle(show_img,extBot,3,(0,0,255),-1)
            # self.showImage(show_img)
            comp_features.append({"extLeft":extLeft,"extRight":extRight,"extTop":extTop,"extBot":extBot})

        comp_features = sorted(comp_features,key=lambda x: x["extRight"],reverse=True)
        if(len(comp_features) < 2):
            print("Not enough components")
            raise Exception("Not enough components")
        comp_features = comp_features[:2]
        interest_points = {"leftGapPoint":comp_features[0]["extLeft"],"RightGapPoint":comp_features[1]["extRight"]}
        cv2.circle(show_img,interest_points["leftGapPoint"],3,(0,255,255),-1)
        cv2.circle(show_img,interest_points["RightGapPoint"],3,(255,0,255),-1)
        # self.showImage(show_img)
        self.int_points.append(interest_points)
        return interest_points
    
    def euclideanDistance(self,p1,p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    def linearFit(self):
        set_r = []
        set_l = []
        for i in range(len(self.int_points)):
            set_r.append(self.int_points[i]["RightGapPoint"])
            set_l.append(self.int_points[i]["leftGapPoint"])
        count = len(self.int_points)
        #print(count)
        s_xy_r,s_xy_l = 0,0
        s_x_r,s_x_l = 0,0
        s_xx_r,s_xx_l = 0,0
        s_y_r,s_y_l = 0,0
        s_yy_r,s_yy_l =0,0
        for pointr,pointl in zip(set_r,set_l):
            s_xy_r += pointr[0]*pointr[1]
            s_xy_l += pointl[0]*pointl[1]
            s_x_r += pointr[0]
            s_x_l += pointl[0]
            s_y_r += pointr[1]
            s_y_l += pointl[1]
            s_xx_r += pointr[0]**2
            s_xx_l += pointl[0]**2
            s_yy_r += pointl[1]**2
            s_yy_l += pointl[1]**2
        #print("sum1_xy: ",s_xy)
        #print("sum1_x: ",s_x)
        #print("sum1_y: ",s_y)
        #print("sum1_xx: ",s_xx)
        #print("sum1_yy: ",s_yy)
        slope_r = (count * s_xy_r - s_x_r * s_y_r) / (count * s_xx_r - s_x_r**2)
        Yint_r = (s_y_r - slope_r * s_x_r) / count
        slope_l = (count * s_xy_l - s_x_l * s_y_l) / (count * s_xx_l - s_x_l**2)
        Yint_l = (s_y_l - slope_l * s_x_l) / count
        print("slope_r: ",slope_r," Yint_r: ",Yint_r)
        print("slope_l: ",slope_l," Yint_l: ",Yint_l)
        dr = self.euclideanDistance(self.int_points[0]["RightGapPoint"],self.int_points[-1]["RightGapPoint"])
        dl = self.euclideanDistance(self.int_points[0]["leftGapPoint"],self.int_points[-1]["leftGapPoint"]) 
        s_point_r = self.int_points[-1]["RightGapPoint"]
        s_point_l = self.int_points[-1]["leftGapPoint"]
        e_point_r = (int(s_point_r[0] - dr * math.cos(math.atan(slope_r)) ),int(s_point_r[1] - dr * math.sin(math.atan(slope_r))))
        e_point_l = (int(s_point_l[0] - dl * math.cos(math.atan(slope_l)) ),int(s_point_l[1] - dl * math.sin(math.atan(slope_l))))
        print("s_point_r: ",s_point_r," e_point_r: ",e_point_r)
        print("s_point_l: ",s_point_l," e_point_l: ",e_point_l)
        temp_img = self.input_image.copy()
        cv2.line(temp_img,s_point_r,e_point_r,(100,125,200),2)
        cv2.line(temp_img,s_point_l,e_point_l,(200,70,10),2)
        self.showImage(temp_img)
        
def main():
    lasers = lasersIO()
    lasers.loadImage('/home/gilad/dev_ws/RobeeVisionDevel/images/14.4.24/1_Color.png')
    # lasers.showImage(lasers.input_image)
    lasers.setDividers(np.array(
            [
                [
                    [0,0],
                    [0,335],
                    [400,0],
                    [0,0],
                    [0,0],
                ],
                [   
                    [0,335],
                    [0,475],
                    [120,475],
                    [715,0],
                    [400,0],
                ],
                [
                    [120,475],
                    [535,475],
                    [847,220],
                    [715,0],
                    [715,0],
                ],
            ],dtype=np.int32
    ))
    # print(lasers.getDividers())
    print(lasers.getDividerShape())
    lasers.maskImage()
    lasers.genColorMask()

    # lasers.showColorMask(0)
    comps0 = lasers.findComponents(lasers.color_masks[0])
    filt0 = lasers.filterComponents(comps0,0)
    sort0 = lasers.sortComponents(filt0)
    int_point0_ = lasers.sortComponentsByFeatures(sort0)
    # lasers.showComponents(sort0)

    # lasers.showColorMask(1)
    comps1 = lasers.findComponents(lasers.color_masks[1])
    filt1 = lasers.filterComponents(comps1,1)
    sort1 = lasers.sortComponents(filt1)
    int_point1_ = lasers.sortComponentsByFeatures(sort1)
    # lasers.showComponents(sort1)

    # lasers.showColorMask(2)
    comps2 = lasers.findComponents(lasers.color_masks[2])
    filt2 = lasers.filterComponents(comps2,2)
    sort2 = lasers.sortComponents(filt2)
    int_point2_ = lasers.sortComponentsByFeatures(sort2)
    # lasers.showComponents(sort2)
    lasers.linearFit()
    
if __name__ == "__main__":
    main()

