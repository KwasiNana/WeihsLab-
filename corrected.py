#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 22:23:32 2023

@author: nanao.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 18:04:13 2023

@author: nanao.
"""

import numpy as np
from xml.dom import minidom
import cv2 as cv  
import os 
import matplotlib.pyplot as plt 
from math import dist


class Pitscan():
    def __init__(self, foldername = "", xml_file = ""):
        self.foldername = foldername
        self.xml_file = xml_file
        
    def read_folder(self):
        images = []
        for file in os.listdir(foldername)[1:]:
            img_path = os.path.join(self.foldername, file)
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            images.append(img)
        return images 
    
    def black_white(self, image):
        thresh,binary = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        return binary
    
    def get_voxel_size(self, xml_file):
        # applies to only xml files within the Pitscan 
        self.xml_file = xml_file
        file = minidom.parse(self.xml_file)
        #use getElementsByTagName() to get tag
        models = file.getElementsByTagName('voxelSize')
        # voxel lengths in mm
        x=float(models[0].attributes['X'].value) 
        y=float(models[0].attributes['Y'].value) 
        z=float(models[0].attributes['Z'].value)
        return x*y*z
    
    def get_volume(self, voxel_file):
        voxel_size = voxel_file.get_voxel_size()
        # applies only to folders containing tiff files 
        volume = []
        for file in os.listdir(self.foldername)[1:]:
            ct_slice = os.path.join(self.foldername, file)
            ct_array = cv.imread(ct_slice, cv.IMREAD_GRAYSCALE)
            binarize_image = self.black_white(ct_array)
            volume.append(cv.countNonZero(binarize_image)*voxel_size)
        return np.sum(np.array(volume))
    
    def mask_img(self, contours, hierarchy): 
        max_length = 0 
        position = 0 
        for pos, contour in enumerate(contours): 
            if len(contour) > max_length:
                max_length = len(contour)
                position = pos
        #second_level = contours[hierarchy[0][position][2]]
        #[x,y], radius = cv.minEnclosingCircle(contours[position])
      
        #for indices ,point in enumerate(second_level):
            #if dist([x,y],point[0]) > radius: 
               #np.delete(second_level, 0)
        
        return contours[position]
            
            
    def radii_lost(self, contours):
        [x,y], radius = cv.minEnclosingCircle(contours)
        diff = []
        for vertex in contours:
            loss = radius - dist(vertex[0][0],[x,y]) 
            diff.append(loss)
        return np.average(np.ndarray(diff))
    
    def radial_contour(self, img): 
        slice_file = cv.imread(img, 0)
        binarize_slice = self.black_white(slice_file)
        contour,_ = cv.findContours(binarize_slice, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        return self.mask_img(contour, _)
    
    def detect_pit(self, img_path):
        
        img_contour = self.radial_contour(img_path)
        color_img = cv.imread(img_path, cv.IMREAD_COLOR)
        smooth_img = cv.GaussianBlur(color_img, (7,7), 0)
        (x,y),radius = cv.minEnclosingCircle(img_contour)
        r_not = int(radius) 
        img_center = [int(x),int(y)]
        
        print(r_not)
        
        while True:
            count = 0
            for point in img_contour:
                length =  int(dist(img_center, point[0]))
                if length == r_not: 
                    count += 1
                    print(length)
                    print(r_not)
                    if count == 0: 
                        print(f"length at count zero {length}")
                        print(f"radius at count zero {r_not}")
                        break 
            #ratio = (2*np.pi*r_not)/count
            r_not -= 1 
            #if ratio <= 0.2 :
                #break 
      
            
        
        
        

    

    def pit_count(self):
        # to account for slices in ct folder
        count = 0
        contours = []
        for file in os.listdir(self.foldername)[1:]:
            img_path = os.path.join(self.foldername, file)
            self.detect_pit(img_path)
            #actual_con = self.radial_contour(img_path)
            #contours.append(actual_con)
            #(x,y), radius = cv.minEnclosingCircle(actual_con)
            break
    
            
       
        
       
        """
        color_img = cv.imread(img_path, cv.IMREAD_COLOR)
        cv.drawContours(color_img, contours[0], -1 , (0,255,0), 3)
        #(x,y), radius = cv.minEnclosingCircle(contours[0])
        #center = (int(x), int(y))
        #radius = int(radius)
        #cv.circle(color_img, center, radius, (0, 0, 255), 2)
        cv.imshow("Pit Contours",cv.GaussianBlur(color_img, (7,7), 0))
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        
       
        
        
        def pit_detection_2d(image, center_x, center_y, r0):

            contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            max_contour = max(contours, key=cv.contourArea)

            

        
            while True:
            
                smoothed_contour = cv.GaussianBlur(max_contour, )

           
                area_contour, _ = cv.contourArea(smoothed_contour)
                ratio = area_contour / (np.pi * (radius ** 2))

                if ratio <= 0.2:
                    break

                radius -= 1 

       
                radius_loss = radius / r0

       
                di = []
                for angle in range(0, 181, 2):
            # Calculate radial distances to the contour for each angle
            # Append the values to the list 'di'
            # di.append(calculated_value)

            return di, radius_loss
        
        
        
        """   
    

        
if __name__ == "__main__": 
    foldername = "/Users/nanao./Desktop/Spyder/images" 
    image_list = Pitscan(foldername)
    image_list.pit_count()
    
    
    
        
    
    
    

    

            