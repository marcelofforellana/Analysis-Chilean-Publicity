# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 22:55:57 2021

@author: marce
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 03:05:04 2021

@author: Marcelo Orellana (marceorellana@alumnos.uai.cl)

Title: Image Processing to get the ads on image(LUN)

"""

#webp to image

from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
import pandas as pd
import os
import shutil

rootdir = r'D:\Scraping_Images'
save_folder=r'D:\DIP'

for subdir, dirs, files in os.walk(rootdir):
    
    for file in files:
        
        filepath = subdir + os.sep + file
        
        spliting = file[0:-4].split(sep="-")
        
        if len(spliting[-1])>=3:
            shutil.copy(filepath, save_folder)
        
        else: 
            
            path_image= os.path.join(subdir, file)
            image = cv2.imread(path_image)
            original = image.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            canny = cv2.Canny(blurred, 120, 255, 1)
            kernel = np.ones((5,5),np.uint8)
            dilate = cv2.dilate(canny, kernel, iterations=1)
            
            # Find contours
            cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            
            aux=[0,0]
            image_number=0
            shapes=list()
            mult_shapes=list()
            
            
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
                ROI = original[y:y+h, x:x+w]
                shapes.append(ROI.shape[:-1])
                image_number+=1
            
            mult_shapes=list()
            
            for i in shapes:
                print (i)
                # Getting the areas of contours
                mult_shapes.append(i[0]*i[1])
                
            index = mult_shapes.index(max(mult_shapes))
            
            # Extract the max contour
            max_contour= cnts[index]
            
            x,y,w,h = cv2.boundingRect(max_contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
            ROI = original[y:y+h, x:x+w]
            shapes.append(ROI.shape[:-1])
            
            
            cv2.imwrite((save_folder+os.sep+file), ROI)
               
            # print(index)
            
            # cv2.imshow('canny', canny)
            # cv2.imshow('image', image)
            # cv2.waitKey(0)

