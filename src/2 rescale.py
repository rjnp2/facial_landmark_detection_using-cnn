#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 21:26:53 2020

@author: rjn
"""

# working with mainly reading, convert to grayscale, resizing images
import cv2
# dealing with arrays                  
import numpy as np  
#reading csv file and datframe
import pandas as pd
# dealing with directories         
import os 

class Rescaling:    
    
    '''
    This class take .csv landmark files from landmark directory,
    extract the given landmarks with their names,
    detect face and crop region of interest and rescale landmark using prepare() method,
    save landmark csv files in given directory using save_landmark() method,
    take landmark files using get_landmark()
    '''
    
    def __init__(self, image_dir, dimensions,face_cascade):
        
        self.image_dir = image_dir
        self.dimensions = dimensions
        self.face_cascade = face_cascade
        
    def prepare(self,landmark,final_dir,land_dir):
                
        shape = landmark.shape
        col = landmark.columns

        self.fland = pd.DataFrame(columns = col )
        self.finished = []
        self.out = []
        
        for f in range(shape[0]):
    
            data = list(landmark.iloc[f,:].values)
            image_name = data[0]
            val = [max(data[1::2]),max(data[2::2])]
            
            try:
                path = os.path.join(self.image_dir, image_name)
                saved = os.path.join(final_dir, image_name)
                land = os.path.join(land_dir, image_name)

                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                         
                faces = self.face_cascade.detectMultiScale(img, 1.15, 8)
                (x,y,w,h) = faces[0]
                h += 10
                w += 10
                x -= 5
                y -= 5
                    
                if w <= (val[0] - x):
                    w = int(val[0] - x + 10)
                
                if h <= (val[1] - y ):
                    h = int(val[1] - y + 10)
                        
                just_face = cv2.resize(img[y:y+h,x:x+w], self.dimensions)
                cv2.imwrite(saved,just_face)
                               
                data[1::2] = (data[1::2] - x) / (w/self.dimensions)
                data[2::2] = (data[2::2] - y) / (h/self.dimensions) 
                
                self.fland.loc[f] = data
                
                for i in range(shape[1]):
                    cv2.circle(just_face, (int(data[1::2][i]),int(data[2::2][i])), 1, (0, 0, 255), -1)
        
                cv2.imwrite(land,just_face)
                
                self.finished.append(image_name)
                
            except:
                self.out.append(image_name) 
            
            else:
                print("work completed")
                
    def save_landmark(self,csv_name = 'final_landmark'):
        '''
        this method save self.landmark as csv_name.csv files
        or
        csv_name if none , it save as final_landmark.csv files
        '''
        
        name = str(csv_name) + '.csv'
        self.fland.to_csv(name,index= False)
        print("sucessful")
        
    def get_landmark(self):
        '''
        get landmark 
        '''
        return self.fland

    def sucess_list(self):
        
        return self.finished
    
    def out_list(self):
        
        return  self.out
