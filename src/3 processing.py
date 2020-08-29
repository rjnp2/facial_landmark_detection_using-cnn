#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 19:10:30 2020

@author: rjn
"""

# working with mainly reading, convert to grayscale, resizing images
import cv2
# dealing with arrays                  
import numpy as np  
import pandas as pd
# dealing with directories         
import os  
from txt_to_csv import LandmarkData
from rescale import Rescaling

#path of animals datasets
image_dir = '/home/rjn/project/UTKface_68/UTKFace'
landmark_dir = '/home/rjn/project/landmark'
final_dir = '/home/rjn/project/UTKface_68/final'
land_dir = '/home/rjn/project/UTKface_68/land'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# image size for resizing the images i.e 200 * 200
dimensions = (200, 200)

class CreateData:
    
    def __init__(self,image_dir,landmark):
        
        self.image_dir = image_dir
        self.landmark = landmark
        self.train =[]
        self.land =[]
        
    def traindata(self,train_name,land_name):
        
        shape = self.landmark.shape
        
        for f in range(shape[0]):
                       
            data = list(self.landmark.iloc[f,:].values)
            image_name = data[0]
        
            path = os.path.join(self.image_dir, image_name)
            
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
            self.train.append(img)
            self.land.append(data[1:])
     

        self.train = np.array(self.train)
        self.land = np.array(self.land)
        
        train_name = train_name + '.npy'
        land_name = land_name + '.npy'
        np.save(train_name, self.train)
        np.save(land_name, self.land)
        
    def get_train(self):
        
        return self.train
    
    def get_land(self):
        
        return self.land 

class1 = LandmarkData(land_dir=landmark_dir, landmark_point=68)
class1.transform()
class1.save_csv()
landmark = class1.get_landmark()

class2 = Rescaling(image_dir=image_dir,dimensions=dimensions,face_cascade=face_cascade)
class2.prepare(landmark= landmark, final_dir=final_dir,land_dir = land_dir)
class2.save_landmark()
landmark = class2.get_landmark()

class3 = CreateData(image_dir,landmark)
class3.traindata(train_name = 'train',land_name = 'land')