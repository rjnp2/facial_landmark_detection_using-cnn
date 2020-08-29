#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:28:38 2020

@author: rjn
"""

# dealing with directories         
import os
#working with DataFrame
import pandas as pd

class LandmarkData:
    '''
    This class take .txt landmark files from landmark directory,
    extract the given landmarks with their names 
    save in landmark csv files using transform() method,
    save landmark csv files in given directory using save() method,
    take csv files using get_csv()
    '''
    
    def __init__(self,land_dir,landmark_point):
        
        self.land_dir = land_dir
        self.landmark_point = landmark_point
        
        self.landmark = self.create_frame()
        
    def create_frame(self):
        '''
        create landmark DataFrame with given landmark point
        first columns is image name and rest are landmark locations
        and return landmark DataFrame
        '''
        columns = []
        for i in range(self.landmark_point +1):            
            if i == 0:
                columns.append("image_name")      
            else:
                x = "X_" + str(i)
                y = "Y_" + str(i)
                columns.append(x)
                columns.append(y)
                
        landmark = pd.DataFrame(columns = columns)
        return landmark

    def transform(self):
        '''
        It is mains method where .txt landmark
        is placed in landmark DataFrame.
        so it easy to insert, remove and update landmark points
        '''
        path = os.listdir(self.land_dir)
        path = [ os.path.join(self.land_dir, i) for i in path]
        f = 0        
        try:
            for file in path:
                print(file[-22:], end=' ')
                with open(file) as ld:
                    data = ld.read()
                data = data.split()
                
                while len(data) != 0:
                    data1 = data[0:137]
                    data1[1:] = [ float(x) for x in data1[1:] if x != ',']
                    self.landmark.loc[f] = data1
                    f += 1
                    data = data[137:]
                    
                print("is completed")
        
        except OSError:
            print('cannot open', file)
        except ValueError:
            print("Could not convert data to an float.")
        else:
            print(f"There are {f} files in landmark")                 
            
    def save_csv(self,csv_name = 'landmark'):
        '''
        this method save self.landmark as csv_name.csv files
        or
        csv_name if none , it save as landmark.csv files
        '''
        name = str(csv_name) + '.csv'
        self.landmark.to_csv(name,index= False)
        
    def get_landmark(self):
        '''
        get landmark 
        '''
        return self.landmark   
