#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 15:45:46 2019

@author: rjn
"""

#importing all neccessary packages
import numpy as np
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

#initialize all variables
train_data = []
test_data = []
train_label = []
test_label = []

epochs = 100
batch_size = 512

class createCNNModel:
    '''
    to create cnn model
    using built-in functions of keras such as
    conv2D,
    maxpooling2D,
    flatten,
    dropout
    '''

    def __init__(self,input_size,output_size,epochs,batch_size):
        
        self.input_size = input_shape
        self.output_size = output_size
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.createmodel()


    def createmodel(self):

        #creating a Neural network is to initialise the network using the Sequential Class from keras.
        self.model = Sequential()
        
        # The first two layers with 64 filters of window size 3x3
        # filters : Denotes the number of Feature detectors
        # kernel_size : Denotes the shape of the feature detector. (3,3) denotes a 3 x 3 matrix.
        # input _shape : standardises the size of the input image
        # activation : Activation function to break the linearity
        self.model.add(Conv2D(filters = 1, kernel_size=(1, 1), padding='same',activation='relu', 
                         input_shape= self.input_shape))
        self.model.add(Conv2D(filters = 64, kernel_size=(3, 3),padding='same', activation='relu'))
    
        # pool_size : the shape of the pooling window.
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
    
        self.model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(0.2))
        
        self.model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(0.2))
        
        self.model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(0.2))
        
        self.model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(0.2))
        
        self.model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(0.2))
    
        self.model.add(Flatten())

        # units: Number of nodes in the layer.
        # activation : the activation function in each node.
        self.model.add(Dense(units = 512, activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(units = 136,name='preds'))
    
    def summary(self):
        
        return self.model.summary()
    
    def compile(self,X_train, X_test, y_train, y_test,save= True):
        
        # Optimiser: the Optimiser  used to reduce the cost calculated by categorical_crossentropy
        # Loss: the loss function used to calculate the error
        # Metrics: the metrics used to represent the efficiency of the model
        # lr : learing rates
        
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        
        self.model.compile(optimizer = adam,loss = 'mean_squared_error', metrics = ['acc'])
        self.model.fit(x = X_train, y= y_train,batch_size= 512, 
                       verbose=1, 
                       epochs=100,
                       validation_data= (X_test,y_test))
        if save:
            self.save()

    def save(self):
        
        # creates a HDF5 fil 'my_model.h5'
        self.model.save('my_model.h5')  
        print("saved sucessfully")
        
    def save_history(self):
        
        # for visualizing losses and accuracy
        # History.history attribute is a record of training loss values
        # metrics values at successive epochs 
        # as well as validation loss values and validation metrics values
        train_loss=self.model.history['loss']
        val_loss=self.model.history['val_loss']
        train_acc=self.model.history['acc']
        val_acc=self.model.history['val_acc']
        xc=range(self.epochs)
        model_histroy = [
                train_loss,
                val_loss,
                train_acc,
                val_acc,
                xc]
        
        np.save('model_histroy.npy',model_histroy)

#loading train_data and test_data 
train_data = np.load('/home/rjnp2/minorproject/image_array_data/train.npy')
land_data = np.load('./image_array_data/land.npy')  

#pre-processing all data   
train_data = np.array(train_data).reshape(-1, 200,200,1).astype('float32')

#converting int datasets into float datasets 
# for easy processing by CPU/GPU
train_data /= 255
x= np.median(ytrain)
ytrain = (ytrain - x)/ x

#saving rows,col,dim of datasets to input_shape variable
#to give input size /shape to CNN model 
input_shape = train_data.shape[1:]

#Split arrays or matrices into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(train_data, land_data, test_size=0.2, random_state=42)

model = createCNNModel()
model.summary()
model.compile(X_train, X_test, y_train, y_test)
model.save_history()