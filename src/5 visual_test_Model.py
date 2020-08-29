#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 18:44:05 2019

@author: rjnp2
"""

#importing all neccessary packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#loading model and test_data 
model_histroy_load = np.load('./model_histroy.npy',allow_pickle = True)

train_loss = model_histroy_load[0,:]
val_loss = model_histroy_load[1,:]
train_acc = model_histroy_load[2,:]
val_acc = model_histroy_load[3,:]
epochs = model_histroy_load[4,:]

#plotting training and validation loss
plt.plot(epochs, train_loss, color='red',linewidth=4, label='Training loss')
plt.plot(epochs, val_loss, color='green', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#plotting training and validation accuracy
plt.plot(epochs, train_acc, color='red',linewidth=4, label='Training acc')
plt.plot(epochs, val_acc, color='green', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()