#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:00:00 2020

@author: dlppdl
"""

#Importing necessary modules
from tkinter import Tk,Label,Button,Frame,LEFT,RIGHT,TOP,BOTTOM,Canvas,PhotoImage,Menu,NW
from PIL import ImageTk, Image
import cv2
import numpy as np
import re
from itertools import count
from detector import image

#Parent window title, icon, size
root=Tk() 
root.title("")
root.call('wm', 'iconphoto', root._w, PhotoImage(file='icon/icon.png'))
root.geometry("400x580")

#Key 'Escape' destroy the window
root.bind('<Escape>', lambda e: root.destroy())

#Disabling the resizable cababilty horizontally and vertically
root.resizable(False,False)

a,b = 0,1

# Title class is where the title of GUI and homepage button goes
class Title:    
    def __new__(cls):

        cls.title_frame = Frame(root, bd = 0 , relief = "raise")
        cls.title_frame.pack(side = TOP)
        
        cls.home_icon = Icon( "icon/home.png")
        cls.home_button = Button(cls.title_frame, image = cls.home_icon,
                                 command = Home, width = 70, height = 50,bd = 0)
        cls.home_button.pack(side = LEFT)
        HoverText(cls.home_button, "Home")
        
        cls.title = Image.open("icon/title.png")
        cls.title = ImageTk.PhotoImage(cls.title)
        cls.title_label = Label(cls.title_frame, image = cls.title)
        cls.title_label.image = cls.title 
        cls.title_label.pack(side = RIGHT)

#Class Icon resize the given icon file and return the resized photo
class Icon:
    def __new__(cls, file):
        cls.icon = PhotoImage(file = file)   
        cls.icon = cls.icon.subsample(3,3)
        return cls.icon

# This class gets all the frames of GIF one by one and show them continuously looks like GIF play
class PlayGIF(Label):
    def load(self, gif):
        if isinstance(gif, str):
            gif = Image.open(gif)
        self.loc = 0
        self.frames = []

        try:
            for i in count(1):
                self.frames.append(ImageTk.PhotoImage(gif.copy()))
                gif.seek(i)
        except EOFError:
            pass

        try:
            self.delay = gif.info['duration']
        except:
            self.delay = 100

        if len(self.frames) == 1:
            self.config(image=self.frames[0])
        else:
            self.next_frame()

    def unload(self):
        self.config(image=None)
        self.frames = None

    def next_frame(self):
        if self.frames:
            self.loc += 1
            self.loc %= len(self.frames)
            self.config(image=self.frames[self.loc])
            self.after(self.delay, self.next_frame)

#The frontal page of GUI with GIF, title, and buttons
class Home:
    def __new__(cls):
        destroy()
        Title()
        global filter_code

        filter_code = 0
        
        lbl = PlayGIF(root)
        lbl.pack()
        lbl.load('icon/gif.gif')
        lbl.config(bd = 0)
        
        cls.camera_icon = Icon( "icon/camera.png")
        camera_button = Button(root, image = cls.camera_icon,command = Camera,
                               width = 70, height = 50,bd = 0)
        camera_button.pack(side = BOTTOM)
        HoverText(camera_button, "Open Camera")

#When you hover over any buttons the text shown in screen the basic works of that thing goes here
class HoverText(Menu):
    def __init__(self, parent, text, command=None):
       Menu.__init__(self,parent, tearoff=0)

       text = re.split('\n', text)
       
       for t in text:
          self.add_command(label = t)
       
       self._displayed=False
       self.master.bind("<Enter>",self.display )
       self.master.bind("<Leave>",self.remove )

    def display(self,event):
       if not self._displayed:
          self._displayed=True
          self.post(event.x_root+1, event.y_root+1)

    def remove(self, event):
     if self._displayed:
       self._displayed=False
       self.unpost()

#It destroys if anything running on the parent window
def destroy():
    for window in root.winfo_children():
        window.destroy()

    try:
        cam.release()
    except:
        pass

#Selection of filter and applying them
class Filter:
    def __init__(self):
        self.button()

    def glass(self):
        global filter_code

        filter_code = 1
        
    def glass3(self):
        global filter_code
        
        filter_code = 2

    def hat(self):
        global filter_code
        
        filter_code = 3
    
    def beard2(self):
        global filter_code
        
        filter_code = 4
    
    def button(self):        
        frame = Frame(root,width = 400,height = 50, bd = 0)
        frame.pack(side = BOTTOM)
        
        width = 95
        height = 50
        
        self.glass_icon = Icon( "icon/glass.png")
        glass_button = Button(frame, image = self.glass_icon,command = self.glass,
               			width = width, height = height, bd = 0)
        glass_button.grid(row= 0, column = 0)
        
        self.glass3_icon = Icon( "icon/glass3.png")
        glass3_button = Button(frame, image = self.glass3_icon,command = self.glass3,
                               width = width, height = height, bd = 0)      
        glass3_button.grid(row = 0, column = 1)
        
        self.hat_icon = Icon( "icon/hat.png")
        hat_button = Button(frame, image = self.hat_icon,command = self.hat,
                               width = width, height = height, bd = 0)      
        hat_button.grid(row = 0, column = 2)
        
        self.beard2_icon = Icon("icon/beard2.png")
        beard2_button = Button(frame,image = self.beard2_icon, command = self.beard2,
                               width = width, height = height, bd = 0)
        beard2_button.grid(row = 0, column = 3)        
        

#Camera access displaying video with or without filter
class Camera:
    
    def __new__(cls):
        global cam,last_frame
        destroy()
        Title()
        
        last_frame = np.zeros((480, 480, 3), dtype=np.uint8)
        cam = cv2.VideoCapture(0)
        
        def video():     

            if not cam.isOpened():
                print("cant open the camera")
                
            flag, frame = cam.read()
            frame_flip = cv2.flip(frame, 1)
            frame = frame_flip[0:480,0:480]
            
            if flag is None:
                print ("Major error!")
            elif flag:
                last_frame = frame.copy()
            
            frame = image(last_frame)

            color_correction = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     
            
            array_image = Image.fromarray(color_correction)
            photo = ImageTk.PhotoImage(image=array_image)
            
            canvas.photo = photo
            canvas.create_image(0,0,image=photo,anchor = NW)
            
            canvas.after(10, video)
        
        canvas = Canvas(root, width = 480, height = 480)
        canvas.pack()
        
        Filter()
        video()
        
        root.mainloop()  
        cam.release()
   
if __name__ == '__main__':
    Home()

root.mainloop()
