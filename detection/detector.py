import numpy as np
from keras.models import load_model
import cv2

model = load_model('model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
dimensions = (96, 96)

def image(frame):
    
    try:   

        faces = face_cascade.detectMultiScale(frame, 1.15, 6) 
        x,y,w,h = faces[0]
        y += 18
        h += 5
        x += 8
        w -= 10
        gray = cv2.resize(frame[y:y+h, x:x+w],dimensions)
        imgd = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        imgd = imgd / 255
        imgd = imgd.reshape(-1,dimensions[0],dimensions[1],1)

        lands = model.predict(imgd)

        lands = np.round((lands.reshape(-1, 2) * 48) + 48)
                
        for land in lands:
            cv2.circle(gray, (int(land[0]), int(
                    land[1])), 1, (0, 255, 0), 1)
            
        gray =  cv2.resize(gray,(w,h))
        frame[y:y+h, x:x+w,] = gray

        return frame

    except:
        return frame
