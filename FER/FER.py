import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

new_model=tf.keras.models.load_model('Final_model_95p07.h5')

import cv2
path="haarcascade_frontalface_default.xml"
font_scale=1.5
font=cv2.FONT_HERSHEY_PLAIN
rectangle_bgr=(255,255,255) #rectangle background white
img=np.zeros((500,500)) #make a black image
text="face emotions" #set some text
(text_width,text_height)=cv2.getTextSize(text,font,fontScale=font_scale,thickness=1)[0] #get width and height of textbox
text_offset_x=10
text_offset_y=img.shape[0]-25 #set the text start position
#make the coordinates of the box with a small padding of two pixels
box_coords=((text_offset_x,text_offset_y),(text_offset_x+text_width+2,text_offset_y-text_height-2))
cv2.rectangle(img,box_coords[0],box_coords[1],rectangle_bgr,cv2.FILLED)
cv2.putText(img,text,(text_offset_x,text_offset_y),font,fontScale=font_scale,color=(0,0,0),thickness=1)

cap=cv2.VideoCapture(1)
if not cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam!")
    
while True:
    ret,frame=cap.read()
    faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,1.1,4)
    for x,y,w,h in faces:
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        facess=faceCascade.detectMultiScale(roi_gray)
        if len(facess)==0:
            print("Face not detected!")
        else:
            for(ex,ey,ew,eh) in facess:
                face_roi=roi_color[ey:ey+eh,ex:ex+ew] #cropping the face
                
        final_image=cv2.resize(face_roi,(224,224))
        final_image=np.expand_dims(final_image,axis=0) #four dimensions
        final_image=final_image/255.0
        
        font=cv2.FONT_HERSHEY_SIMPLEX
        
        Predictions=new_model.predict(final_image)
        
        font_scale=1.5
        font=cv2.FONT_HERSHEY_PLAIN
        
        if (np.argmin(Predictions)==0):
            status="Angry"
        elif (np.argmin(Predictions)==1):
            status='Disgust'
        elif (np.argmin(Predictions)==2):
            status='Fear'
        elif (np.argmin(Predictions)==3):
            status='Happy'
        elif (np.argmin(Predictions)==4):
            status='Sad'
        elif (np.argmin(Predictions)==5):
            status='Surprise'
        else:
            status="Neutral"
            
        x1,y1,w1,h1=0,0,175,75
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        
        cv2.imshow('Face Emotion Recognition',frame)
        
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
            
cap.release()
cv2.destroyAllWindows()
            