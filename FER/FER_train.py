import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random

Datadirectory = "train/" #training dataset
Classes = ["0", "1", "2", "3", "4", "5", "6"] #List of folders
img_size = 224

# read all the images and convert them into array
training_Data=[] # data array
def create_training_Data():
    for category in Classes:
        path=os.path.join(Datadirectory,category)
        class_num=Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img))
                new_array=cv2.resize(img_array,(img_size,img_size))
                training_Data.append([new_array,class_num])
            except Exception as e:
                pass

create_training_Data()
print(len(training_Data))
random.shuffle(training_Data) #data must be shuffled otherwise machine will learn particular sequence of images

#temp=np.array(training_Data)

x=[]
y=[]
for features,label in training_Data:
    x.append(features)
    y.append(label)
    
x=np.array(x).reshape(-1,img_size,img_size,3) #converting it to 4 dimension, -1 means end, 3 means channels

#normalize the data
print("normalization started...")
x= x/255.0; #max black means 255
print("normalization done!")
y=np.array(y)

model=tf.keras.applications.MobileNetV2()

#Transfer learning - tuning, weights will start from last check point

base_input = model.layers[0].input #input

base_output = model.layers[-2].output

final_output = layers.Dense(128)(base_output) #adding new layer, after the output of global pooling layer 
final_output = layers.Activation('relu')(final_output) #activation function
final_output = layers.Dense(64)(final_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(7,activation='softmax')(final_output) #classes are 7, classification layer

new_model = keras.Model(inputs=base_input, outputs=final_output)

new_model.compile(loss="sparse_categorical_crossentropy", optimizer = "adam", metrics =["accuracy"])

new_model.fit(x,y,epochs=10)

new_model.save('FinalTrainedModel.h5')

#use trained model

