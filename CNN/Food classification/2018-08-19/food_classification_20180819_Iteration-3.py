#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 10:19:33 2018

@author: abhiyush
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

from keras.layers import Dropout
from keras import regularizers

import matplotlib.pyplot as plt
import numpy as np



def LeNet(width, height, channel):
    
    classifier = Sequential()
    
    # Step-1 Convolutional layer
    classifier.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape= (width, height, channel), activation = 'relu'))
    classifier.add(Dropout = 0.02)
    # Step-2 Maxpooling 
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    
    # Adding another convolutional layer
    classifier.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu')) 
    classifier.add(Dropout = 0.02)
    # Adding another maxpooling layer
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    
    # Step-3 Flatten
    classifier.add(Flatten())
    
    #Step-4 Full connection
    classifier.add(Dense(units = 128, activation = 'relu', kernel_regularizer = regularizers.l2(0.001)))
    classifier.add(Dropout(0.25))
    classifier.add(Dense(units = 128, activation = 'relu', kernel_regularizer = regularizers.l2(0.002)))
    classifier.add(Dropout(0.25))
    classifier.add(Dense(units = 10, activation = 'softmax'))
    
    return classifier

classifier = LeNet(128,128,3)
classifier.summary()

# Compiling CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting CNN to images 
# Image augmentation 

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   rotation_range = 15,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory("/home/abhiyush/mPercept/Computer vision /Covolution Neural Network (CNN)/Food classification/food_sample_10/training_set",
                                                 target_size = (128,128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

validation_set = validation_datagen.flow_from_directory("/home/abhiyush/mPercept/Computer vision /Covolution Neural Network (CNN)/Food classification/food_sample_10/validation_set",
                                                        target_size = (128,128),
                                                        batch_size = 32,
                                                        class_mode = 'categorical')
    
classifier.fit_generator(training_set,
                         steps_per_epoch = training_set.samples,
                         epochs = 2,
                         validation_data = validation_set,
                         validation_steps = validation_set.samples)

classifier.load_weights("/home/abhiyush/mPercept/Computer vision /Covolution Neural Network (CNN)/Food classification/2018-08-20/food_classification_20180820_Iteration-1.h5")    
    
test_image = image.load_img("/home/abhiyush/mPercept/Computer vision /Covolution Neural Network (CNN)/Food classification/food_sample/test_set/pizza/326809.jpg",
                            target_size = (64,64))   
plt.imshow(test_image) 
test_image_array = np.array(test_image) 
test_image_array.shape
test_image_array_expand_dims = np.expand_dims(test_image_array, axis = 0)
test_image_array_expand_dims.shape

test_image = image.load_img("/home/abhiyush/mPercept/Computer vision /Covolution Neural Network (CNN)/Food classification/food_sample/test_set/donuts/58396.jpg",
                            target_size = (64,64))   
plt.imshow(test_image) 
test_image_array = np.array(test_image) 
test_image_array.shape
test_image_array_expand_dims = np.expand_dims(test_image_array, axis = 0)
test_image_array_expand_dims.shape
 
    
result = classifier.predict(test_image_array_expand_dims)    

list(training_set)


    
    
    