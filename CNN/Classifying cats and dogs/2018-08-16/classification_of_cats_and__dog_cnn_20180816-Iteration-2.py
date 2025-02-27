# -*- coding: utf-8 -*-
"""classification_of_cats_and _dog_cnn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dvZX5QFrjAT532LxUZhLZ-OvzgBTGQGr
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 11:33:42 2018

@author: abhiyush
"""

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing import image

import numpy as np
import matplotlib.pyplot as plt


def LeNet(width, height, channel):
    # Initialising the CNN
    classifier = Sequential()
    
    # Step 1 - Convolution
    classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (width, height, channel), activation = 'relu'))
    # Adding the dropout
    classifier.add(Dropout(0.2))
    
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(Dropout(0.2))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Step 3 - Flattening
    classifier.add(Flatten())
    
    # Step 4 - Full connection
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    
    return classifier

classifier = LeNet(128, 128, 3)
classifier.summary()
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/content/Convolutional_Neural_Networks/dataset/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


test_set = test_datagen.flow_from_directory('/content/Convolutional_Neural_Networks/dataset/test_set',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'binary')

!wget http://www.superdatascience.com/wp-content/uploads/2017/03/Convolutional-Neural-Networks.zip

from google.colab import files

files.os.listdir()

!unzip Convolutional-Neural-Networks.zip

ls

test_image = image.load_img('/content/Convolutional_Neural_Networks/dataset/test_set/dogs/dog.4005.jpg', target_size = (128,128))
plt.imshow(test_image)
test_image_array = image.img_to_array(test_image) #test_image_array = np.array(test_image)/255.
test_image_array.shape
test_image_array_expand_dims = np.expand_dims(test_image_array, axis = 0)
test_image_array_expand_dims.shape

files.os.listdir()

cd Convolutional_Neural_Networks

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 2,
                         validation_data = test_set,
                         validation_steps = 2000)

ls

# Prediction Cats = 0 and dogs = 1
# classifying all the image at once
result_dogs = []
for i in range(4001, 5001):
    test_image = image.load_img('/content/Convolutional_Neural_Networks/dataset/test_set/dogs/dog.'+str(i)+'.jpg', target_size = (128,128))
    test_image_array = image.img_to_array(test_image)/255
    test_image_array_expand_dims = np.expand_dims(test_image_array, axis = 0)
    predict = classifier.predict(test_image_array_expand_dims)

    if float(predict) >= 0.5:
        prediction = "Dog"
    else:
        prediction = "Cat"
        
    result_dogs.append("dog-" + str(i) + "-->" + prediction + ", probability : " + str(float(predict)))
    

result_cats = []
for i in range(4001, 5001):
    test_image = image.load_img('/content/Convolutional_Neural_Networks/dataset/test_set/cats/cat.'+str(i)+'.jpg', target_size = (128,128))
    test_image_array = np.array(test_image)/255
    test_image_array_expand_dims = np.expand_dims(test_image_array, axis = 0)
    predict = classifier.predict(test_image_array_expand_dims)

    if float(predict) >= 0.5:
        prediction = "Dog"
    else:
        prediction = "Cat"
        
    result_cats.append("cat-" + str(i) + "-->" + prediction + ", probability : " + str(float(predict)))

result_cats

# Incorrect prediction for cats
dog = 0
for result in result_cats:
    if "Dog" in result:
        dog = dog + 1
print(dog)

# Incorrect prediction for dogs
cat = 0
for result in result_dogs:
    if "Cat" in result:
        cat = cat + 1
print(cat)

fault = (216+178)/2000
print (fault)

accuracy = 1 - fault

print("The accuracy of the model is :", accuracy)

import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

