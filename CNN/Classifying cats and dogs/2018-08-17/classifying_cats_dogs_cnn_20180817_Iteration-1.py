# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#!/usr/bin/env python

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing import image
from keras import regularizers

import numpy as np
import matplotlib.pyplot as plt


def LeNet(width, height, channel):
    # Initialising the CNN
    classifier = Sequential()
    
    # Step 1 - Convolution
    classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (width, height, channel), activation = 'relu'))
    #Adding drop out which is used to remove overfitting
    classifier.add(Dropout(0.4))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    #Adding drop out
    classifier.add(Dropout(0.4))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
     # Adding a third convolutional layer
    classifier.add(Conv2D(32, (3,3), activation = 'relu'))
    #Adding drop out
    classifier.add(Dropout(0.4))
    classifier.add(MaxPooling2D(pool_size = (2,2)))
     
    #Step 3 - Flattening
    classifier.add(Flatten())
    
    # Step 4 - Full connection
    classifier.add(Dense(units = 128, activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))
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
                                   rotation_range = 30,
                                   horizontal_flip = True)


validation_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/kaggle/working/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


validation_set = validation_datagen.flow_from_directory('/kaggle/working/validation_set',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = training_set.samples,
                         epochs = 2,
                         validation_data = validation_set,
                         validation_steps = validation_set.samples)

classifier.save_weights("/kaggle/working/cats_dogs_cnn_20180817_img_size_128_iteration-1.h5")


classifier.load_weights("/kaggle/working/cats_dogs_cnn_20180817_img_size_128_iteration-1.h5")


# Prediction Cats = 0 and dogs = 1
# classifying all the image at once
result_dogs = []
for i in range(4001, 5001):
    test_image = image.load_img('/kaggle/working/test_set/dogs/dog.'+str(i)+'.jpg', target_size = (128,128))
    test_image_array = image.img_to_array(test_image)/255
    test_image_array_expand_dims = np.expand_dims(test_image_array, axis = 0)
    predict = classifier.predict(test_image_array_expand_dims)

    if float(predict) >= 0.5:
        prediction = "Dog"
    else:
        prediction = "Cat"
        
    result_dogs.append("dog-" + str(i) + "-->" + prediction + ", probability : " + str(float(predict)))
    

result_cats = []
for i in range(4001, 5000):
    test_image = image.load_img('/kaggle/working/test_set/cats/cat.'+str(i)+'.jpg', target_size = (128,128))
    test_image_array = np.array(test_image)/255
    test_image_array_expand_dims = np.expand_dims(test_image_array, axis = 0)
    predict = classifier.predict(test_image_array_expand_dims)

    if float(predict) >= 0.5:
        prediction = "Dog"
    else:
        prediction = "Cat"
        
    result_cats.append("cat-" + str(i) + "-->" + prediction + ", probability : " + str(float(predict)))
 
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

fault = (dog+cat)/2000
print (fault)

accuracy = 1 - fault

print("The accuracy of the model is :", accuracy)


