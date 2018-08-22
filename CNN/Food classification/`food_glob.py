#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 13:17:30 2018

@author: abhiyush
"""

import glob
import shutil
import os



food101 = os.listdir("/home/abhiyush/mPercept/Computer vision /Covolution Neural Network (CNN)/Food classification/food41/images")
food_name = sorted(food101)

# To make an empty folder named as training_set, validation_set and test_set

new_directory_trai = '/home/abhiyush/mPercept/Computer vision /Covolution Neural Network (CNN)/Food classification/Untitled Folder/training_set'

if not os.path.exists(new_directory):
    os.makedirs(new_directory)

# To make an empty folder for each category of the food

for name in food_name:
    print(name)

    newpath_training = '/home/abhiyush/mPercept/Computer vision /Covolution Neural Network (CNN)/Food classification/food_glob/training_set/' + str(name)
    newpath_validation = '/home/abhiyush/mPercept/Computer vision /Covolution Neural Network (CNN)/Food classification/food_glob/validation_set/' + str(name)
    newpath_test = '/home/abhiyush/mPercept/Computer vision /Covolution Neural Network (CNN)/Food classification/food_glob/test_set/' + str(name)
    
    if not os.path.exists(newpath_training):
        os.makedirs(newpath_training)
    if not os.path.exists(newpath_validation):
        os.makedirs(newpath_validation)
    if not os.path.exists(newpath_test):
        os.makedirs(newpath_test)

# To make a dictionary to store the list of directories
food_name_dict = {}
for name in food_name:
    print(name)
    food_name_dict[name] = glob.glob('/home/abhiyush/mPercept/Computer vision /Covolution Neural Network (CNN)/Food classification/food41/images/' + str(name) + '/*.jpg')


# To store the data in the respective folders
for name in food_name:
    print(name)
    i = 0
    destination_training = '/home/abhiyush/mPercept/Computer vision /Covolution Neural Network (CNN)/Food classification/food_glob/training_set/' + str(name)
    destination_validation = '/home/abhiyush/mPercept/Computer vision /Covolution Neural Network (CNN)/Food classification/food_glob/validation_set/' + str(name)
    destination_test = '/home/abhiyush/mPercept/Computer vision /Covolution Neural Network (CNN)/Food classification/food_glob/test_set/' + str(name)

    for f in food_name_dict[name]:
        print(f)
        
        if i<800:
            shutil.copy(f, destination_training)
        if i>=800 and i < 900:
            shutil.copy(f, destination_validation)
        if i>=900 and i<1000:
            shutil.copy(f, destination_test)
        i = i + 1
        print (i)
        



