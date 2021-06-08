# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:44:34 2020

@author: Saeid
"""

import glob
import os
import itertools
import numpy as np
import random
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.losses import categorical_crossentropy
from keras.models import save_model, Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from keras.applications.vgg16 import preprocess_input
from keras.activations import relu, sigmoid, softmax
from keras.utils import plot_model


dir_main_dataset = "C:/Users/Saeid/Desktop/Bubble_images/Main_Data/"
dir_analyzed_dataset = "C:/Users/Saeid/Desktop/Bubble_images/Analyzing_Data/"

num_train_samples = 500
num_test_samples = 100
num_validation_samples = 100

# Organize the dataset into train, test and validation sets
os.chdir('C:/Users/Saeid/Desktop/Bubble_images/Analyzing_Data/')

if os.path.isdir('train/1') == False:
    
    os.makedirs('train/1')
    os.makedirs('train/0')
    os.makedirs('test/1')
    os.makedirs('test/0')
    os.makedirs('validation/1')
    os.makedirs('validation/0')
    
    # Creating Train Data for folders containing bubbles
    for c in random.sample(glob.glob(dir_main_dataset + "1/*"),num_train_samples):
        shutil.move(c, dir_analyzed_dataset + 'train/1')
    
    # Creating Train Data for folders without bubbles
    for c in random.sample(glob.glob(dir_main_dataset + "0/*"),num_train_samples):
        shutil.move(c, dir_analyzed_dataset + 'train/0')
    
    # Creating Test Data for folders containing bubbles
    for c in random.sample(glob.glob(dir_main_dataset + "1/*"),num_test_samples):
        shutil.move(c, dir_analyzed_dataset + 'test/1')
    
    # Creating Test Data for folders without bubbles
    for c in random.sample(glob.glob(dir_main_dataset + "0/*"),num_test_samples):
        shutil.move(c, dir_analyzed_dataset + 'test/0')

    # Creating Validation Data for folders containing bubbles
    for c in random.sample(glob.glob(dir_main_dataset + "1/*"),num_validation_samples):
        shutil.move(c, dir_analyzed_dataset + 'validation/1')
    
    # Creating Validation Data for folders without bubbles
    for c in random.sample(glob.glob(dir_main_dataset + "0/*"),num_validation_samples):
        shutil.move(c, dir_analyzed_dataset + 'validation/0')

# Defining the directories relate to train, test and validation
train_dir = dir_analyzed_dataset + 'train/'
test_dir = dir_analyzed_dataset + 'test/'
validation_dir = dir_analyzed_dataset + 'validation/'

# Defining train, test and validation batches as generators
train_batches = ImageDataGenerator(preprocessing_function = preprocess_input)\
                .flow_from_directory(directory = train_dir,
                                     target_size = (224,224),
                                     classes = ['0','1'],
                                     batch_size = 10)
                
test_batches = ImageDataGenerator(preprocessing_function = preprocess_input)\
                .flow_from_directory(directory = test_dir,
                                     target_size = (224,224),
                                     classes = ['0','1'],
                                     batch_size = 10,
                                     shuffle = False)
                
validation_batches = ImageDataGenerator(preprocessing_function = preprocess_input)\
                    .flow_from_directory(directory = validation_dir,
                                         target_size = (224,224),
                                         classes = ['0','1'],
                                         batch_size = 10)

imgs, labels = next(train_batches)

# A function for plotting the images

def plot_image(images_arr, labels):
    
    fig, axes = plt.subplots(nrows = 1, ncols = 10, figsize = [20,3])
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(images_arr[i])
        
        if labels[i][0] == 1:
            xlabel = "Cat"    
        else:
            xlabel = "Dog"
        
        ax.set_xlabel(xlabel)
    
    plt.tight_layout()
    plt.show()

#plot_image(imgs,labels)

##################################
# Building a Sequential Model_ CNN
##################################
myModel = Sequential()

# first Convolutional layer
filters1 = 32
kernel_size1 = 3
myModel.add(Conv2D(filters = filters1,
                   kernel_size = kernel_size1,
                   strides=(1, 1),
                   padding='SAME',
                   activation=relu,
                   use_bias=True,
                   input_shape = (224,224,3)))

# first Max Pooling layer
pool_size1 = (2,2)
strides1 = 2
myModel.add(MaxPool2D(pool_size = pool_size1,
                      strides = strides1,
                      padding = 'valid'))

# second Convolutional layer
filters2 = 64
kernel_size2 = 3
myModel.add(Conv2D(filters = filters2,
                   kernel_size = kernel_size2,
                   strides=(1, 1),
                   padding='SAME',
                   activation=relu,
                   use_bias=True))

# second Max Pooling layer
pool_size2 = (2,2)
strides2 = 2
myModel.add(MaxPool2D(pool_size = pool_size2,
                      strides = strides2,
                      padding = 'valid'))

# Flatten layer
myModel.add(Flatten())

# Output_layer
myModel.add(Dense(units = 2, activation = softmax))

# summary
myModel.summary()

# plotting the model
plot_model(model = myModel,
           to_file = "Saved_Parameters/Cats_Vs_Dogs.jpg",
           show_shapes = True,
           show_layer_names = True,
           dpi = 300)


#####################
# Compiling our Model
#####################
myModel.compile(optimizer = Adam(lr = 0.0001),
                loss = categorical_crossentropy,
                metrics = ['accuracy'])

####################
# Training our Model
####################
# 1) There is no need to have y because we have already added the labels into \
# the train, test and validation batches!
# 2) Instead of validation_split,we use validation_data, because we have \
# already split it in val_batches!
# 3) There is also no need for batch_size, because we have already made batches!
train_epochs = 3
network_history = myModel.fit(x = train_batches,
                              validation_data = validation_batches,
                              epochs = train_epochs,
                              verbose = 2)

history = network_history.history

myModel_loss = history['loss']
myModel_acc = history['accuracy']
myModel_val_loss = history['val_loss']
myModel_val_acc = history['val_accuracy']

# a function for plotting the accuracy and loss of val and train datasets
def plot_acc(Model_loss,Model_acc,Model_val_loss,Model_val_acc):
    
    fig, [ax1,ax2] = plt.subplots(nrows = 1, ncols = 2, figsize = [15,7])
    plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
    num_epochs = np.arange(1,train_epochs+1)
    
    ax1.plot(num_epochs, Model_loss,label = "loss")
    ax1.scatter(num_epochs, Model_loss)
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("loss")
    ax1.set_xticks(num_epochs)
    ax1.set_yticks(Model_loss)
    
    ax2.plot(num_epochs, Model_acc,label = "accuracy")
    ax2.scatter(num_epochs, Model_acc)
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("accuracy")
    ax2.set_xticks(num_epochs)
    ax2.set_yticks(Model_acc)
    
    ax1.plot(num_epochs, Model_val_loss,label = "validation loss")
    ax1.scatter(num_epochs, Model_val_loss)
    ax1.legend()
    
    ax2.plot(num_epochs, Model_val_acc,label = "validation accuracy")
    ax2.scatter(num_epochs, Model_val_acc)     
    ax2.legend()
        
plot_acc(Model_loss = myModel_loss,
         Model_acc = myModel_acc,
         Model_val_loss = myModel_val_loss,
         Model_val_acc = myModel_val_acc)

######################
# Evaluating the Model
######################
loss, acc = myModel.evaluate(x = test_batches,
                             verbose = 2)

print("Test dataset: \nLoss = {}\nAccuracy = {}".format(loss,acc))

######################
# Predicting the Model
######################
Y_test_pred = myModel.predict(x = test_batches,
                              verbose = 2)

Cls_test_pred = np.argmax(Y_test_pred,axis = 1)

# Confusion Matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm = confusion_matrix(y_true = test_batches.classes,
                      y_pred = Cls_test_pred)

classes = ['Cat','Dog']

plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues)

##################
# Saving the Model
##################
myModel.save(filepath = "Saved_Parameters/Cats_Vs_Dogs_model.h5",
             overwrite = True,
             include_optimizer = True)
myModel.save_weights(filepath = "Saved_Parameters/Cats_Vs_Dogs_model_W.h5",
             overwrite = True)


