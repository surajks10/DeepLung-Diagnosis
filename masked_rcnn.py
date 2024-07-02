import os
from PIL import Image
import itertools
from tqdm import tqdm
from sklearn.utils import shuffle

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Activation, BatchNormalization, Masking
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator    

train = list(os.walk('train'))
label_names = train[0][1]     #Assigns the list of label names (subdirectory names) to the variable label_names.
dict_labels = dict(zip(label_names, list(range(len(label_names)))))  #Creates a dictionary where each label name is mapped to a unique integer.
print(dict_labels) 


def dataset(path):  # put path till the highest directory level (Here flow from directory method of Imagedatagenerator is used)
    images = []
    labels = []
    for folder in tqdm(os.listdir(path)):

        # dict_labels is the dictionary whose key:value pairs are classes:numbers
        # representing them
        value_of_label = dict_labels[folder]

        for file in (os.listdir(os.path.join(path, folder))):
            path_of_file = os.path.join(os.path.join(path, folder), file)

            image = cv2.imread(path_of_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (228, 228))
            images.append(image)
            labels.append(value_of_label)

    images = np.array(images, dtype='float32') / 255.0
    labels = np.array(labels)

    return images, labels


images, labels = dataset('train')
images, labels = shuffle(images, labels)
batch_size = 12  # Number of samples that will be processed in each iteration during training.

train_datagen = ImageDataGenerator(  #Imagedatagenerator increase the diversity of the training set which helps 
                                        #in making the model more robust.

    rescale=1./255,  # Normalizes pixel values to the range of 0 to 1.
    zoom_range=0.2,  # Randomly zooms in or out of images.
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[1, 1.1],
    fill_mode='constant'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical'   #Indicates that the labels are one-hot encoded for a multi-class classification problem
)

test_generator = test_datagen.flow_from_directory(
    'test',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = test_datagen.flow_from_directory(
    'val',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical'
)

model = Sequential()    #model will be built layer-by-layer in a linear stack.
model.add(Masking(mask_value=0., input_shape=(128, 128, 3)))  # Masking layer to ignore padding values.
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))  #ReLU (Rectified Linear Unit) activation function to introduce non-linearity.
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  #2x2, which reduces the spatial dimensions by half.

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())  # Flattens the 3D output of the last convolutional block into a 1D vector
model.add(Dense(units=512, activation="relu"))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=64, activation="relu"))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(units=2, activation="softmax"))
model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, epochs=5, verbose=1, validation_data=val_generator, shuffle=False)
model.save('Masked_CNN_pneumonia.hdf5')
