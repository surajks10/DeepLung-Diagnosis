import os
import numpy as np
import cv2              #Various libraries and modules are imported for handling images,
                        #creating and training the neural network, and managing warnings.
from PIL import Image
import matplotlib.pyplot as plt
import keras
import warnings
warnings.filterwarnings("ignore")
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input, Multiply
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


data = []
labels = []
data_1 = os.listdir("cell_images/cell_images/Parasitized/")

for i in data_1:
    try:
        image = cv2.imread("cell_images/cell_images/Parasitized/" + i)
        image_from_array = Image.fromarray(image, "RGB")
        size_image = image_from_array.resize((50, 50))
        data.append(np.array(size_image))
        labels.append(0)  #indicating parasitized)
    except AttributeError:
        print("")

# Load Uninfected images
Uninfected = os.listdir("cell_images/cell_images/Uninfected/")
for b in Uninfected:
    try:
        image = cv2.imread("cell_images/cell_images/Uninfected/" + b)
        array_image = Image.fromarray(image, "RGB")
        size_image = array_image.resize((50, 50))
        data.append(np.array(size_image))
        labels.append(1)  #indicating uninfected
    except AttributeError: 
        print("")

# Convert to numpy arrays
Cells = np.array(data)  #Converts the data and labels lists to numpy arrays for easier manipulation.
labels = np.array(labels)
s = np.arange(Cells.shape[0])
np.random.shuffle(s)
Cells = Cells[s]
labels = labels[s]
labels = keras.utils.to_categorical(labels)

# Splitting data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(Cells, labels, test_size=0.2, random_state=42) 

# Define Masked CNN model
input_shape = (50, 50, 3)
input_layer = Input(shape=input_shape)   #Defines two input layers: input_layer for the image and mask_input_layer for the mask.
mask_input_layer = Input(shape=input_shape)
masked_input = Multiply()([input_layer, mask_input_layer])

conv1 = Conv2D(filters=16, kernel_size=2, padding="same", activation="relu")(masked_input)
pool1 = MaxPooling2D(pool_size=2)(conv1)

conv2 = Conv2D(filters=32, kernel_size=2, padding="same", activation="relu")(pool1)   #Apply max pooling to reduce spatial dimensions.
pool2 = MaxPooling2D(pool_size=2)(conv2)

conv3 = Conv2D(filters=64, kernel_size=2, padding="same", activation="relu")(pool2)
pool3 = MaxPooling2D(pool_size=2)(conv3)

flatten_layer = Flatten()(pool3)

dense_layer1 = Dense(500, activation="relu")(flatten_layer)
output_layer = Dense(2, activation="sigmoid")(dense_layer1) 
 #The final layer has 2 units for binary classification (parasitized or uninfected).

masked_model = Model(inputs=[input_layer, mask_input_layer], outputs=output_layer)
masked_model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])  
#RMSprop (Root Mean Square Propagation) would adjusts the learning rate for each parameter individually

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define image and mask generators for training data
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    shear_range=0.1,
    fill_mode="nearest"
)

train_generator = train_datagen.flow(
    [X_train, X_train],
    Y_train,
    batch_size=32
)

# Define image and mask generators for validation data
val_datagen = ImageDataGenerator()

val_generator = val_datagen.flow(
    [X_test, X_test],
    Y_test,
    batch_size=32
)

# Train the model
history = masked_model.fit(train_generator, validation_data=val_generator, epochs=5, verbose=1)

# Save the model
masked_model.save("malaria_masked_cnn.h5")