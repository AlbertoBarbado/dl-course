# -*- coding: utf-8 -*-
# @author: Alberto Barbado Gonzalez
# Máster Deep Learning Structuralia

# Importing the Keras libraries and packages
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras.models import Sequential # Para inicializar la NN (como es una Secuencia de layers, lo hago igual que con ANN; no uso la inici. de Graph)
from tensorflow.keras.layers import Convolution2D # Para hacer el paso de convolución, 1er step
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D # Para el Pooling step, paso 2
from tensorflow.keras.layers import Flatten # Para el flattening, step 3
from tensorflow.keras.layers import Dense # Para añadir los fully-connected layers hacia el layer de outputs
from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support

height, width = (28, 28) # 28x28 ya que es la dimensionalidad de los datos de keras
batch_size = 128
num_classes = 10
epochs = 12

# Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], height, width, 1)
x_test = x_test.reshape(x_test.shape[0], height, width, 1)
input_shape = (height, width, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# =============================================================================
# Extracción de variables
# =============================================================================
# Inicialización de la CNN
model = Sequential()

# Paso 1 - 1a Convolución
# En Convolution: nº filtros, filas, columnas. 
model.add(Convolution2D(filters=6,
                        kernel_size=(3, 3), 
                        activation='relu',
                        input_shape=input_shape))

# Paso 2 - 1er Avg. Pooling
model.add(AveragePooling2D(pool_size=(2, 2),
                               strides=2))

# Paso 3 - 2nda Convolución
model.add(Convolution2D(filters=16,
                        kernel_size=(3, 3),
                        activation='relu'))

# Paso 4 - 2ndo Avg. Pooling
model.add(AveragePooling2D(pool_size=(2, 2),
                               strides=2))

# Paso 5 - Flattening
model.add(Flatten())

# =============================================================================
# Red NN para classificación Fully Connected
# =============================================================================

# Entrada: n_batch x 120
# HL: 120 x 120
# Salida: n_batch x 120
model.add(Dense(units=120, activation='relu'))

# Entrada: n_batch x 120
# HL: 120 x 84
# Salida: n_batch x 84
model.add(Dense(units=84, activation='relu'))

# Entrada: n_batch x 84
# Salida: Clasificación multiclase para 10 categorías

model.add(Dense(units=10, activation = 'softmax')) 

model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Obtain predictions for all test set
y_pred = model.predict(x_test).round()

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit([1,2,3,4,5,6,7,8,9,10])
y_pred = lb.inverse_transform(y_pred)
y_test = lb.inverse_transform(y_test)

# Evaluate results
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ", cm)
precision, recall, fbeta, support = precision_recall_fscore_support(y_test, y_pred)
print("Precision: ")
print(precision)
print("Recall: ")
print(recall)