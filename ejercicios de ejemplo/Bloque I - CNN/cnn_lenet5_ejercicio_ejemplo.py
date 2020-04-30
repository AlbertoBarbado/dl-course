# -*- coding: utf-8 -*-
# @author: Alberto Barbado Gonzalez
# Máster Deep Learning Structuralia

# Importing the Keras libraries and packages
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from shutil import copyfile, rmtree

from tensorflow.keras.models import Sequential # Para inicializar la NN (como es una Secuencia de layers, lo hago igual que con ANN; no uso la inici. de Graph)
from tensorflow.keras.layers import Convolution2D # Para hacer el paso de convolución, 1er step
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D # Para el Pooling step, paso 2
from tensorflow.keras.layers import Flatten # Para el flattening, step 3
from tensorflow.keras.layers import Dense # Para añadir los fully-connected layers hacia el layer de outputs
from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Dense, Dropout, Flatten
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, auc, log_loss


def prepare_train_test_set(path_ref="",train_per=0.8):
    """
    Function that loads the original images and splits them into a train and test
    folder according to the split percentage defined in the arguments.

    Parameters
    ----------
    path_ref : String, optional
        Relative path for the folder where the images are located and
        where the new fodlers are gonnna be created. The default is "".
    train_per : Float, optional
        Train set percentage. The default is 0.8.

    Returns
    -------
    None.

    """
    path_full = path_ref + '/flowers'
    entries = os.listdir(path_full + '/') 
    os.makedirs(path_ref + '/train') # Crear carpeta de train
    os.makedirs(path_ref + '/test') # Crear carpeta de test
    rmtree(path_full + '/flowers') # Eliminar esta carpeta innecesaria
    
    for entry in entries:
        print("Preparing datasets...")
        print("Entry: ", entry)
        files_inside = os.listdir(path_full + '/' + entry + '/')
        np.random.shuffle(files_inside)
        len_train = np.int(np.round(train_per*len(files_inside)))
        len_test = len(files_inside) - len_train
        
        os.makedirs(path_ref + '/train' + '/' + entry)
        os.makedirs(path_ref + '/test' + '/' + entry)
    
        [copyfile(path_full + '/' + entry + '/' + file, path_ref + '/train/' + entry + '/' + file)  for file in files_inside[:len_train]]
        [copyfile(path_full + '/' + entry + '/' + file, path_ref + '/test/' + entry + '/' + file)  for file in files_inside[len_train:]]
        

def image_data_generator(data_dir="",
                         train_data=False,
                         batch_size=10,
                         target_size=(100, 100),
                         color_mode='rgb',
                         class_mode='binary',
                         shuffle=True):
    """
    Function to load the images and use them in the NN.
    
    Parameters
    ----------
    data_dir : String, optional
        Path where the images are located. The default is "".
    train_data : TYPE, optional
        Whether to load datata with a train preprocessing or load it raw. 
        The default is False.
    batch_size : Integer, optional
        Batch size. The default is 10.
    target_size : Tuple, optional
        Dimensionality of the images (height, width). The default is (100, 100).
    color_mode : String, optional
        Color model used. The default is 'rgb'.
    class_mode : String, optional
        Class mode. 'categorical' for multiclass and 'binary' for binary.
        The default is 'binary'.
    shuffle : Boolean, optional
        Specifies whether to shuffle or not that input data. The default is True.
        
    Returns
    -------
    generator : generator
        Generator with the images to use later on.

    """
    
    if train_data:
        datagen = ImageDataGenerator(rescale=1./255,
                                     # rotation_range=20,
                                     # width_shift_range=0.2,
                                     # height_shift_range=0.2,
                                     # shear_range=0.2,
                                     # zoom_range=0.2,
                                     # horizontal_flip=True,
                                     validation_split=0.2
                                     )
    else:
        datagen = ImageDataGenerator(rescale=1./255)
        
    generator = datagen.flow_from_directory(data_dir,
                                            target_size=target_size,
                                            color_mode=color_mode,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            class_mode=class_mode)
    return generator


def build_model(optimizer="adam",
                loss='binary_crossentropy',
                height=32,
                width=32,
                channels=1,
                output_classes=1,
                final_activation="sigmoid"):
    """
    Architecture for the CNN
    Convolutional #1 

    Activation any activation function, we will relu
    Average Pooling #1 
    
    Convolutional #2 
    
    Activation any activation function, we will relu
    Average Pooling #2 
    
    Flatten Flatten the output shape of the final pooling layer
    Fully Connected #1 outputs 120
    
    Activation any activation function, we will relu
    Fully Connected #2 outputs 84
    
    Activation any activation function, we will relu
    Fully Connected (Logits) #3 output output_classes

    Parameters
    ----------
    optimizer : String, optional
        Algorithm for optimization. The default is "adam".
    loss : String, optional
        Loss function used. The default is 'binary_crossentropy'.
    height : String, optional
         Image height. The default is 32.
    width : String, optional
        Image width. The default is 32.
    channels : Integer, optional
        Number of channels used. The default is 1.
    output_classes : Integer, optional
        Number of output classes. The default is 1.
    final_activation : String, optional
        Final activation function. 'sigmoid' for binary, 'softmax' for multiclass.
        The default is "sigmoid".

    Returns
    -------
    model : Object
        Trained model.
    """

    # Inicialización de la CNN
    model = Sequential()
    
    # Paso 1 - 1a Convolución
    # En Convolution: nº filtros, filas, columnas. 
    # Se define también la dimensión del kernel (mismo para todos los canales)
    model.add(Convolution2D(filters=6,
                            kernel_size=(3, 3), 
                            padding='same',
                            activation='relu',
                            input_shape=(height,width,channels)))
    
    
    # Paso 2 - 1er Avg. Pooling
    # El tamaño del kernel del avg. pooling es 2x2
    model.add(AveragePooling2D(pool_size=(2, 2),
                               strides=2))
    
    # Paso 3 - 2nda Convolución
    model.add(Convolution2D(filters=16,
                            kernel_size=(3, 3),
                            padding='valid',
                            activation='relu'))
    
    # Paso 4 - 2ndo Avg. Pooling
    model.add(AveragePooling2D(pool_size=(2, 2),
                               strides=2))
    
    # Paso 5 - Flattening
    model.add(Flatten())
    
    # Paso 6 - Fully connected layers
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    
    # Output
    model.add(Dense(units=output_classes,
                    activation = final_activation)) 
    
    # Model compile
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model


if __name__ == "__main__":
    
    # ========================================================================
    # 1. Setup
    # ========================================================================
    # Parametros (I)
    tf.random.set_seed(42)
    path_ref = 'datasets/flowers-recognition'
    prepare_train_test_set(path_ref=path_ref ,train_per=0.8)
    
    # Parametros (II)
    batch_size = 400
    height, width = (240, 320)
    epochs = 10
    color_mode = "rgb" # 'rgb' or 'grayscale'
    optimizer = "adam"
    loss = "categorical_crossentropy"
    class_mode="categorical"
    path_train = path_ref + '/train'
    path_test = path_ref + '/test'
    output_classes=5 # Number of output classes
    final_activation="softmax"
    
    # Canales según el tipo de color_mode
    if color_mode == "grayscale":
        channels = 1
        grayscale = True
    else:
        channels = 3
        grayscale = False
    
    # Visualizar imagen de ejemplo
    img = load_img(path_train + '/daisy/5673551_01d1ea993e_n.jpg',
                    target_size = (height, width),
                    grayscale=grayscale)
    imgplot = plt.imshow(img)
    
    # Cargar las imágenes de train/test
    train_generator = image_data_generator(path_train,
                                           train_data=True,
                                           batch_size=batch_size,
                                           target_size=(height, width),
                                           color_mode=color_mode,
                                           class_mode=class_mode,
                                           shuffle=True)
    
    test_generator = image_data_generator(path_test,
                                          train_data=False,
                                          batch_size=batch_size,
                                          target_size=(height, width),
                                          color_mode=color_mode,
                                          class_mode=class_mode,
                                          shuffle=True)

    # ========================================================================
    # 2. Entrenamiento
    # ========================================================================
    # Definición del modelo y visualización de la arquitectura definida.
    model = build_model(optimizer=optimizer,
                        loss=loss,
                        height=height,
                        width=width,
                        channels=channels,
                        output_classes=output_classes,
                        final_activation=final_activation)
    
    print(model.summary())
    
    # Hago el fit de los sets de datos al modelo y entrenamiento del mismo
    model.fit_generator(train_generator, 
                        steps_per_epoch=batch_size,
                        epochs=epochs)
    
    
    # Guardamos el modelo en un archivo binario
    model.save('model_flowers.h5')
    
    # ========================================================================
    # 3. Evaluation
    # ========================================================================
    # Analizar las predicciones para imágenes independientes.
    test_image = load_img(path_test + '/rose/110472418_87b6a3aa98_m.jpg',
                          target_size = (height, width),
                          grayscale=grayscale)
    imgplot = plt.imshow(test_image)
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    print("Prediction: ", result)
    
    test_image = load_img(path_test + '/sunflower/35477171_13cb52115c_n.jpg',
                          target_size = (height, width),
                          grayscale=grayscale)
    imgplot = plt.imshow(test_image)
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    print("Prediction: ", result)
    
    # Loss/Accuracy
    score = model.evaluate(test_generator, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    # Transformar los datos de test en matrices numéricas.
    entries = os.listdir(path_test)
    X_test = []
    y_test = []
    for entry in entries:
        subpath = path_test + '/' + entry
        files = []
        for _, _, f in os.walk(subpath):
            files += f
            
        files = [x for x in files if 'jpg' in x]
        X_test += [np.expand_dims(img_to_array(load_img(subpath + '/' + f, 
                                                        target_size = (height, width),
                                                        grayscale=grayscale)), axis = 0) for f in files]
        if entry == "daisy":
            y_test += [0]*len(files)
        elif entry == "dandelion":
            y_test += [1]*len(files)
        elif entry == "rose":
            y_test += [2]*len(files)
        elif entry == "sunflower":
            y_test += [3]*len(files)
        elif entry == "tulip":
            y_test += [4]*len(files)
    
    # Obtener las predicciones para todos los datos de test
    y_pred = [model.predict_classes(x)[0] for x in X_test]
    y_pred = []
    for x in X_test:
        y_pred.append(model.predict_classes(x)[0])
    
    # Evaluar resultados
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix: ", cm)
    print("Precision: ", precision_score(y_test, y_pred, average='macro'))
    print("Recall: ", recall_score(y_test, y_pred, average='macro'))
    print("f1_score: ", f1_score(y_test, y_pred, average='macro'))



