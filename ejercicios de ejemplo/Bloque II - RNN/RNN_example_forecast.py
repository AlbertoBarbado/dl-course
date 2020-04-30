# -*- coding: utf-8 -*-
# @author: Alberto Barbado Gonzalez
# Máster Deep Learning Structuralia

import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Embedding, LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =============================================================================
# 0. General functions
# =============================================================================
def create_rnn_model(x_input, n_lstm=8, loss='mean_squared_error',
                     optimizer='adam'):
    """
    Función para crear la RNN. Como parámetro de entrada sólo necesita la matriz
    de features para especificar la dimensionalidad de entrada de la NN.

    Returns
    -------
    model : object
        Trained model.

    Parameters
    ----------
    x_input : array
        Matriz de features de entrada.
    n_lstm : int, optional
        Number of lstm used. The default is 8.
    loss : string, optional
        loss metric. The default is 'mean_squared_error'.
    optimizer : string, optional
        optimizer. The default is 'adam'.

    Returns
    -------
    model : object
        Trained model.

    """

    # Begin sequence
    model = tf.keras.Sequential()
    
    # Add a LSTM layer with 8 internal units.
    model.add(LSTM(n_lstm, input_shape=x_input.shape[-2:]))
    
    # Output
    model.add(Dense(1))
    
    # Compile model
    model.compile(loss=loss, optimizer=optimizer)
    
    return model

def univariate_data(dataset, start_index, end_index, history_size, target_size):
    """
    Función para obtener la matriz de features en función de la secuencia de entrada
    univariante. Partiendo de dicha secuencia, y para un intervalo de la misma
    en función de unos índices especificados, 
    Parameters
    ----------
    dataset : TYPE
        DESCRIPTION.
    start_index : TYPE
        DESCRIPTION.
    end_index : TYPE
        DESCRIPTION.
    history_size : TYPE
        DESCRIPTION.
    target_size : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    data = []
    labels = []
    
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    
    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[min(indices):max(indices)+1].values, (history_size, 1)))
        labels.append(dataset[i+target_size])
        
    return np.array(data), np.array(labels)

# =============================================================================
# 1. Load & prepare data
# =============================================================================
# Set seed
tf.random.set_seed(42)
  
# Load data
# path_folder = "datasets/air-passengers"
# df_raw = pd.read_csv("{0}/AirPassengers.csv".format(path_folder))

# Define params
path_folder = "datasets/jaipur-weather-forecasting"
col_feature = 'meantempm'
col_datetime = 'date'

# Load data
df_raw = pd.read_csv("{0}/JaipurFinalCleanData.csv".format(path_folder))
df_raw = df_raw.sort_values(by=[col_datetime], ascending=True)

# Plot data
df_raw.plot(x=col_datetime, y=col_feature, rot=45)

# Standarize the dataset
scaler = StandardScaler()
df_input = df_raw.copy()
df_input[col_feature] = scaler.fit_transform(df_input[col_feature].values.reshape(-1, 1))
df_input = pd.Series(data=df_input[col_feature].values)

# Train/Test split
train_size = int(len(df_input) * 0.8)
test_size = len(df_input) - train_size
df_train, df_test = df_input[0:train_size], df_input[train_size:len(df_input):]
print(len(df_train), len(df_test))

"""
df_ref = df_train[0:26]
df = pd.DataFrame()
df = df.append(pd.DataFrame(df_train[0:5].reset_index(drop=True)).T)
df = df.append(pd.DataFrame(df_train[1:6].reset_index(drop=True)).T)
df = df.append(pd.DataFrame(df_train[2:7].reset_index(drop=True)).T)
df = df.append(pd.DataFrame(df_train[3:8].reset_index(drop=True)).T)
df = df.append(pd.DataFrame(df_train[4:9].reset_index(drop=True)).T)
df = df.append(pd.DataFrame(df_train[5:10].reset_index(drop=True)).T)

df['pred'] = df_train[5:11].values

"""

# Generate features
n_past_history = 20
n_future_target = 0

x_train, y_train = univariate_data(dataset = df_input,
                                   start_index = 0, 
                                   end_index = train_size,
                                   history_size = n_past_history,
                                   target_size = n_future_target)

x_test, y_test = univariate_data(dataset = df_input,
                                 start_index = train_size, 
                                 end_index = None,
                                 history_size = n_past_history,
                                 target_size = n_future_target)

# Transform to tensor
batch_size = 25

train_univariate = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_univariate = train_univariate.batch(batch_size).repeat()

test_univariate = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_univariate = test_univariate.batch(batch_size).repeat()

# =============================================================================
# 2. Train RNN
# =============================================================================
# General params
evaluation_interval = 30
epochs = 100
steps_per_epoch =  10

# Create RNN
model = create_rnn_model(x_train)
model.summary()

# Fit RNN
model.fit(train_univariate, 
          epochs=epochs,
          steps_per_epoch=steps_per_epoch)

# Save file
model.save('model_forecats_weather.h5')

# =============================================================================
# Evaluate
# =============================================================================
# Obtain predictions
y_pred = model.predict(x_test)
y_pred = [x[0] for x in y_pred] # change output format
df_preds = pd.DataFrame({'y_pred':y_pred, 'y_test':y_test})
df_preds.index = df_raw[train_size+n_past_history:][col_datetime].values # Set indexes with the date

# Obtain metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE: ", mse)
print("MAE: ", mae)
print("R2: ", r2)

# Unscale data
df_preds['y_pred'] = scaler.inverse_transform(df_preds['y_pred'])
df_preds['y_test'] = scaler.inverse_transform(df_preds['y_test'])

# Plot results
df_preds.plot(rot=45)
