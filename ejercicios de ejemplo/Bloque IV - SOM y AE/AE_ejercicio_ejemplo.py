# -*- coding: utf-8 -*-
# @author: Alberto Barbado Gonzalez
# Máster Deep Learning Structuralia

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score
                            

def autoencoder_function(nv, nh, optimizer="adam",
                         loss='mean_squared_error',
                         metrics=['accuracy']):
    """
    Main function to define the AE Neural Network.

    Parameters
    ----------
    nv : TYPE
        DESCRIPTION.
    nh : TYPE
        DESCRIPTION.
    optimizer : TYPE, optional
        DESCRIPTION. The default is "adam".
    loss : TYPE, optional
        DESCRIPTION. The default is 'mean_squared_error'.
    metrics : TYPE, optional
        DESCRIPTION. The default is ['accuracy'].

    Returns
    -------
    autoencoder : TYPE
        DESCRIPTION.

    """
    
    # Define input
    input_layer = Input(shape=(nv,))
    
    # Encoding
    encoder = Dense(nh, activation='relu',
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
    # Decoding
    decoder = Dense(nv, activation='sigmoid')(encoder)
    
    # Model
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return autoencoder


# =============================================================================
# 1. Prepare data
# =============================================================================
# Disable eager_exec
tf.compat.v1.disable_eager_execution()

# Load & prepare data
df_raw = pd.read_csv("creditcard.csv").drop(columns=['Time'])

# Plot target variable
labels = ['Normal', 'Fraud']
sizes = [len(df_raw[df_raw['Class']==0]), len(df_raw[df_raw['Class']==1])]
colors = ['lightskyblue','red']
explode = (0, 0.1)
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title("Normal vs Fraud Clients")

# Normalize
sc = MinMaxScaler(feature_range = (0, 1)) 
df_raw['Amount'] = sc.fit_transform(df_raw['Amount'].values.reshape(-1, 1))

# =============================================================================
# 2. Supervised
# =============================================================================
# Train/Eval split
X_train, X_test = train_test_split(df_raw, test_size=0.2, random_state=42)

# Train only in non outliers
X_train = X_train[X_train.Class == 0]
X_train = X_train.drop(['Class'], axis=1)
y_test = X_test['Class']
X_test = X_test.drop(['Class'], axis=1)
X_train = X_train.values
X_test = X_test.values

# Parameters
nv = X_train.shape[1] # visible unists
nh = 32 # hidden units
epochs=50
batch_size=256

# Train AE
model = autoencoder_function(nv, nh,optimizer="adam", loss='mean_squared_error',
                             metrics=['accuracy'])
history = model.fit(X_train, X_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True)

# Plot metrics
plt_history = history.history
plt.plot(plt_history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['validation'], loc='upper right')

# Obtain predictions
threshold = 2.9 # Set threshold for reconstruction error

predictions = model.predict(X_test)
se = np.mean(np.power(X_test - predictions, 2), axis=1)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': y_test})

# Visualize [Chart]
groups = error_df.groupby('true_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Fraud" if name == 1 else "Normal")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show()

# Visualize [CM]
y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, y_pred)
sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, 
            annot=True, fmt="d", annot_kws={"fontsize":12})
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# Other metrics
print("F1 score: ", f1_score(error_df.true_class, y_pred))

# =============================================================================
# 3. Unsupervised
# =============================================================================
# Train/Eval split
df_input = df_raw.copy().drop(columns=['Class']) # No class column
X_train, X_test = train_test_split(df_input, test_size=0.2, random_state=42)

# Parameters
nv = X_train.shape[1] # visible unists
nh = 32 # hidden units
epochs=50
batch_size=256

# Train AE
model = autoencoder_function(nv, nh,optimizer="adam", loss='mean_squared_error',
                             metrics=['accuracy'])
history = model.fit(X_train, X_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True)

# Obtain predictions
threshold = 2.9 # Set threshold for reconstruction error
threshold = 25 # Set threshold for reconstruction error

predictions = model.predict(X_test)
se = np.mean(np.power(X_test - predictions, 2), axis=1)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse})

# Visualize [Chart]
fig, ax = plt.subplots()

ax.plot(error_df.index, error_df.reconstruction_error, marker='o', ms=3.5, linestyle='')
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for unlabelled data")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show()

