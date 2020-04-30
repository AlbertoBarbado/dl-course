# -*- coding: utf-8 -*-
# @author: Alberto Barbado Gonzalez
# Máster Deep Learning Structuralia

import re
import numpy as np
import pandas as pd
import tensorflow as tf
import gensim.downloader as api

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, log_loss
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# 0. General functions
# =============================================================================
def word_vector(df_input, lemmatizer, word_vectors, vocabulary, col_sentences):
    """
    Función para preprocesar las palabras de entrada y obtener una lista con 
    los las matrices de embeddings de las palabras de cada registro.

    Parameters
    ----------
    df_input : dataframe
        dataframe de entrada con todos los textos.
    lemmatizer : object
        objeto del lematizador de NLTK.
    word_vectors : object
        objecto con los word2vec del vocabnbulario de Gensim.
    vocabulary : list
        lista con las palabras existentes en el vocabulario de Gensim.
    col_sentences : str
        columna del dataframe donde están las frases.

    Returns
    -------
    X : list
        Lista de listas en las que cada registro tiene la lista con los arrays
        de los embeddings de las palabras de esa frase. Es decir, X[0] tiene 
        una lista donde cada elemento corresponde a los embeddings de una palabra.
        Así, por ejemplo, X[0][2] será un vector de dimensión 100 donde aparece
        el vector de embeddings de la tercera palabra de la primera frase.
    """
    
    
    X = []
    
    for text in df_input[col_sentences]:
        
        # Tokenizo cada frase
        words = re.findall(r'\w+', text.lower(),flags = re.UNICODE) # Paso a minusculas todo
        # Eliminación de las stop_words 
        words = [word for word in words if word not in stopwords.words('english')]
        # Elimino guiones y otros simbolos raros
        words = [word for word in words if not word.isdigit()] # Elimino numeros    
        # Stemming 
        words = [lemmatizer.lemmatize(w) for w in words]
        # Eliminar palabras que no estén en el vocabulario
        words = [word for word in words if word in vocabulary]
        # Word2Vec
        words_embeddings = [word_vectors[x] for x in words] 
            
        # Guardo la frase final
        X.append(words_embeddings) # lo guardo como un numpy array
        
    return X


def create_RNN(x_train, K, n_lstm=8, loss='categorical_crossentropy', optimizer='adam'):
    """
    Función para crear la RNN. Como parámetro de entrada sólo necesita la matriz
    de features para especificar la dimensionalidad de entrada de la NN.

    Parameters
    ----------
    x_input : array
        Matriz de features de entrada.
    K: int
        Clases de salida
    n_lstm : int, optional
        Number of lstm used. The default is 8.
    loss : string, optional
        loss metric. The default is 'categorical_crossentropy'.
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
    model.add(LSTM(n_lstm, input_shape=x_train.shape[-2:]))
    
    # Add Dropout
    # model.add(Dropout(0.5))
    
    # # Another layer
    # model.add(Dense(100, activation='relu'))
    
    # # Output
    model.add(Dense(K, activation='softmax'))
    
    # Compile model
    model.compile(loss=loss, optimizer=optimizer)
    
    return model


# =============================================================================
# 1. Load Data
# =============================================================================
# Load files
tf.random.set_seed(42)
path_files = "datasets/bbc-fulltext-and-category"
df_raw = pd.read_csv(path_files+'/bbc-text.csv')

# path_files = "datasets/sentiment-review"
# df_raw = pd.read_csv(path_files+'/train/train.tsv', sep='\t')

# Shuffle input
df_raw = df_raw.sample(frac=1)

# Load word2vec
word_vectors = api.load("glove-wiki-gigaword-100")
vocabulary = [x for x in word_vectors.vocab]

# Set lemmatizer
lemmatizer = WordNetLemmatizer() 

# Check embeddings of one word
vector = word_vectors['computer'] 
print(vector)

# Label encoding
lb = LabelEncoder()
df_raw['category'] = lb.fit_transform(df_raw['category'])

X = pd.DataFrame(df_raw['text'])
y = df_raw['category']

# # Undersampling, same classes
# rus = RandomUnderSampler(random_state=42)
# X, y = rus.fit_resample(X, y)

# =============================================================================
# 2. Preprocess
# =============================================================================
# Obtain X variable and prepare y.
X = word_vector(X, 
                lemmatizer,
                word_vectors, 
                vocabulary,
                col_sentences="text")

# One-hot encode output
y = to_categorical(y)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Obtain tensor: [N_SENTENCES x SEQ_LENGTH x EMBEDDING_FEATURES]
SEQ_LENGTH = np.int(np.round(np.percentile([len(x) for x in X], 99, interpolation = 'midpoint')))
# SEQ_LENGTH = np.int(np.round(np.percentile([len(x) for x in X], 100, interpolation = 'midpoint')))
data_train = pad_sequences(X_train, 
                           maxlen=SEQ_LENGTH,
                           padding="post", 
                           truncating="post")

data_test = pad_sequences(X_test, 
                          maxlen=SEQ_LENGTH,
                          padding="post", 
                          truncating="post")

# =============================================================================
# 3. Train model
# =============================================================================
# Params
# M = 50 # hidden layer size
K = y_train.shape[1] # N classes
# V = data_train.shape[2] # EMBEDDING_FEATURES
batch_size = 500
epochs = 100

# Create RNN
model = create_RNN(x_train = data_train,
                   K = K,
                   n_lstm = 200,
                   loss = 'categorical_crossentropy',
                   optimizer = 'adam')
print(model.summary())

# Fit model
model.fit(data_train,
          y_train,
          epochs = epochs, 
          batch_size = batch_size)

# Save model
model.save('model_nlp_reviews2.h5')

# =============================================================================
# 4. Evaluate
# =============================================================================
# Obtain predictions
y_pred = model.predict(data_test)

# Obtain original values (not one-hot encoded)
if type(y_test) != list:
    y_test = [np.argmax(x) for x in y_test] 
y_pred = [np.argmax(x) for x in y_pred] 

# Evaluate results
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ", cm)
print("Precision: ", precision_score(y_test, y_pred, average='macro'))
print("Recall: ", recall_score(y_test, y_pred, average='macro'))
print("f1_score: ", f1_score(y_test, y_pred, average='macro'))

