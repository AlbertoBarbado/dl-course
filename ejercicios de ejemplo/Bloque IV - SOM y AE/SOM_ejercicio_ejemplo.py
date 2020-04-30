# -*- coding: utf-8 -*-
# @author: Alberto Barbado Gonzalez
# Máster Deep Learning Structuralia

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Define the Winner Take All units
class SOM():
    
    def __init__(self, m, n, dim, num_iterations=1, learning_rate = 0.5, sigma = None):
        
        self._m = m
        self._n = n
        self._neighbourhood = []
        self._topography = []
        self._num_iterations = int(num_iterations) 
        self._learned = False
        self.dim = dim
        self.d = 0
        
        if sigma is None:
            sigma = max(m,n)/2.0    # Constant radius
        else:
            sigma = float(sigma)
    
        self.learning_rate = learning_rate
        self.learning_rate_0 = learning_rate
        self.sigma = sigma
        self.sigma_0 = sigma
        
        print('Network created with dimensions',m,n)
            
        # Weight Matrix and the topography of neurons
        self._W = tf.random.normal([m*n, dim], seed = 0)
        
        # Topography nos define un array de (30x30) x  2 (900 x 2) donde 
        # cada fila representa las coordenadas en el mapa de esa unidad. Por ejemplo
        # la unidad 5 de la lista está en las coordenadas [0,5], la 200 en la [6,20], y la ultima es la [30,30]
        self._topography = np.array(list(self._neuron_location(m, n)))
            
               
        
    def training(self, x, i):
            m = self._m
            n = self._n 
            
            # Finding the Winner and its location
            # Se obtiene la distancia para cada unidad de la malla con respecto a la entrada
            # tf.stack([x for i in range(m*n)]) crea una matriz con el vector de entrada repetido para poderselo restar a cada vector de pesos
            # Asi, si la entrada es [1x3] y la malla es 30x30 (900) la salida es [900 x 3]
            # Se le resta el vector de pesos y se eleva al cuadrado el resultado con tf.pow(...,2)
            # Se tiene en este punto una matriz [900 x 3] donde cada fila es la diferencia de la entrada con cada unidad
            # Asi, para cada fila, hay que sumar los valores de las columnas, ya que estos son los valores de cada variable
            # Eso lo sumamos con tf.reduce_sum. Con ello, nos queda un vector [900 x 1]
            # Con tf.sqrt sacamos la raiz cuadrada, y en ese vector final de [900 x 1] cada fila es la distancia de esa unidad a la entrada
            # De todas ellas, elegimos la mejor para el BMU. tf.argmin nos da el id de esa BMU
            d = tf.sqrt(tf.reduce_sum(tf.pow(self._W - tf.stack([x for i in range(m*n)]),2),1))
            self.BMU_idx = tf.argmin(d,0)
            self.d = d
            
            # slice_start = tf.pad(tf.reshape(self.WTU_idx, [1]),np.array([[0,1]]))
            # self.WTU_loc = tf.reshape(tf.slice(self._topography, slice_start,[1,2]), [2])
            # Obtenemos, para esa unidad, su posición en el mapa (topografia), y expresamos
            # el vector de posicion (numpy array) como un tensor de dimension (1,2)
            self.BMU_loc = tf.reshape(self._topography[self.BMU_idx], [1, 2])
            
            # Actualizacion del radio y del learning rate usando las ecuaciones
            # vistas en la teoria.
            # Change learning rate and radius as a function of iterations
            lambda_coeff = self._num_iterations/self.sigma_0
            learning_rate = self.learning_rate_0*np.exp(-i/lambda_coeff)
            sigma =  self.sigma_0*np.exp(-i/lambda_coeff)
            
            # Calculating Neighbourhood function
            # d_ij = tf.sqrt(tf.pow(self._W - tf.stack([x for i in range(m*n)]),2),1)
            beta_ij = np.exp((-d**2)/(2*sigma**2))
            
            # Choose Neighbours
            neighbs = [self._check_point(p[0], p[1], self.BMU_loc.numpy()[0][0], self.BMU_loc.numpy()[0][1], sigma) for p in self._topography]
            
            # Update weights
            # Actualizamos los pesos. Aqui definimos el incremento a sumar a cada peso previo.
            # Para las unidades que no están dentro del area de la BMU, su actualización se 
            # multiplica por 0 y no se cambian por ello sus pesos.
            weight_multiplier = tf.math.multiply(beta_ij, neighbs)
            weight_multiplier = tf.math.multiply(learning_rate, weight_multiplier)
            
            # Tras ello, obtenemos la diferencia entre el vector de entrada y cada peso
            # Esa diferencia se multiplica por el valor obtenido antes. 
            # Para poderlo multiplicar, hay que haver un tf.stack para que 
            # Todo el multiplicador pueda multiplicar a cada peso de la conexión entre
            # el nodo y la entrada. Ese multiplicador afecta a toda la unidad, y por ello
            # afecta por igual a todas sus conexiones
            delta_W = tf.subtract(tf.stack([x for i in range(m * n)]),self._W)
            weight_multiplier = tf.stack([weight_multiplier for i in range(n_dim)], axis=1)
            update_value = tf.multiply(weight_multiplier, weight_multiplier)
            update_value = tf.multiply(weight_multiplier, delta_W)
            
            # Definido el valor de actualización de los pesos de cada unidad, se 
            # actualiza la matriz de pesos y se guarda en el objeto.
            new_W = self._W + update_value
            self._W = new_W

    
    # La función fit sirve para iterar por cada epoch, y en cada una de ellas 
    # recorrer todos los registros. La matriz de entrada se pone de forma 
    # aleatoria previamente.
    def fit(self, X):
        
        np.random.shuffle(X)
        X = tf.cast(X, tf.float32)
        
        for i in range(self._num_iterations):
            for x in X:
                 self.training(x,i)
        
        # Store a centroid grid for easy retrieval
        # Guardamos en un formato [n,m,dimensiones] los pesos respecto a la entrada
        # de cada unidad.
        self._Wts = list(self._W)
        self._locations = list(self._topography)
        self._learned = True
        
    # Comprobar si un punto esta dentro del radio alrededor de otro
    def _check_point(self, x, y, center_x, center_y, radius):
                check = (x - center_x)**2 + (y - center_y)**2 < radius**2
                if check == True:
                    return 1
                else:
                    return 0
    
    # Obtener la BMU
    def winner(self, x):
        if not self._learned:
            raise ValueError("SOM not trained yet")
            
        return self.BMU_loc.numpy()
    
    # Funcion para generar las posiciones en la maya para un array de unidades
    def _neuron_location(self,m,n):
        for i in range(m):
            for j in range(n):
                yield np.array([i,j])

    # Con esta funcion recorremos todos los datapoints y para cada uno de ellos
    # obtenemos la BMU asociada. Para ello, calculamos la norma de la diferencia
    # de ese vector de entrada con los pesos de cada unidad: ||v-w_ij||. De todos
    # esos valores, se selecciona como BMU para ese vector de entrada la que de
    # menor resultado.
    def map_vects(self, X):
        if not self._learned:
            raise ValueError("SOM not trained yet")

        to_return = []
        for vect in X:
            min_index = min([i for i in range(len(self._Wts))],
                            key=lambda x: np.linalg.norm(vect -
                                                         self._Wts[x]))
            to_return.append(self._locations[min_index])
            
        return to_return
         
    # Devolver la malla en la que cada unidad tenga asociada su distancia. 
    # Asi, es una matriz de (m, n) donde cada registro es la distancia final de esa unidad (i,j)
    def distance_map(self):
        if not self._learned:
            raise ValueError("SOM not trained yet")
        mapping = tf.reshape(self.d, shape=(self._m, self._n)).numpy()

        return mapping

# =============================================================================
# 1. Preparar datos
# =============================================================================
# Cargar datos
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.copy().drop(columns=['CustomerID', 'Class']).values
y = dataset.iloc[:, -1].values # Variable que dice si la application del customer fue aprovada o no

# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1)) # Para que todos los valores estén entre 0 y 1
X = sc.fit_transform(X)

# =============================================================================
# 2. Entrenar modelo
# =============================================================================
# Hyperparametros
n_dim = np.shape(X)[1]
grid_size = (10, 10)
sigma = 10
num_iterations = 50
learning_rate = 0.5

# Fit & Train
som =  SOM(grid_size[0], grid_size[1], dim=n_dim, 
           num_iterations = num_iterations,
           learning_rate = learning_rate,
           sigma = sigma)
som.fit(X)

# Malla con el MID de cada unidad
distance_matrix = som.distance_map().T

# =============================================================================
# 3. Visualización de resultados
# =============================================================================
from pylab import bone, pcolor, colorbar
bone() # Inicializo la ventana de visualizacion
pcolor(distance_matrix) # Para pintar el som. El .T para poner la matriz traspuesta. Lo que pinto es el MID de los nodos
colorbar() # Para tener la leyenda de colores. Veré que los MID van de 0 a 1, porque están escalados

max_value = np.amax(distance_matrix)
min_value = np.amin(distance_matrix)

list_mid = list(np.reshape(distance_matrix, (grid_size[0]*grid_size[1],)))
list_mid.sort()
list_mid = [j for j in list_mid if j > 1.48]
list_idx = [np.where(distance_matrix==j) for j in list_mid]
list_idx = [[idx_max[0][0], idx_max[1][0]] for idx_max in list_idx]

mappings = som.map_vects(X)

# =============================================================================
# 4. Detectar anomalias
# =============================================================================
df_users = pd.DataFrame()
for i, x in enumerate(X):  # i son los valores de los indices, y x son los distitos vectores de customers en cada iteracion, y reccorro el dataset X con enumerate(X)
    w = mappings[i] # BMU para ese registro
    # Si el BMU coincide con las unidades de los outliers, lo identificamos como fraudulento
    is_fraud = False
    # Fraude si la variable tiene de BMU una de las de la lista de fraudulentas
    if [w[0], w[1]] in list_idx:
        is_fraud = True
    # Guardar resultados
    df_users = df_users.append(pd.DataFrame({'user':[dataset.iloc[i]['CustomerID']],
                                             'mapping':[w],
                                             'is_fraud':[is_fraud],
                                             'credit_approval':[dataset.iloc[i]['Class']]}))

