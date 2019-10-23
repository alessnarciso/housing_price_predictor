#Hello World (in Machine Learning)
#First machine learning program
#Using the model to predict the relationship between x and y; y=2x-1

#libraries
import tensorflow as tf
import numpy as np
from tensorflow import keras

#single layer; single neuron
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

#training data
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

#Predict the output to the equation if x = 10, answer should be 19
print(model.predict([90.0]))
