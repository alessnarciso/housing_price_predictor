#House Pricing Predictor
#Exercise 1 from Coursera Intro to TensorFlow
#In this exercise you'll try to build a neural network that predicts the price of a house according to a simple formula.

#So, imagine if house pricing was as easy as a house costs 
#50k + 50k per bedroom, so that a 1 bedroom house costs 100k, 
#a 2 bedroom house costs 150k etc.

#How would you create a neural network that learns this relationship 
#so that it would predict a 7 bedroom house as costing close to 400k etc.

#########

#formula: price (in k) = 50 + 50*bedroom

import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

#training data
bedrooms = np.array([0.0, 1.0,  2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
prices = np.array([50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0], dtype=float)

model.fit(bedrooms, prices, epochs=1000)

#predict the price for num of bedrooms
bedroom_num = 7.0
predicted_price = (model.predict([bedroom_num]))

print("\nFor ", bedroom_num, " bedrooms, predicted total price = $" , predicted_price*1000)
