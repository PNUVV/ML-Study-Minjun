import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [11, 22, 33, 44, 53, 66, 77, 87, 95]
model = Sequential()
model.add(Dense(1, input_dim=1, activation='linear'))

sgd = optimizers.SGD(learning_rate=0.01)
 
model.compile(optimizer=sgd, loss='mse', metrics=['mse'])

model.fit(x, y, epochs=100)
print(model.predict(x))
