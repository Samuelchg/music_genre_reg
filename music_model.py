from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense

def model():
    model = Sequential()
    model.add(Dense(30, input_dim=518, init="uniform",activation="relu"))
    model.add(Dense(40, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(20, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(8))
    model.add(Activation("softmax"))
    return model