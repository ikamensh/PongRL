""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle
import gym
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten, MaxPool2D
from keras.optimizers import RMSprop
from keras.losses import mean_squared_logarithmic_error
from datetime import datetime

# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
render = False


# model initialization
D = 80  # input dimensionality: 80x80 grid

model = Sequential()
model.add(Convolution2D(4, (3, 3), dilation_rate=(2, 2),input_shape=(D,D,1)))
model.add(MaxPool2D())
model.add(Convolution2D(4, (3, 3), dilation_rate=(2, 2)))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(RMSprop(), mean_squared_logarithmic_error)


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).reshape(1,80,80,1)

A=np.ones((210,160,3))
B=prepro(A)

print(model.predict_proba(B))

model.summary()