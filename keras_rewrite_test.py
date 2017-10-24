""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle
import gym
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten, MaxPool2D
from keras.optimizers import adam
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
model.add(Flatten(input_shape=(D,D,1)))
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(adam(lr=1e-4), mean_squared_logarithmic_error)


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(list(range(0, r.size))):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

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


D = None

for _ in range(5):
    C = []
    for _ in range(10):
        A=np.ones((210,160,3))
        B=prepro(A)
        C.append(B)
    C *= np.array(C) * 1.01
    C = C.reshape(-1 ,80, 80, 1)
    D = C if D is None else np.concatenate((D,C),axis=0)




temp = np.zeros_like(D)
temp = 0.5
y_train_episode = temp + 0.5*D.ravel()*discount_rewards(np.array(range(320000)))

print(D.shape)
print(y_train_episode)
print(model.predict_proba(D))
