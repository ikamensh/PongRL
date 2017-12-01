from keras.layers import Input
from keras.activations import relu
from keras.engine import Model
from keras.layers import Convolution2D, Dense, Flatten
import tensorflow as tf
from play import action_space_size


def define_model(input_tensor):
    with tf.name_scope("Q_NET"):
        inputL = Input(tensor=input_tensor)
        h1 = Convolution2D(filters=32, kernel_size=(5,5), strides=(4,4), activation=relu) (inputL)
        h2 = Convolution2D(filters=64, kernel_size=(3,3), strides=(2,2), activation=relu) (h1)
        h3 = Convolution2D(filters=64, kernel_size=(3,3), activation=relu) (h2)
        f = Flatten() (h3)
        h4 = Dense(512, activation=relu) (f)
        out = Dense(action_space_size) (h4)
        return Model(inputL, out)