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
batch_size = 10  # every how many episodes to do a param update?
resume = False  # resume from previous checkpoint?
render = False


env = gym.make("Pong-v0")

# model initialization
D = 80  # input dimensionality: 80x80 grid

if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = Sequential()
    model.add(Convolution2D(4, (3, 3), dilation_rate=(2, 2), input_shape=(D, D, 1)))
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
    return I.astype(np.float).reshape(80,80,1)

observation = env.reset()
prev_x = None  # used in computing the difference frame
x_train, y_train= [], []

running_reward = None
reward_sum = 0
episode_number = 0

while True:
    if render: env.render()
    x_train_episode, y_train_episode = [], []

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(shape=(1,D,D,1))
    prev_x = cur_x
    x_train_episode.append(x)

    # forward the policy network and sample an action from the returned probability
    up_prob = model.predict_proba(x.reshape(1,80,80,1), verbose=0)
    action = 2 if np.random.uniform() < up_prob else 3  # roll the dice!
    y = 1 if action == 2 else 0  # a "fake label"
    y_train_episode.append(y)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward


    if done:  # an episode finished
        episode_number += 1
        x_train_episode = np.array(x_train_episode)
        y_train_episode *=  np.array(y_train_episode)*reward_sum

        x_train.append(x_train_episode)
        y_train.append(y_train_episode)

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            model.fit(np.array(x_train),np.array(y_train),epochs=5, verbose=2)
        if episode_number % batch_size*4 == 0:
            x_train.clear()
            y_train.clear()

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))

        if episode_number%100==0:
            with open('learnlog_karpathy.log', 'a+') as f:
                f.write('| %d | %f | %f | %s' % (episode_number, reward_sum, running_reward, str(datetime.now())))


        if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))

        #prepare for new run of the simulation
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None

    if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
        print(('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))