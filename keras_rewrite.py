""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle
import gym
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten, MaxPool2D
from keras.optimizers import adam
from keras.losses import binary_crossentropy
from datetime import datetime

callback = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True,
             write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

# hyperparameters
batch_size = 10  # every how many episodes to do a param update?
gamma=0.99
resume = False  # resume from previous checkpoint?
render = False


env = gym.make("Pong-v0")

# model initialization
D = 80  # input dimensionality: 80x80 grid

if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = Sequential()
    model.add(Flatten(input_shape=(D,D,1)))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(adam(lr=1e-5), binary_crossentropy)


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).reshape(1,80,80,1)

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(list(range(0, r.size))):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

observation = env.reset()
prev_x = None  # used in computing the difference frame
x_train, y_train= None, None

running_reward = None
reward_sum = 0
episode_number = 0
x_train_episode, y_train_episode, r_episode = [], [], []
while True:
    if render: env.render()

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(shape=(1,D,D,1))
    prev_x = cur_x
    x_train_episode.append(x)

    # forward the policy network and sample an action from the returned probability
    up_prob = model.predict_proba(x, verbose=0)
    action = 2 if np.random.uniform() < up_prob else 3  # roll the dice!
    y = 1 if action == 2 else 0  # a "fake label"
    y_train_episode.append(y)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward
    r_episode.append(reward)

    if done:  # an episode finished
        print(len(x_train_episode))
        episode_number += 1
        x_train_episode = np.array(x_train_episode).reshape(-1,D,D,1)
        y_train_episode = np.array(y_train_episode)

        temp = np.zeros_like(y_train_episode)
        temp = 0.5
        y_train_episode = temp + 0.5*y_train_episode*discount_rewards(np.array(r_episode))

        print("x_train_episode has shape of " + str(x_train_episode.shape))
        print("y_train_episode has shape of " + str(y_train_episode.shape))
        print(y_train_episode)

        x_train = x_train_episode if x_train is None else np.concatenate((x_train, x_train_episode),axis=0)
        y_train = y_train_episode if y_train is None else np.concatenate((y_train, y_train_episode),axis=0)
        x_train_episode, y_train_episode, r_episode = [], [], []

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            model.fit(np.array(x_train),np.array(y_train),epochs=3, verbose=1, callbacks=[callback])
            x_train, y_train= None, None


        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f\n' % (reward_sum, running_reward))

        if episode_number%50==0:
            with open('learnlog_karpathy.log', 'a+') as f:
                f.write('| %d | %f | %f | %s' % (episode_number, reward_sum, running_reward, str(datetime.now())))


        #prepare for new run of the simulation
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None

    if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
        print(('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
