import gym
import numpy as np
from exp_buffer import Experience
from Atari_mock import My_Atari_Mock
from random import random

#env = My_Atari_Mock()
env = gym.make("Pong-v0")
action_space_size = 6



def shrink(pic):
    #print("shrinking pic with shape: " + str(pic.shape))

    black_n_white = sum( pic[:,:,i] for i in range(3) ) / 3
    crop = black_n_white[25:-25]
    downsample_even = crop[::2, ::2]
    downsample_odd = crop[1::2, 1::2]
    downsample = (downsample_even + downsample_odd) / 2

    return np.reshape(downsample, [1, 80, 80, 1])

def init_stack(downsampled):
    #print("init stack uses pic with shape: " + str(downsampled.shape))

    copied = np.repeat(downsampled,4, axis=3)

    return copied


def rollout(action_choice_op = None, inp_placeholder = None, sess = None, isRandom = True):
    experiences_buffer = []
    # print("env_reset")
    observation1 = init_stack(shrink(env.reset()))
    done = False
    r_episode = 0
    while done is False:
        if isRandom:
            action_chosen = env.action_space.sample()
        else:
            action_chosen = sess.run(action_choice_op, feed_dict={inp_placeholder: observation1})

        total_r = 0.
        four_obs = []
        for i in range(4):
            #print("env_step with action= {}".format(action_chosen))
            obs, r, done, _ = env.step(action_chosen)
            total_r += r
            four_obs.append(shrink(obs))
            # if done:
            #     break

        observation2 = np.concatenate(four_obs, axis=3)
        experiences_buffer.append(Experience(observation1, action_chosen, np.array([total_r]), observation2))
        observation1 = observation2
        r_episode += total_r

    return experiences_buffer, total_r


#sess.run(tf.global_variables_initializer())