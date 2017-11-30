import numpy as np
import gym
from dqn.exp_buffer import Experience
from dqn.Atari_mock import My_Atari_Mock

#env = gym.make("Pong-v0")
env = My_Atari_Mock()

experiences_buffer = []

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


def rollout(action_choice_op, inp_placeholder, sess):
    observation1 = init_stack(shrink(env.reset()))
    done = False
    while done is False:
        action_chosen = sess.run(action_choice_op, feed_dict={inp_placeholder: observation1})

        total_r = 0.
        four_obs = []
        for i in range(4):
            #print(action_chosen)
            obs, r, done, _ = env.step(action_chosen)
            total_r += r
            four_obs.append(shrink(obs))
        observation2 = np.concatenate(four_obs, axis=3)
        experiences_buffer.append(Experience(observation1, action_chosen, np.array([total_r]), observation2))
        observation1 = observation2


#sess.run(tf.global_variables_initializer())