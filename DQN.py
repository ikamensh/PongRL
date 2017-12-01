import tensorflow as tf
from model import define_model

from play import rollout

state_size = [None, 80, 80, 4]
single_image_shape=[1, 80, 80, 4]
disc_rate=0.99

sess = tf.Session()

with tf.name_scope("active_model"):
    inp = tf.placeholder(dtype=tf.float32, shape = state_size)
    model = define_model(inp)
    action_choice = tf.reshape(tf.multinomial(model.output, num_samples=1), shape=[1])

with tf.name_scope("frozen_model"):
    inp_frozen = tf.placeholder(dtype=tf.float32, shape = state_size)
    frozen_model = define_model(inp_frozen)
    frozen_model.trainable = False


def loss(a, r):
    with tf.name_scope("loss"):


        rng = tf.reshape(tf.range(tf.shape(a)[0]), shape=[-1])
        indices = tf.stack([rng, a], axis=1)
        pred_q = tf.gather_nd(model.output, indices) # input is to be s1


        target = r + disc_rate*tf.reduce_max(frozen_model.output) # frozen_input is to be s2
        loss_abs_error = tf.reduce_mean(abs(pred_q - target),axis=0)

        return loss_abs_error

action = tf.placeholder(dtype=tf.int32, shape=[None])
reward = tf.placeholder(dtype=tf.float32, shape=[None])

mae_loss = loss(action, reward)

optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3)
training_step = optimizer.minimize(mae_loss)


writer = tf.summary.FileWriter('./my_graph', sess.graph)
writer.close()

sess.run(tf.global_variables_initializer())

from time import time
t = time()


def timestamp(i):
    global t
    dt = time() - t
    t = time()
    print("{} done in {} sec".format(i, dt))

t = time()

# timestamp("500 rollouts")

#Experience exp:    def __init__(self,s1, a, r, s2)
import random
import numpy as np



def stack_batch(sample):
    s1_stack, a_stack, r_stack, s2_stack = [],[],[],[]
    for exp in sample:
        s1_stack.append(exp.s1)
        a_stack.append(exp.a)
        r_stack.append(exp.r)
        s2_stack.append(exp.s2)
    return np.vstack(s1_stack), \
           np.reshape(np.vstack(a_stack), newshape=[-1]), \
           np.reshape(np.vstack(r_stack),newshape=[-1]), \
           np.vstack(s2_stack)



def train_on_batch_of_size( size , exp_buff ):
    b_s1, b_a, b_r, b_s2 = stack_batch(random.sample(exp_buff, size))
    sess.run(training_step, feed_dict={inp: b_s1, action: b_a, reward: b_r, inp_frozen: b_s2})

from matplotlib import pyplot as plt

def training_loop():

    rewards_progression = [0]

    for step in range(int(1e5)):
        exp_buff=[]
        rewards = 0
        for i in range(500):
            exps, r = rollout(action_choice, inp, sess)
            exp_buff.extend(exps)
            rewards += r
            rewards_progression.append( rewards / 500 )

        for i in range(100):
            train_on_batch_of_size(8, exp_buff)

        if step % 5 == 0:
            w = model.get_weights()
            frozen_model.set_weights(w)

        if step%50 == 0:
            timestamp(step)
            plt.clf()
            plt.plot(np.array(rewards_progression))
            plt.savefig("progress.png".format(step))

training_loop()

