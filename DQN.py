import tensorflow as tf

from model import define_model
from play import rollout
from train import timestamp, stack_batch
import random
from collections import deque


import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')


state_size = [None, 80, 80, 4]
single_image_shape=[1, 80, 80, 4]
disc_rate=0.99

sess = tf.Session()

with tf.name_scope("active_model"):
    inp = tf.placeholder(dtype=tf.float32, shape = state_size)
    model = define_model(inp)
    action_choice = tf.reshape(tf.multinomial(model.output, num_samples=1), shape=[1])
    greedy_action =  tf.argmax(model.output)

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
        tf.summary.scalar("loss", loss_abs_error)

        return loss_abs_error

action = tf.placeholder(dtype=tf.int32, shape=[None])
reward = tf.placeholder(dtype=tf.float32, shape=[None])

mae_loss = loss(action, reward)

optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3)
training_step = optimizer.minimize(mae_loss)


writer = tf.summary.FileWriter('./my_graph', sess.graph)
all_summaries = tf.summary.merge_all()

sess.run(tf.global_variables_initializer())

def train_on_batch_of_size( size , exp_buff, make_summary):

    b_s1, b_a, b_r, b_s2 = stack_batch(random.sample(exp_buff, size))
    if make_summary:
        summaries, _ = sess.run([all_summaries, training_step], feed_dict={inp: b_s1, action: b_a, reward: b_r, inp_frozen: b_s2})
        return summaries
    else:
        sess.run( training_step, feed_dict={inp: b_s1, action: b_a, reward: b_r, inp_frozen: b_s2})

def training_loop():

    fail_buff = deque(maxlen=2000)
    success_buff = deque(maxlen=2000)

    timestamp("looking for random success")
    r = 0
    while r < 1:
        exps, r = rollout(isRandom=True)
        fail_buff.extend(exps)
    success_buff.extend(exps)

    timestamp("starting training!")
    for step in range(int(1e5)):


        rewards = 0
        for i in range(3):
            exps, r = rollout(action_choice, inp, sess, isRandom=False)
            rewards += r
            if r == 1:
                success_buff.extend(exps)
            else:
                fail_buff.extend(exps)

        summary = tf.Summary()
        summary.value.add(tag='training reward', simple_value=rewards / 3)
        writer.add_summary(summary, step)




        for i in range(30):
            train_on_batch_of_size(8, success_buff, False)
        for i in range(10):
            train_on_batch_of_size(8, fail_buff, False)

        summary = train_on_batch_of_size(8, fail_buff, True)
        writer.add_summary(summary, step)

        w = model.get_weights()
        frozen_model.set_weights(w)

        if step%10 == 0:
            timestamp(step)
            _, r = rollout(greedy_action, inp, sess, isRandom=False)
            summary = tf.Summary()
            summary.value.add(tag='greedy reward', simple_value=r)
            writer.add_summary(summary, step)

            writer.flush()


        if step % 20 == 0:
            model.save_weights("weights{}.h5".format(step//1000), overwrite=True)

training_loop()
writer.close()


