from time import time
import numpy as np


t = time()

def timestamp(i):
    global t
    dt = time() - t
    t = time()
    print("{} done in {} sec".format(i, dt))


def stack_batch(sample):
    s1_stack, a_stack, r_stack, s2_stack = [],[],[],[]
    for exp in sample:
        s1_stack.append(exp.s1)
        a_stack.append(exp.a)
        r_stack.append(exp.r)
        s2_stack.append(exp.s2)


    print(s.shape for s in s1_stack)
    print(a_stack)
    print(r_stack)
    print(s.shape for s in s2_stack)

    return np.vstack(s1_stack), \
           np.reshape(np.vstack(a_stack), newshape=[-1]), \
           np.reshape(np.vstack(r_stack),newshape=[-1]), \
           np.vstack(s2_stack)