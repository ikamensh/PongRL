import numpy as np

class My_Atari_Mock:

    def __init__(self):
        self.countdown = None

    def reset(self):
        self.countdown = 100
        return np.zeros(shape=[210,160,3], dtype=np.float32)


    def step(self, action):
        self.countdown -=1
        return np.ones(shape=[210, 160, 3], dtype=np.float32), 0.2, self.countdown <= 0, None