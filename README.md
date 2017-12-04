# PongRL
Implementation of DQN in tensorflow / keras.

Environment solved: gym atari Pong, using a network of 3 layer Convolutional + 1 hidden Dense layer.

Requirements: tensorflow 1.4, keras 2

to start training run
    ```
    python DQN.py
    ```
    
    
to monitor training, connect tensorboard :
    ```
    tensorboard logdir=my_graph
    ```
    

Overview (Tensorboard) : 

![Alt text](/screenshots/2017-12-04_10h11_21.png?raw=true "Overview of the training")

Model:

![Alt text](/screenshots/2017-12-04_10h15_37.png?raw=true "trained model")
