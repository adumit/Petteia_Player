import GameBoard as gb
import numpy as np

"""
- The neural network should take the 64x4 matrix that represents a move as input
- Should output a score from [-1, +1]
"""

import tensorflow as tf

# ------------------- Function Definitions -------------------
def weight_variable(shape, name):
  return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv3d(x, W, strides):
  return tf.nn.conv3d(x, W, strides=strides, padding='SAME')

def max_pool3d_2x2(x, ksize, strides):
  return tf.nn.max_pool3d(x, ksize=ksize, strides=strides, padding='SAME')

# ------------------- Network Definition -------------------

x = tf.placeholder(tf.float32, shape=[None, 256])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

x_image = tf.reshape(x, [-1, 4, 8, 8, 1])

# first convolutional layer
W_conv1 = weight_variable([4, 2, 2, 1, 32], 'W_conv1')
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1, strides=[1,4,1,1,1]) + b_conv1)
h_pool1 = max_pool3d_2x2(h_conv1, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1])

# second convolutional layer
W_conv2 = weight_variable([1, 4, 4, 32, 64], 'W_conv2') ### TODO: THIS CALL IS QUESTIONABLE... why 1, 4, 4 as first three args
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2, strides=[1,4,1,1,1]) + b_conv2)
h_pool2 = max_pool3d_2x2(h_conv1, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1])


# densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024], 'W_fc1')
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# softmax for classifying as good move or bad move
W_fc2 = weight_variable([1024, 1], 'W_fc2')
b_fc2 = bias_variable([1])
y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

squared_error = tf.square(y_conv-y_)
train_step = tf.train.AdamOptimizer(1e-4).minimize(squared_error)

init_op = tf.initialize_all_variables()
saver = tf.train.Saver()

# --------------- Running the network ---------------

# To evaluate a gameboard, run this with a running session:
# print(sess.run(y, feed_dict={x: INPUTS}))

# To train on a gameboard, run this:
# train_step.run(feed_dict={x: INPUTS, y_: LABELS})
# Where x: is the input game boards and y is the labels for each


def main():
    nn_saver_dir = '/nn_checkpoints'

    with tf.Session() as sesh:
        sesh.run(init_op)

        # saver.restore(sesh, '/nn_checkpoints/MODELGOESHERE.ckpt'
        # print("Model restored")

        games_played = 0
        while(True):
            # ----- Play a game of Petteia -----
            # Generate gameboards, one for each side
            gameboardRed = gb.GameBoard()
            gameBoardBlack = gb.GameBoard()
            for iter in range(200):
                # Make a red move
                possible_moves = gameboardRed.generate_moves()
                move_scores = []
                for i in range(len(possible_moves)):
                    matrix_board = gameboardRed.to_matrix(possible_moves[i])
                    move_scores.append(sesh.run(y_conv, feed_dict={x: matrix_board}))
                gameboardRed.make_move(possible_moves[np.argmax(move_scores)])
                gameBoardBlack.make_move(possible_moves[np.argmax(move_scores)])



