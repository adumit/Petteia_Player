import GameBoard as gb
import numpy as np
import math
from random import random

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

x = tf.placeholder(tf.float32, shape=[8, 8, 4])
y_ = tf.placeholder(tf.float32, shape=[1])

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
W_fc1 = weight_variable([512, 1024], 'W_fc1')
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 512])
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


if __name__ == "__main__":
    nn_saver_dir = '/nn_checkpoints'

    should_restore = False

    with tf.Session() as sesh:

        if should_restore:
            saver.restore(sesh, '/nn_checkpoints/MODELGOESHERE.ckpt')
            print("Model restored")
        else:
            sesh.run(init_op)

        games_played = 0
        while(True):
            # ----- Play a game of Petteia -----
            # Generate gameboards, one for each side
            moves_made_red = []
            # moves_made_black = []
            gameboardRed = gb.GameBoard()
            gameboardBlack = gb.GameBoard()
            for iter in range(100):
                # Make a move for red
                possible_moves = gameboardRed.generate_moves()
                move_scores = []
                for i in range(len(possible_moves)):
                    try:
                        matrix_board = gameboardRed.to_matrix(possible_moves[i])
                    except TypeError:
                        print(possible_moves[i])
                        print(gameboardRed.print_board())
                        break
                    move_scores.append(sesh.run(y_conv, feed_dict={x: matrix_board}))
                # With the best move found, move on both red and black boards
                best_move = possible_moves[np.argmax(move_scores)]
                moves_made_red.append(gameboardRed.to_matrix(best_move))
                gameboardRed.make_move(best_move)
                move_for_black = ((7 - best_move[0][0], 7 - best_move[0][1]), (7 - best_move[1][0], 7 - best_move[1][1]))
                gameboardBlack.make_move(move_for_black)

                # Make a move for black
                possible_moves = gameboardBlack.generate_moves()
                move_scores = []
                for i in range(len(possible_moves)):
                    matrix_board = gameboardBlack.to_matrix(possible_moves[i])
                    move_scores.append(sesh.run(y_conv, feed_dict={x: matrix_board}))
                # With the best move found, move on both red and black boards
                best_move = possible_moves[np.argmax(move_scores)]
                # moves_made_black.append(gameboardBlack.to_matrix(best_move))
                gameboardBlack.make_move(best_move)
                move_for_red = ((7 - best_move[0][0], 7 - best_move[0][1]), (7 - best_move[1][0], 7 - best_move[1][1]))
                gameboardRed.make_move(move_for_red)

            # Print out for debugging
            print("RED SIDE BOARD")
            gameboardRed.print_board()

            # Evaluate the board,
            # Winning outright should be worth a lot
            # Winning should be scaled by the number of pieces a side won by
            # The effect should not be linear, use diff^1.5
            # If the sides tied, still adjust the weights slightly
            num_red = len([x for x in gameboardRed.pos if x])
            num_black = len([x for x in gameboardRed.neg if x])
            if num_red > num_black:
                red_score = math.pow((num_red - num_black), 1.5)
            elif num_black > num_red:
                red_score = -math.pow((num_black - num_red), 1.5)
            else: # TIED, slightly change weights
                red_score = random() - .5 # random() generates numbers on (0,1) subtract 0.5 to get to (-.5, .5)
            # Reward for winning a game entirely
            if num_red == 0:
                red_score -= 5
            elif num_black == 0:
                red_score +=5

            for move in moves_made_red:
                train_step.run(feed_dict={x: move, y_: np.array([red_score])})



