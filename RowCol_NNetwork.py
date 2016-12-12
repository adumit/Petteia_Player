import GameBoard as gb
import numpy as np
import math
from random import random
from datetime import datetime
import sys

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

# ------------------- Red-side Network Definition -------------------
x_red = tf.placeholder(tf.float32, shape=[8, 8, 4])
y__red = tf.placeholder(tf.float32, shape=[1])

x_image_red = tf.reshape(x_red, [-1, 4, 8, 8, 1])

# First row convolution
W_row_conv1_red = weight_variable([4, 1, 8, 1, 8], 'W_row_conv1_red')
b_row_conv1_red = bias_variable([8])
h_row_conv1_red = tf.nn.relu(conv3d(x_image_red, W_row_conv1_red, strides=[1,4,1,1,1]) + b_row_conv1_red)
h_row_pool1_red = max_pool3d_2x2(h_row_conv1_red, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1])

# Second row convolutional layer
W_row_conv2_red = weight_variable([1, 2, 2, 8, 16], 'W_row_conv2_red')
b_row_conv2_red = bias_variable([16])
h_row_conv2_red = tf.nn.relu(conv3d(h_row_pool1_red, W_row_conv2_red, strides=[1,1,1,1,1]) + b_row_conv2_red)
h_row_pool2_red = max_pool3d_2x2(h_row_conv2_red, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1])

# First column convolution
W_col_conv1_red = weight_variable([4, 8, 1, 1, 8], 'W_col_conv1_red')
b_col_conv1_red = bias_variable([8])
h_col_conv1_red = tf.nn.relu(conv3d(x_image_red, W_col_conv1_red, strides=[1,4,1,1,1]) + b_col_conv1_red)
h_col_pool1_red = max_pool3d_2x2(h_col_conv1_red, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1])

# Second column convolutional layer
W_col_conv2_red = weight_variable([1, 2, 2, 8, 16], 'W_col_conv2_red')
b_col_conv2_red = bias_variable([16])
h_col_conv2_red = tf.nn.relu(conv3d(h_col_pool1_red, W_col_conv2_red, strides=[1,1,1,1,1]) + b_col_conv2_red)
h_col_pool2_red = max_pool3d_2x2(h_col_conv2_red, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1])

# Combination convolutional layer
W_comb_conv_red = weight_variable([1, 2, 2, 16, 32], 'W_comb_conv2_red')
b_comb_conv_red = bias_variable([32])
h_comb_conv_red = tf.nn.relu(conv3d(tf.concat(3, [h_col_pool2_red, h_row_pool2_red]), W_comb_conv_red, strides=[1,1,1,1,1]) + b_comb_conv_red)
h_comb_pool_red = max_pool3d_2x2(h_comb_conv_red, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1])

# densely connected layer
W_fc1_red = weight_variable([64, 320], 'W_fc1_red')
b_fc1_red = bias_variable([320])
h_pool2_flat_red = tf.reshape(h_comb_pool_red, [-1, 64])
h_fc1_red = tf.nn.relu(tf.matmul(h_pool2_flat_red, W_fc1_red) + b_fc1_red)

# softmax for classifying as good move or bad move
W_fc2_red = weight_variable([320, 1], 'W_fc2_red')
b_fc2_red = bias_variable([1])
y_conv_red = tf.matmul(h_fc1_red, W_fc2_red) + b_fc2_red

# This loss function should seek to choose the best moves that lead to the best outcome
# Mutliplying a good move * good outcome will lead to a very large (negative here) number.
# Which is the best minimization. A high predicted move * a bad outcome will be very bad,
# which satifies the goal
loss_func_red = -tf.log(y_conv_red)*y__red
train_step_red = tf.train.AdamOptimizer(1e-4).minimize(loss_func_red)

# ------------------- Black-side Network Definition -------------------
x_black = tf.placeholder(tf.float32, shape=[8, 8, 4])
y__black = tf.placeholder(tf.float32, shape=[1])

x_image_black = tf.reshape(x_black, [-1, 4, 8, 8, 1])

# first convolutional layer
W_conv1_black = weight_variable([4, 4, 4, 1, 32], 'W_conv1_black')
b_conv1_black = bias_variable([32])
h_conv1_black = tf.nn.relu(conv3d(x_image_black, W_conv1_black, strides=[1,4,1,1,1]) + b_conv1_black)
h_pool1_black = max_pool3d_2x2(h_conv1_black, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1])

# second convolutional layer
W_conv2_black = weight_variable([1, 4, 4, 32, 64], 'W_conv2_black')
b_conv2_black = bias_variable([64])
h_conv2_black = tf.nn.relu(conv3d(h_pool1_black, W_conv2_black, strides=[1,4,1,1,1]) + b_conv2_black)
h_pool2_black = max_pool3d_2x2(h_conv1_black, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1])


# densely connected layer
W_fc1_black = weight_variable([512, 1024], 'W_fc1_black')
b_fc1_black = bias_variable([1024])
h_pool2_flat_black = tf.reshape(h_pool2_black, [-1, 512])
h_fc1_black = tf.nn.relu(tf.matmul(h_pool2_flat_black, W_fc1_black) + b_fc1_black)

# softmax for classifying as good move or bad move
W_fc2_black = weight_variable([1024, 1], 'W_fc2_black')
b_fc2_black = bias_variable([1])
y_conv_black = tf.matmul(h_fc1_black, W_fc2_black) + b_fc2_black

# This loss function should seek to choose the best moves that lead to the best outcome
# Mutliplying a good move * good outcome will lead to a very large (negative here) number.
# Which is the best minimization. A high predicted move * a bad outcome will be very bad,
# which satifies the goal
loss_func_black = -tf.log(y__black)*y_conv_black
train_step_black = tf.train.AdamOptimizer(1e-4).minimize(loss_func_black)


# --------------- Global Tensorflow variables ---------------
init_op = tf.initialize_all_variables()
saver = tf.train.Saver()

# --------------- Running the network ---------------

# To evaluate a gameboard, run this with a running session:
# print(sess.run(y, feed_dict={x: INPUTS}))

# To train on a gameboard, run this:
# train_step.run(feed_dict={x: INPUTS, y_: LABELS})
# Where x: is the input game boards and y is the labels for each


if __name__ == "__main__":
    nn_saver_dir = './nn_checkpoints/'

    # Constants for overall file
    should_restore = False
    should_write_output_to_file = False

    # Constants for game playing
    aggressive = True

    with tf.Session() as sesh:

        sesh.run(init_op)
        if should_restore:
            saver.restore(sesh, nn_saver_dir + 'aggressive_90_90_4by4input_checkpoint_1000.ckpt')
            print("Model restored")


        if should_write_output_to_file:
            sys.stdout = open('print_out.txt', 'w')

        games_played = 0
        while True:
            start_time = datetime.now()
            # ----- Play a game of Petteia -----
            # Generate gameboards, one for each side
            moves_made_red = []
            moves_made_black = []
            gameboardRed = gb.GameBoard()
            gameboardBlack = gb.GameBoard()
            for iter in range(100):
                # Make a move for red
                possible_moves = gameboardRed.generate_moves()
                if len(possible_moves) == 0:
                    break
                move_scores = np.zeros(len(possible_moves))
                for i in range(len(possible_moves)):
                    matrix_board = gameboardRed.to_matrix(possible_moves[i])
                    move_scores[i] = sesh.run(y_conv_red, feed_dict={x_red: matrix_board})
                # With the best move found, move on both red and black boards
                rand_choice = random()
                if (rand_choice < .9):
                    best_move = possible_moves[np.argmax(move_scores)]
                elif (rand_choice < .95) and len(possible_moves) > 1:
                    best_move = possible_moves[move_scores.argsort()[1]]
                elif (rand_choice < .985) and len(possible_moves) > 2:
                    best_move = possible_moves[move_scores.argsort()[2]]
                else:
                    best_move = possible_moves[math.floor(random() * len(move_scores))]
                moves_made_red.append(gameboardRed.to_matrix(best_move))
                gameboardRed.make_move(best_move)
                move_for_black = ((7 - best_move[0][0], 7 - best_move[0][1]), (7 - best_move[1][0], 7 - best_move[1][1]))
                gameboardBlack.make_move(move_for_black)

                # Make a move for black
                possible_moves = gameboardBlack.generate_moves()
                if len(possible_moves) == 0:
                    break
                best_move = None
                # Write in different ways for black to play here
                if aggressive:
                    capture_moves = gameboardBlack.generate_capture_moves(possible_moves)
                    if len(capture_moves) > 0:
                        move_scores = np.zeros(len(capture_moves))
                        for i in range(len(capture_moves)):
                            matrix_board = gameboardBlack.to_matrix(capture_moves[i])
                            move_scores[i] = sesh.run(y_conv_black, feed_dict={x_black: matrix_board})
                        best_move = capture_moves[np.argmax(move_scores)]

                if best_move is None:
                    move_scores = np.zeros(len(possible_moves))
                    for i in range(len(possible_moves)):
                        matrix_board = gameboardBlack.to_matrix(possible_moves[i])
                        move_scores[i] = sesh.run(y_conv_black, feed_dict={x_black: matrix_board})
                        if should_write_output_to_file:
                            if i == 0:
                                print(datetime.now() - before_pass)
                    # With the best move found, move on both red and black boards
                    rand_choice = random()
                    if (rand_choice < .9):
                        best_move = possible_moves[np.argmax(move_scores)]
                    elif (rand_choice < .94) and len(possible_moves) > 1:
                        best_move = possible_moves[move_scores.argsort()[1]]
                    elif (rand_choice < .98) and len(possible_moves) > 2:
                        best_move = possible_moves[move_scores.argsort()[2]]
                    else:
                        best_move = possible_moves[math.floor(random() * len(move_scores))]
                moves_made_black.append(gameboardBlack.to_matrix(best_move))
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
                # random() generates numbers on (0,1)
                # Want to get numbers in (-0.1, 0.1)
                red_score = random()*0.2 - 0.1
            # Reward for winning a game entirely
            if num_red == 0:
                red_score -= 5
            elif num_black == 0:
                red_score +=5
            black_score = -red_score

            for move in moves_made_red:
                train_step_red.run(feed_dict={x_red: move, y__red: np.array([red_score])})
            for move in moves_made_black:
                train_step_black.run(feed_dict={x_black: move, y__black: np.array([black_score])})
            games_played += 1
            print("Time to play and train on game " + str(games_played) + ": " + str(datetime.now() - start_time))

            if games_played % 200 == 0:
                saver.save(sesh, nn_saver_dir + 'aggressive_rowcol_arch_' + str(games_played) + '.ckpt')

