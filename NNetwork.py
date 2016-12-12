import GameBoard as gb
import numpy as np
import math
from random import random
from datetime import datetime
import sys
import re
import os
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

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
x_red = tf.placeholder(tf.float32, shape=[None, 8, 8, 4])
y__red = tf.placeholder(tf.float32, shape=[1])

x_image_red = tf.reshape(x_red, [-1, 4, 8, 8, 1])

# TODO: Try testing with different convolution sizes at the first layer
# first convolutional layer
W_conv1_red = weight_variable([4, 4, 4, 1, 32], 'W_conv1_red')
b_conv1_red = bias_variable([32])
h_conv1_red = tf.nn.relu(conv3d(x_image_red, W_conv1_red, strides=[1,4,1,1,1]) + b_conv1_red)
h_pool1_red = max_pool3d_2x2(h_conv1_red, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1])

# second convolutional layer
W_conv2_red = weight_variable([1, 2, 2, 32, 64], 'W_conv2_red')
b_conv2_red = bias_variable([64])
h_conv2_red = tf.nn.relu(conv3d(h_pool1_red, W_conv2_red, strides=[1,1,2,2,1]) + b_conv2_red)
h_pool2_red = max_pool3d_2x2(h_conv1_red, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1])


# densely connected layer
W_fc1_red = weight_variable([512, 1024], 'W_fc1_red')
b_fc1_red = bias_variable([1024])
h_pool2_flat_red = tf.reshape(h_pool2_red, [-1, 512])
h_fc1_red = tf.nn.relu(tf.matmul(h_pool2_flat_red, W_fc1_red) + b_fc1_red)

# softmax for classifying as good move or bad move
W_fc2_red = weight_variable([1024, 1], 'W_fc2_red')
b_fc2_red = bias_variable([1])
y_conv_red = tf.matmul(h_fc1_red, W_fc2_red) + b_fc2_red

# This loss function should seek to choose the best moves that lead to the best outcome
# Mutliplying a good move * good outcome will lead to a very large (negative here) number.
# Which is the best minimization. A high predicted move * a bad outcome will be very bad,
# which satifies the goal
loss_func_red = -y__red*tf.maximum(0.01, y_conv_red)
train_step_red = tf.train.AdamOptimizer(1e-4).minimize(loss_func_red)

# ------------------- Black-side Network Definition -------------------
x_black = tf.placeholder(tf.float32, shape=[None, 8, 8, 4])
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
h_conv2_black = tf.nn.relu(conv3d(h_pool1_black, W_conv2_black, strides=[1,1,2,2,1]) + b_conv2_black)
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
loss_func_black = -y__black*tf.maximum(0.01, y_conv_black)
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
    should_restore = True
    should_write_output_to_file = False
    test_output = False

    # Constants for game playing
    aggressive = True

    test_out = ""
    max_checkpoint_val = 0

    with tf.Session() as sesh:

        sesh.run(init_op)
        if should_restore:
            checkpoint_vals = []
            for root, dirs, files in os.walk(nn_saver_dir):
                for f in files:
                    checkpoint_vals.append(re.search(r'\d+', f))
            checkpoint_vals = [int(x.group()) for x in checkpoint_vals if x]
            max_checkpoint_val = max(checkpoint_vals)
            saver.restore(sesh, nn_saver_dir + 'aggressive_new_loss_checkpoint_' + str(max_checkpoint_val) +'.ckpt')
            print("Model restored")

        games_played = 0
        while games_played < 50001:
            if should_write_output_to_file:
                f = open('print_out.txt', 'w')
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
                before_pass = datetime.now()
                move_scores = sesh.run(y_conv_red, feed_dict={x_red: [gameboardRed.to_matrix(m) for m in possible_moves]})
                if should_write_output_to_file:
                    if iter == 0:
                        f.write("Time to pass all moves through network: " + str(datetime.now() - before_pass) + "\n")
                # With the best move found, move on both red and black boards
                rand_choice = random()
                if iter == 10:
                    if should_write_output_to_file:
                        f.write("Max score for iteration 10 and game " + str(games_played) + "was: " + str(np.max(move_scores)) + "\n")
                    else:
                        print(np.max(move_scores))
                if (rand_choice < .9):
                    best_move = possible_moves[np.argmax(move_scores)]
                elif (rand_choice < .95) and len(possible_moves) > 1:
                    best_move = possible_moves[move_scores.argsort()[1]]
                elif (rand_choice < .98) and len(possible_moves) > 2:
                    best_move = possible_moves[move_scores.argsort()[2]]
                else:
                    best_move = possible_moves[int(math.floor(random() * len(move_scores)))]
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
                        move_scores = sesh.run(y_conv_black, feed_dict={x_black: [gameboardBlack.to_matrix(m) for m in capture_moves]})
                        best_move = capture_moves[np.argmax(move_scores)]

                if best_move is None:
                    move_scores = sesh.run(y_conv_black, feed_dict={x_black: [gameboardBlack.to_matrix(m) for m in possible_moves]})
                    # With the best move found, move on both red and black boards
                    rand_choice = random()
                    if (rand_choice < .9):
                        best_move = possible_moves[np.argmax(move_scores)]
                    elif (rand_choice < .94) and len(possible_moves) > 1:
                        best_move = possible_moves[move_scores.argsort()[1]]
                    elif (rand_choice < .98) and len(possible_moves) > 2:
                        best_move = possible_moves[move_scores.argsort()[2]]
                    else:
                        best_move = possible_moves[int(math.floor(random() * len(move_scores)))]
                moves_made_black.append(gameboardBlack.to_matrix(best_move))
                gameboardBlack.make_move(best_move)
                move_for_red = ((7 - best_move[0][0], 7 - best_move[0][1]), (7 - best_move[1][0], 7 - best_move[1][1]))
                gameboardRed.make_move(move_for_red)

            # Print out for debugging
            # if should_write_output_to_file:
            #     if games_played % 10 == 0:
            #         f.write("RED SIDE BOARD: \n")
            #         f.write(gameboardRed.print_board(should_return=True))
            # else:
            #     print("RED SIDE BOARD")
            #     gameboardRed.print_board()

            # Evaluate the board,
            # Winning outright should be worth a lot
            # Winning should be scaled by the number of pieces a side won by
            # The effect should not be linear, use diff^1.5
            # If the sides tied, still adjust the weights slightly
            num_red = len([x for x in gameboardRed.pos if x])
            num_black = len([x for x in gameboardRed.neg if x])
            if num_red > num_black:
                red_score = math.pow((num_red - num_black), 1.5)/22.7
            elif num_black > num_red:
                red_score = -math.pow((num_black - num_red), 1.5)/22.7
            else: # TIED, slightly change weights
                # random() generates numbers on (0,1)
                # Want to get numbers in (-0.025, 0.025)
                red_score = random()*0.05 - 0.025
            # # Reward for winning a game entirely (NOT IN RIGHT NOW)
            # if num_red == 0:
            #     red_score -= 1
            # elif num_black == 0:
            #     red_score += 1
            black_score = -red_score

            # Tensorflow (like numpy) will broadcast the single value here (the score from the game) to the entire array when doing
            # The multiplication, so it is fine to leave it as a single value rather than a list of equivalent losses with the length of
            # the number of moves
            train_step_red.run(feed_dict={x_red: moves_made_red, y__red: np.array([red_score / len(moves_made_red)])})
            train_step_black.run(feed_dict={x_black: moves_made_black, y__black: np.array([black_score / len(moves_made_black)])})

            games_played += 1
            if test_output:
                test_out += "Time to play and train on game " + str(games_played) + ": " + str(datetime.now() - start_time) + "\n"
            if should_write_output_to_file:
                f.write("Time to play and train on game " + str(games_played) + ": " + str(datetime.now() - start_time) + "\n\n")
            else:
                print("Time to play and train on game " + str(games_played) + ": " + str(datetime.now() - start_time))

            if games_played % 10000 == 0:
                saver.save(sesh, nn_saver_dir + 'aggressive_new_loss_checkpoint_' + str(games_played + max_checkpoint_val) + '.ckpt')
            if should_write_output_to_file:
                f.close()
        sesh.close()
    with open("print_out.txt", "w") as f:
        f.write(test_out)
