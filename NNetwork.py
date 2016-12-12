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
class ConvLayer(object):

    def __init__(self, input, depth, in_channel, out_channel, window, strides, name_suffix):
        """
        :param input: Tensorflow variable that is the input to this layer
        :param depth: Depth to look at for each convolution
        :param in_channel: Number of channels going into the layer
        :param out_channel: Number of channels out
        :param window: A length-2 list of the window to look at
        :param strides: A length-2 list of the stride size
        :param name_suffix: Suffix to append to the variable names
        """
        self.input = input
        self.W_conv = weight_variable([depth, window[0], window[1], in_channel, out_channel], name="W" + name_suffix)
        self.b_conv = bias_variable([out_channel])
        self.h_conv = tf.nn.relu(conv3d(self.input, self.W_conv, strides=[1, in_channel, 1, 1, 1]) + self.b_conv)
        self.h_pool = max_pool3d_2x2(self.h_conv, ksize=[1, 1, window[0], window[1], 1], strides=[1, 1, strides[0], strides[1], 1])


class FCLayer(object):

    def __init__(self, input, weight_dimensions, name_suffix):
        self.W_fc = weight_variable(shape=weight_dimensions, name="W" + name_suffix)
        self.b_fc = bias_variable(shape=[weight_dimensions[1]])
        self.output = tf.nn.relu(tf.matmul(input, self.W_fc) + self.b_fc)


class NNetwork(object):

    def __init__(self, color):
        self.x = tf.placeholder(tf.float32, shape=[None, 8, 8, 4])
        self.y = tf.placeholder(tf.float32, shape=[1])
        self.x_image = tf.reshape(self.x, [-1, 4, 8, 8, 1])

        self.layer1 = ConvLayer(self.x_image, 4, 1, 32, [2, 2], [2, 2], "_conv1_" + color)
        self.layer2 = ConvLayer(self.layer1.h_pool, 1, 32, 64, [2, 2], [2, 2], "_conv2_" + color)
        self.layer2flattened = tf.reshape(self.layer2.h_pool, [-1, 256])
        self.layer3 = FCLayer(self.layer2flattened, [256, 512], "_fc1_" + color)
        self.layer4 = FCLayer(self.layer3.output, [512, 1], "_fc2_" + color)

        self.y_hat = self.layer4.output
        self.loss = self.y_hat * self.y
        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

# --------------- Running the network ---------------

# To evaluate a gameboard, run this with a running session:
# print(sess.run(y, feed_dict={x: INPUTS}))

# To train on a gameboard, run this:
# train_step.run(feed_dict={x: INPUTS, y_: LABELS})
# Where x: is the input game boards and y is the labels for each


if __name__ == "__main__":
    nn_saver_dir = './nn_checkpoints/'

    should_restore = False
    should_write_output_to_file = False

    with tf.Session() as sess:

        redNetwork = NNetwork("red")
        blackNetwork = NNetwork("black")
        init_op = tf.initialize_all_variables()
        saver = tf.train.Saver()

        sess.run(init_op)
        if should_restore:
            saver.restore(sess, './nn_checkpoints/divergence_90_60_random_checkpoint_1000.ckpt')
            print("Model restored")


        if should_write_output_to_file:
            sys.stdout = open('print_out.txt', 'w')

        games_played = 0
        while(True):
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
                    move_scores[i] = sess.run(redNetwork.y_hat, feed_dict={redNetwork.x: matrix_board})
                # With the best move found, move on both red and black boards
                rand_choice = random()
                if (rand_choice < .9):
                    best_move = possible_moves[np.argmax(move_scores)]
                elif (rand_choice < .95):
                    best_move = possible_moves[move_scores.argsort()[1]]
                elif (rand_choice < .985):
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
                move_scores = np.zeros(len(possible_moves))
                for i in range(len(possible_moves)):
                    matrix_board = gameboardBlack.to_matrix(possible_moves[i])
                    before_pass = datetime.now()
                    move_scores[i] = sess.run(blackNetwork.y_hat, feed_dict={blackNetwork.x: matrix_board})
                    if should_write_output_to_file:
                        if i == 0:
                            print(datetime.now() - before_pass)
                # With the best move found, move on both red and black boards
                rand_choice = random()
                if (rand_choice < .6):
                    best_move = possible_moves[np.argmax(move_scores)]
                elif (rand_choice < .75):
                    best_move = possible_moves[move_scores.argsort()[1]]
                elif (rand_choice < .9):
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
                # Want to get numbers in (-0.125, 0.125)
                red_score = random()*0.25 - .125
            # Reward for winning a game entirely
            if num_red == 0:
                red_score -= 5
            elif num_black == 0:
                red_score +=5
            black_score = -red_score

            for move in moves_made_red:
                redNetwork.optimizer.run(feed_dict={redNetwork.x: move, redNetwork.y: np.array([red_score])})
            for move in moves_made_black:
                blackNetwork.optimizer.run(feed_dict={blackNetwork.x: move, blackNetwork.y: np.array([black_score])})
            print("Time for one game: " + str(datetime.now() - start_time))

            if games_played % 200 == 0:
                saver.save(sess, nn_saver_dir + 'divergence_90_60_random_checkpoint_' + str(games_played + 1000) + '.ckpt')
            games_played += 1

