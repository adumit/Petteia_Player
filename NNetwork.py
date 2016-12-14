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
class ConvLayer(object):

    def __init__(self, layer_input, in_channel, out_channel, weight_dims, conv_strides,
                 pool_ksize, pool_strides, name_suffix):
        """
        :param input: Tensorflow variable that is the input to this layer
        :param depth: Depth to look at for each convolution
        :param in_channel: Number of channels going into the layer
        :param out_channel: Number of channels out
        :param window: A length-2 list of the window to look at
        :param strides: A length-2 list of the stride size
        :param name_suffix: Suffix to append to the variable names
        """
        self.input = layer_input
        self.W_conv = weight_variable(weight_dims + [in_channel, out_channel],
                                      name="W" + name_suffix)
        self.b_conv = bias_variable([out_channel])
        self.h_conv = tf.nn.relu(conv3d(self.input, self.W_conv, strides=conv_strides) + self.b_conv)
        self.h_pool = max_pool3d_2x2(self.h_conv, ksize=pool_ksize, strides=pool_strides)


class FCLayer(object):

    def __init__(self, layer_input, weight_dimensions, name_suffix):
        self.input = layer_input
        self.W_fc = weight_variable(shape=weight_dimensions, name="W" + name_suffix)
        self.b_fc = bias_variable(shape=[weight_dimensions[1]])
        self.output = tf.nn.tanh(tf.matmul(self.input, self.W_fc) + self.b_fc)


class NNetwork(object):

    def __init__(self, color):
        self.x = tf.placeholder(tf.float32, shape=[None, 8, 8, 4])
        self.y = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_image = tf.reshape(self.x, [-1, 4, 8, 8, 1])

        self.layer1 = ConvLayer(layer_input=self.x_image, in_channel=1, out_channel=32,
                                weight_dims=[4, 4, 4], conv_strides=[1, 4, 2, 2, 1],
                                pool_ksize=[1, 1, 2, 2, 1], pool_strides=[1, 1, 2, 2, 1],
                                name_suffix="conv1_" + color)
        self.layer2 = ConvLayer(layer_input=self.layer1.h_pool, in_channel=32, out_channel=64,
                                weight_dims=[1, 4, 4], conv_strides=[1, 1, 2, 2, 1],
                                pool_ksize=[1, 1, 2, 2, 1], pool_strides=[1, 1, 2, 2, 1],
                                name_suffix="conv2_" + color)
        self.layer2flattened = tf.reshape(self.layer2.h_pool, [-1, 64])
        self.layer3 = FCLayer(self.layer2flattened, [64, 128], "_fc1_" + color)
        self.layer4 = FCLayer(self.layer3.output, [128, 1], "_fc2_" + color)

        self.y_hat = self.layer4.output
        self.loss = tf.square(self.y_hat - self.y)
        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        print(color + " network is built.")

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
    test_output = False

    # Constants for game playing
    aggressive = True

    test_out = ""
    max_checkpoint_val = 0

    with tf.Session() as sess:

        redNetwork = NNetwork("red")
        blackNetwork = NNetwork("black")

        init_op = tf.initialize_all_variables()
        saver = tf.train.Saver()

        sess.run(init_op)
        if should_restore:
            checkpoint_vals = []
            for root, dirs, files in os.walk(nn_saver_dir):
                for f in files:
                    checkpoint_vals.append(re.search(r'\d+', f))
            checkpoint_vals = [int(x.group()) for x in checkpoint_vals if x]
            max_checkpoint_val = max(checkpoint_vals)
            saver.restore(sess, tf.train.latest_checkpoint(nn_saver_dir))
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
            while len(gameboardRed.pos) > 0 and len(moves_made_red) < 100:
                # Make a move for red
                possible_moves = gameboardRed.generate_moves()
                if len(possible_moves) == 0:
                    break
                before_pass = datetime.now()
                move_scores = sess.run(redNetwork.y_hat, feed_dict={redNetwork.x: [gameboardRed.to_matrix(m) for m in possible_moves]})
                if should_write_output_to_file:
                    if iter == 0:
                        f.write("Time to pass all moves through network: " + str(datetime.now() - before_pass) + "\n")
                # With the best move found, move on both red and black boards
                rand_choice = random()
                if len(moves_made_red) == 10:
                    if should_write_output_to_file:
                        f.write("Max score for iteration 10 and game " + str(games_played) + "was: " + str(np.max(move_scores)) + "\n")
                    else:
                        print(np.max(move_scores))
                        print(np.min(move_scores))
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
                        move_scores = sess.run(blackNetwork.y_hat, feed_dict={blackNetwork.x: [gameboardBlack.to_matrix(m) for m in capture_moves]})
                        best_move = capture_moves[np.argmax(move_scores)]

                if best_move is None:
                    move_scores = sess.run(blackNetwork.y_hat, feed_dict={blackNetwork.x: [gameboardBlack.to_matrix(m) for m in possible_moves]})
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
                # red_score = random()*0.05 - 0.025
                red_score = 0.0
            # # Reward for winning a game entirely (NOT IN RIGHT NOW)
            # if num_red == 0:
            #     red_score -= 1
            # elif num_black == 0:
            #     red_score += 1
            black_score = -red_score
            discount_vec_red = np.array([math.pow(.99, p) for p in range(len(moves_made_red), 0, -1)])
            discount_vec_black = np.array([math.pow(.99, p) for p in range(len(moves_made_black), 0, -1)])

            red_score_vec = np.reshape(red_score * discount_vec_red, newshape=[-1, 1])
            black_score_vec = np.reshape(black_score * discount_vec_black, newshape=[-1, 1])

            redNetwork.optimizer.run(feed_dict={redNetwork.x: moves_made_red,
                                                redNetwork.y: red_score_vec})
            blackNetwork.optimizer.run(feed_dict={blackNetwork.x: moves_made_black,
                                                  blackNetwork.y: black_score_vec})

            games_played += 1
            if test_output:
                test_out += "Time to play and train on game " + str(games_played) + ": " + str(datetime.now() - start_time) + "\n"
            if should_write_output_to_file:
                f.write("Time to play and train on game " + str(games_played) + ": " + str(datetime.now() - start_time) + "\n\n")
            else:
                print("Time to play and train on game " + str(games_played) + ": " + str(datetime.now() - start_time))

            if games_played % 200 == 0:
                redNetwork.saver.save(sess, nn_saver_dir + 'aggressive_new_loss_checkpoint_' + str(games_played + max_checkpoint_val) + '.ckpt')
            if should_write_output_to_file:
                f.close()
        sess.close()
    with open("print_out.txt", "w") as f:
        f.write(test_out)
