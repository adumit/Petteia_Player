import GameBoard as gb
import numpy as np
import math
import argparse
from random import random
from datetime import datetime
import sys
import re
import os
import tensorflow as tf
import warnings
import pickle
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

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

# ------------------- Network Definitions -------------------
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
        # TODO: Experiment with max and average pooling
        self.h_pool = max_pool3d_2x2(self.h_conv, ksize=pool_ksize, strides=pool_strides)


class DeconvLayer(object):
    def __init__(self, layer_input, weight_dim, output_shape, stride_shape, name_suffix):
        # value = tf.convert_to_tensor(layer_input, name="value")
        # filter = tf.convert_to_tensor(weight_dim, name="filter")
        # print(value.get_shape())
        # print(filter.get_shape())
        self.W_deconv = weight_variable(shape=weight_dim, name="W_" + name_suffix)
        self.deconv = tf.nn.conv3d_transpose(layer_input, filter=self.W_deconv, output_shape=output_shape, strides=stride_shape, padding="SAME")
        self.b_deconv = bias_variable(shape=[weight_dim[3]])
        self.output = self.deconv + self.b_deconv


class FCLayer(object):

    def __init__(self, layer_input, weight_dimensions, name_suffix):
        self.input = layer_input
        self.W_fc = weight_variable(shape=weight_dimensions, name="W" + name_suffix)
        self.b_fc = bias_variable(shape=[weight_dimensions[1]])
        self.activation = tf.matmul(self.input, self.W_fc) + self.b_fc


class NNetwork(object):

    def __init__(self, color, aggressive=False, look_ahead=False):
        self.aggressive = aggressive
        self.look_ahead = look_ahead

        with tf.variable_scope(color + "_scope"):
            self.temp_batch_size = tf.placeholder(dtype=tf.int32, shape=[]) # Included to be alike Deconv net
            self.x = tf.placeholder(tf.float32, shape=[None, 8, 8, 4])
            self.y = tf.placeholder(tf.float32, shape=[None, 1])
            self.x_image = tf.reshape(self.x, [-1, 4, 8, 8, 1])

            self.layer1 = ConvLayer(layer_input=self.x_image, in_channel=1, out_channel=32,
                                    weight_dims=[4, 4, 4], conv_strides=[1, 4, 2, 2, 1],
                                    pool_ksize=[1, 1, 2, 2, 1], pool_strides=[1, 1, 2, 2, 1],
                                    name_suffix="conv1_" + color)
            self.layer2 = ConvLayer(layer_input=self.layer1.h_pool, in_channel=32, out_channel=64,
                                    weight_dims=[1, 4, 4], conv_strides=[1, 1, 1, 1, 1],
                                    pool_ksize=[1, 1, 1, 1, 1], pool_strides=[1, 1, 1, 1, 1],
                                    name_suffix="conv2_" + color)

            self.layer2flattened = tf.reshape(self.layer2.h_pool, [-1, 256])
            self.layer3 = FCLayer(self.layer2flattened, [256, 512], "_fc1_" + color)
            self.layer4 = FCLayer(tf.nn.relu(self.layer3.activation), [512, 1], "_fc2_" + color)

            self.y_hat = self.layer4.activation
            self.loss = tf.reduce_mean(tf.square(self.y_hat - self.y))
            self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=color + "_scope"))

        print(color + " network is built.")

    def minimax_look_ahead_move(self, sess, gameBoard, possible_moves, top_n=-1):
        """
        Find the move that maximizes my current expectation of reward accounting for the fact that my opponent is going
        to jointly maximize the next move score and minimize the score I get on the next move as well. This takes the
        form of my opponent choosing a move s.t. their score - score I would choose is maximized. I seek to maximize
        my current move score - that expectation.
        Can be given fewer
        """

        def get_indexed_elements(list, indices):
            """Helper function to grab the items at the given indices from the provided list"""
            return [list[i] for i in indices]

        if top_n >= len(possible_moves):
            top_n = len(possible_moves) - 1
        first_move_scores = sess.run(self.y_hat, feed_dict={
            self.x: [gameBoard.to_matrix(m) for m in possible_moves]})
        if top_n > 0:
            top_indices = np.argpartition(first_move_scores[:,0], top_n)[-top_n:]
            possible_moves = get_indexed_elements(possible_moves, top_indices)
            first_move_scores = get_indexed_elements(first_move_scores, top_indices)
        max_opp_score = []
        for m, score in zip(possible_moves, first_move_scores):
            # 'I' make a move and then flip the board so 'I' can see from my opponents point of view
            flipped_board = gameBoard.move_and_flip_board(m)
            # Generate all my opponent's moves and scores
            possible_opponent_moves = flipped_board.generate_moves()
            if len(possible_moves) == 0:
                opponent_scores = [-1.0]
            else:
                opponent_scores = sess.run(self.y_hat, feed_dict={
                    self.x: [flipped_board.to_matrix(move) for move in possible_opponent_moves]})
            if top_n > 0:
                top_indices = np.argpartition(opponent_scores[:,0], top_n)[-top_n:]
                possible_opponent_moves = get_indexed_elements(possible_opponent_moves, top_indices)
                opponent_scores = get_indexed_elements(opponent_scores, top_indices)
            my_next_scores = []
            # For each (score, move) in my opponent's possibilities, append the max value I would receive
            for opp_m, opp_score in zip(possible_opponent_moves, opponent_scores):
                my_new_board = flipped_board.move_and_flip_board(opp_m)
                my_next_possible_moves = my_new_board.generate_moves()
                if len(my_next_possible_moves) == 0:
                    my_next_move_scores = [-1.0]
                else:
                    my_next_move_scores = sess.run(self.y_hat, feed_dict={
                        self.x: [my_new_board.to_matrix(move) for move in my_next_possible_moves]})
                my_next_scores.append(np.max(my_next_move_scores))
            # Opponent would choose the move that maximizes his current payoff minus the payoff I would get
            max_opp_score.append(np.max([opp_s - max_s for opp_s, max_s in zip(opponent_scores, my_next_scores)]))
        # I want to maximize my score on this move minus the score my opponent would expect to achieve
        return possible_moves[np.argmax([my_score - max_opp_score
                                         for my_score, max_opp_score in zip(first_move_scores, max_opp_score)])]

    def choose_move(self, sess, gameBoard, look_ahead_top_n=3, percent_random=0.05):
        best_move = None

        possible_moves = gameBoard.generate_moves()
        if self.aggressive:
            capture_moves = gameBoard.generate_capture_moves(
                possible_moves)
            if len(capture_moves) > 0:
                move_scores = sess.run(self.y_hat, feed_dict={
                    self.x: [gameBoard.to_matrix(m) for m in capture_moves]})
                return capture_moves[np.argmax(move_scores)]
        if len(possible_moves) == 0:
            return None
        rand_choice = random()
        # Do exploration with random percentage
        if rand_choice > 1 - percent_random:
            return possible_moves[int(math.floor(random() * len(possible_moves)))]

        # If look ahead is enabled, find the best move via look ahead with top_n
        if self.look_ahead:
            return self.minimax_look_ahead_move(sess, gameBoard, possible_moves, look_ahead_top_n)

        # Otherwise,
        move_scores = sess.run(self.y_hat, feed_dict={
            self.x: [gameBoard.to_matrix(m) for m in possible_moves]})
        return possible_moves[np.argmax(move_scores)]


class DeconvNetwork(object):

    def __init__(self, color, aggressive=False, look_ahead=False):
        self.aggressive = aggressive
        self.look_ahead = look_ahead

        with tf.variable_scope(color + "_scope"):
            self.temp_batch_size = tf.placeholder(dtype=tf.int32, shape=[])
            self.x = tf.placeholder(tf.float32, shape=[None, 8, 8, 4])
            self.y = tf.placeholder(tf.float32, shape=[None, 1])
            self.x_image = tf.reshape(self.x, [-1, 4, 8, 8, 1])

            self.layer1 = ConvLayer(layer_input=self.x_image, in_channel=1, out_channel=32,
                                    weight_dims=[4, 4, 4], conv_strides=[1, 4, 2, 2, 1],
                                    pool_ksize=[1, 1, 2, 2, 1], pool_strides=[1, 1, 2, 2, 1],
                                    name_suffix="conv1_" + color)

            self.layer2 = ConvLayer(layer_input=self.layer1.h_conv, in_channel=32, out_channel=64,
                                    weight_dims=[1, 4, 4], conv_strides=[1, 1, 1, 1, 1],
                                    pool_ksize=[1, 1, 1, 1, 1], pool_strides=[1, 1, 1, 1, 1],
                                    name_suffix="conv2_" + color)

            self.deconv1 = DeconvLayer(layer_input=self.layer2.h_conv,
                                       weight_dim=[1, 4, 4, 32, 64],
                                       output_shape=[self.temp_batch_size, 1, 4, 4, 32],
                                       stride_shape=[1, 1, 1, 1, 1],
                                       name_suffix="decov_1")
            self.deconv2 = DeconvLayer(layer_input=tf.nn.relu(self.deconv1.output),
                                       weight_dim=[1, 2, 2, 1, 32],
                                       output_shape=[self.temp_batch_size, 4, 8, 8, 1],
                                       stride_shape=[1, 4, 2, 2, 1],
                                       name_suffix="decov_2")

            self.reconstruct_loss = tf.reduce_mean(tf.square(self.deconv2.output - self.x_image))

            self.layer2flattened = tf.reshape(self.layer2.h_conv, [-1, 1024])
            self.layer3 = FCLayer(self.layer2flattened, [1024, 2048], "_fc1_" + color)
            self.layer4 = FCLayer(tf.nn.relu(self.layer3.activation), [2048, 1], "_fc2_" + color)

            self.y_hat = self.layer4.activation
            self.policy_loss = tf.reduce_mean(tf.square(self.y_hat - self.y))
            self.loss = self.policy_loss + self.reconstruct_loss
            self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=color + "_scope"))

        print(color + " network is built.")

    def minimax_look_ahead_move(self, sess, gameBoard, possible_moves, top_n=-1):
        """
        Find the move that maximizes my current expectation of reward accounting for the fact that my opponent is going
        to jointly maximize the next move score and minimize the score I get on the next move as well. This takes the
        form of my opponent choosing a move s.t. their score - score I would choose is maximized. I seek to maximize
        my current move score - that expectation.
        Can be given fewer
        """

        def get_indexed_elements(list, indices):
            """Helper function to grab the items at the given indices from the provided list"""
            return [list[i] for i in indices]

        if top_n >= len(possible_moves):
            top_n = len(possible_moves) - 1
        first_move_scores = sess.run(self.y_hat, feed_dict={
            self.x: [gameBoard.to_matrix(m) for m in possible_moves]})
        if top_n > 0:
            top_indices = np.argpartition(first_move_scores[:,0], top_n)[-top_n:]
            possible_moves = get_indexed_elements(possible_moves, top_indices)
            first_move_scores = get_indexed_elements(first_move_scores, top_indices)
        max_opp_score = []
        for m, score in zip(possible_moves, first_move_scores):
            # 'I' make a move and then flip the board so 'I' can see from my opponents point of view
            flipped_board = gameBoard.move_and_flip_board(m)
            # Generate all my opponent's moves and scores
            possible_opponent_moves = flipped_board.generate_moves()
            if len(possible_moves) == 0:
                opponent_scores = [-1.0]
            else:
                opponent_scores = sess.run(self.y_hat, feed_dict={
                    self.x: [flipped_board.to_matrix(move) for move in possible_opponent_moves]})
            if top_n > 0:
                top_indices = np.argpartition(opponent_scores[:,0], top_n)[-top_n:]
                possible_opponent_moves = get_indexed_elements(possible_opponent_moves, top_indices)
                opponent_scores = get_indexed_elements(opponent_scores, top_indices)
            my_next_scores = []
            # For each (score, move) in my opponent's possibilities, append the max value I would receive
            for opp_m, opp_score in zip(possible_opponent_moves, opponent_scores):
                my_new_board = flipped_board.move_and_flip_board(opp_m)
                my_next_possible_moves = my_new_board.generate_moves()
                if len(my_next_possible_moves) == 0:
                    my_next_move_scores = [-1.0]
                else:
                    my_next_move_scores = sess.run(self.y_hat, feed_dict={
                        self.x: [my_new_board.to_matrix(move) for move in my_next_possible_moves]})
                my_next_scores.append(np.max(my_next_move_scores))
            # Opponent would choose the move that maximizes his current payoff minus the payoff I would get
            max_opp_score.append(np.max([opp_s - max_s for opp_s, max_s in zip(opponent_scores, my_next_scores)]))
        # I want to maximize my score on this move minus the score my opponent would expect to achieve
        return possible_moves[np.argmax([my_score - max_opp_score
                                         for my_score, max_opp_score in zip(first_move_scores, max_opp_score)])]

    def choose_move(self, sess, gameBoard, look_ahead_top_n=3, percent_random=0.05):
        best_move = None

        possible_moves = gameBoard.generate_moves()
        if self.aggressive:
            capture_moves = gameBoard.generate_capture_moves(
                possible_moves)
            if len(capture_moves) > 0:
                move_scores = sess.run(self.y_hat, feed_dict={
                    self.x: [gameBoard.to_matrix(m) for m in capture_moves]})
                return capture_moves[np.argmax(move_scores)]
        if len(possible_moves) == 0:
            return None
        rand_choice = random()
        # Do exploration with random percentage
        if rand_choice > 1.0 - percent_random:
            return possible_moves[int(math.floor(random() * len(possible_moves)))]

        # If look ahead is enabled, find the best move via look ahead with top_n
        if self.look_ahead:
            return self.minimax_look_ahead_move(sess, gameBoard, possible_moves, look_ahead_top_n)

        # Otherwise,
        move_scores = sess.run(self.y_hat, feed_dict={
            self.x: [gameBoard.to_matrix(m) for m in possible_moves]})
        return possible_moves[np.argmax(move_scores)]



# --------------- Running the network ---------------

# To evaluate a gameboard, run this with a running session:
# print(sess.run(y, feed_dict={x: INPUTS}))

# To train on a gameboard, run this:
# train_step.run(feed_dict={x: INPUTS, y_: LABELS})
# Where x: is the input game boards and y is the labels for each

def setup_training(opt):
    # run_name = input("Name of run: ")
    os.makedirs("Run_Data/" + opt.run_name + "/", exist_ok=True)

    # red_saver_dir = "./checkpoints/" + input("Directory of red network in checkpoints: ") + "/"
    # black_saver_dir = "./checkpoints/" + input("Directory of black network in checkpoints: ") + "/"
    os.makedirs(opt.red_dir, exist_ok=True)
    os.makedirs(opt.black_dir, exist_ok=True)

    # should_restore_red = input("Restore red network?")
    # should_restore_black = input("Restore black network?")
    #
    # red_look_ahead = input("Should red use look ahead?")
    # black_aggressive = input("Should black be aggressive?")

    # if should_restore_red.lower() in ["y", "t", "true", "yes"]:
    #     FLAGS_should_restore_red = True
    # else:
    #     FLAGS_should_restore_red = False
    # if should_restore_black.lower() in ["y", "t", "true", "yes"]:
    #     FLAGS_should_restore_black = True
    # else:
    #     FLAGS_should_restore_black = False

    FLAGS_should_restore_red = bool(opt.restore_red)
    FLAGS_should_restore_black = bool(opt.restore_black)

    FLAGS_aggressive = bool(opt.black_agg)
    FLAGS_look_ahead = bool(opt.red_look_ahead)
    # if black_aggressive.lower() in ["y", "t", "true", "yes"]:
    #     FLAGS_aggressive = True
    # else:
    #     FLAGS_aggressive = False
    # if red_look_ahead.lower() in ["y", "t", "true", "yes"]:
    #     FLAGS_look_ahead = True
    # else:
    #     FLAGS_look_ahead = False

    print("Run settings: ")
    print("Run name: " + str(opt.run_name))
    print("Red is being restored: " + str(FLAGS_should_restore_red))
    print("Black is being restored: " + str(FLAGS_should_restore_black))
    print("Red is using look ahead: " + str(FLAGS_look_ahead))
    print("Black is being aggressive: " + str(FLAGS_aggressive))

    max_checkpoint_val_red = 0
    max_checkpoint_val_black = 0

    with tf.Session() as sess:
        if bool(opt.black_deconv):
            print("Red is deconv")
            redNetwork = DeconvNetwork("red", look_ahead=FLAGS_look_ahead)
        else:
            print("Red is regular")
            redNetwork = NNetwork("red", look_ahead=FLAGS_look_ahead)
        if bool(opt.black_deconv):
            print("Black is deconv")
            blackNetwork = DeconvNetwork("black", aggressive=FLAGS_aggressive)
        else:
            print("Black is regular")
            blackNetwork = NNetwork("black", aggressive=FLAGS_aggressive)

        init_op = tf.initialize_all_variables()

        sess.run(init_op)

        if FLAGS_should_restore_red:
            checkpoint_vals = []
            for root, dirs, files in os.walk(red_saver_dir):
                for f in files:
                    checkpoint_vals.append(re.search(r'\d+', f))
            checkpoint_vals = [int(x.group()) for x in checkpoint_vals if x]
            max_checkpoint_val_red = max(checkpoint_vals)
            redNetwork.saver.restore(sess, tf.train.latest_checkpoint(red_saver_dir))
            print("Red model restored")
        if FLAGS_should_restore_black:
            checkpoint_vals = []
            for root, dirs, files in os.walk(black_saver_dir):
                for f in files:
                    checkpoint_vals.append(re.search(r'\d+', f))
            checkpoint_vals = [int(x.group()) for x in checkpoint_vals if x]
            max_checkpoint_val_black = max(checkpoint_vals)
            blackNetwork.saver.restore(sess, tf.train.latest_checkpoint(black_saver_dir))
            print("Black model restored")

        red_scores = []
        games_played_red = max_checkpoint_val_red
        games_played_black = max_checkpoint_val_black
        while games_played_red < opt.num_games:
            start_time = datetime.now()
            # ----- Play a game of Petteia -----
            # Generate gameboards, one for each side
            moves_made_red = []
            moves_made_black = []
            gameboardRed = gb.GameBoard()
            gameboardBlack = gb.GameBoard()
            red_move_max_min_diffs = []
            while len(gameboardRed.pos) > 0 and len(gameboardBlack.pos) > 0 and len(
                    moves_made_red) < 100:

                # Make a move for red
                best_move = redNetwork.choose_move(sess, gameboardRed, 5,
                                                   percent_random=opt.exploration_param)
                if best_move is None:
                    break
                moves_made_red.append(gameboardRed.to_matrix(best_move))
                gameboardRed.make_move(best_move)
                move_for_black = ((7 - best_move[0][0], 7 - best_move[0][1]),
                                  (7 - best_move[1][0], 7 - best_move[1][1]))
                gameboardBlack.make_move(move_for_black)

                # Make a move for black
                best_move = blackNetwork.choose_move(sess, gameboardBlack,
                                                     percent_random=opt.exploration_param)
                if best_move is None:
                    break
                moves_made_black.append(gameboardBlack.to_matrix(best_move))
                gameboardBlack.make_move(best_move)
                move_for_red = ((7 - best_move[0][0], 7 - best_move[0][1]),
                                (7 - best_move[1][0], 7 - best_move[1][1]))
                gameboardRed.make_move(move_for_red)

            # Print out for debugging
            #   print("RED SIDE BOARD")
            #   gameboardRed.print_board()

            # Evaluate the board,
            # Winning outright should be worth a lot
            # Winning should be scaled by the number of pieces a side won by
            # The effect should not be linear, use diff^1.5
            # If the sides tied, still adjust the weights slightly
            num_red = len([x for x in gameboardRed.pos if x])
            num_black = len([x for x in gameboardRed.neg if x])
            if num_red > num_black:
                red_score = math.pow((num_red - num_black), 1.5) / 22.7
            elif num_black > num_red:
                red_score = -math.pow((num_black - num_red), 1.5) / 22.7
            else:  # TIED, slightly change weights or don't change at all
                # random() generates numbers on (0,1)
                # Want to get numbers in (-0.025, 0.025)
                # red_score = random()*0.05 - 0.025
                red_score = 0.0
            # # Reward for winning a game entirely (Test if this should be in?)
            # if num_red == 0:
            #     red_score -= 1
            # elif num_black == 0:
            #     red_score += 1
            black_score = -red_score
            discount_vec_red = np.array(
                [math.pow(.98, p) for p in range(len(moves_made_red), 0, -1)])
            discount_vec_black = np.array(
                [math.pow(.98, p) for p in range(len(moves_made_black), 0, -1)])

            red_score_vec = np.reshape(red_score * discount_vec_red, newshape=[-1, 1])
            black_score_vec = np.reshape(black_score * discount_vec_black, newshape=[-1, 1])

            redNetwork.optimizer.run(feed_dict={redNetwork.temp_batch_size: len(moves_made_red),
                                                redNetwork.x: moves_made_red,
                                                redNetwork.y: red_score_vec})
            blackNetwork.optimizer.run(
                feed_dict={blackNetwork.temp_batch_size: len(moves_made_black),
                           blackNetwork.x: moves_made_black,
                           blackNetwork.y: black_score_vec})
            red_scores.append(red_score)
            games_played_red += 1
            games_played_black += 1
            print("Game " + str(games_played_red) + " had " + str(len(moves_made_red)) +
                  " moves where red had " + str(num_red) + " pieces left" +
                  " and took " + str(datetime.now() - start_time))

            if games_played_red % 100 == 0:
                redNetwork.saver.save(sess, red_saver_dir + "red_",
                                      global_step=games_played_red)
                blackNetwork.saver.save(sess, black_saver_dir + "black_",
                                        global_step=games_played_black)
                with open("Run_Data/" + run_name + "/" + "_" + str(
                                games_played_red - 100) + "-" + str(games_played_red) + ".pickle",
                          "ab+") as f:
                    pickle.dump(obj=red_scores, file=f)
                red_scores = []
        sess.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a Petteia Playing Network')

    # Run parameters
    parser.add_argument('--run_name', type=str, default='test_run', help='Name of run')
    parser.add_argument('--red_dir', type=str, default='./checkpoints/test_red/',
                        help='Checkpoint directory for red network')
    parser.add_argument('--black_dir', type=str, default='./checkpoints/test_black/',
                        help='Checkpoint directory for black network')
    parser.add_argument('--restore_red', type=int, default=0,
                        help='Should restore the red network from checkpoint directory?')
    parser.add_argument('--restore_black', type=int, default=0,
                        help='Should restore the black network from checkpoint directory?')
    parser.add_argument('--red_look_ahead', type=int, default=0,
                        help='Should the red network use look ahead?')
    parser.add_argument('--black_agg', type=int, default=1,
                        help='Should the black network be aggressive?')
    parser.add_argument('--red_deconv', type=int, default=0,
                        help='Should the red network use deconvolutions?')
    parser.add_argument('--black_deconv', type=int, default=0,
                        help='Should the black network use deconvolutions?')

    # Training params
    parser.add_argument('--exploration_param', type=float, default=0.05,
                        help='Percent of moves that should be random')
    parser.add_argument('--num_games', type=int, default=1000,
                        help='How many games should the networks play?')

    params = parser.parse_args()

    setup_training(params)

