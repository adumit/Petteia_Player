import NNetwork
import sys
import GameBoard as gb
import tensorflow as tf
import numpy as np
import re

if __name__ == "__main__":
    nn_saver_dir = './nn_checkpoints/'
    red_saver_dir = "./red_network_checkpoints/"
    non_numeric = re.compile(r'[^\d]+')

    with tf.Session() as sess:
        redNetwork = NNetwork.NNetwork("red", look_ahead=True)
        blackNetwork = NNetwork.NNetwork("black")
        init_op = tf.initialize_all_variables()
        saver = tf.train.Saver()
        sess.run(init_op)
        # redNetwork.saver.restore(sess, tf.train.latest_checkpoint(red_saver_dir))

        while True:
            user_choice = input("What would you like to do? \n"
                                "'pr': Play a game against the regular AI.\n"
                                "'pr+': Play a game against the regular AI with mini-max look ahead.\n"
                                "'pa': Play a game against the aggressive AI.\n"
                                "'w': Watch two AIs play each other.")
            if 'pr' in user_choice:
                moves_made_red = []
                moves_made_black = []
                gameboardRed = gb.GameBoard()
                gameboardBlack = gb.GameBoard()
                while len([x for x in gameboardRed.pos if x]) > 0 and len([x for x in gameboardRed.neg if x]) > 0:
                    possible_moves = gameboardRed.generate_moves()
                    if len(possible_moves) == 0:
                        break
                    if "+" in user_choice:
                        best_move = redNetwork.minimax_look_ahead_move(sess, gameboardRed, possible_moves, top_n=-1)
                    else:
                        move_scores = sess.run(redNetwork.y_hat, feed_dict={
                            redNetwork.x: [gameboardRed.to_matrix(m) for m in possible_moves]})
                        # With the best move found, move on both red and black boards
                        best_move = possible_moves[np.argmax(move_scores)]
                    moves_made_red.append(gameboardRed.to_matrix(best_move))
                    gameboardRed.make_move(best_move)
                    move_for_black = ((7 - best_move[0][0], 7 - best_move[0][1]),(7 - best_move[1][0], 7 - best_move[1][1]))
                    gameboardBlack.make_move(move_for_black)

                    print("AI Played piece " + str(move_for_black[0]) + " to " + str(move_for_black[1]))
                    gameboardBlack.print_board()
                    # Ask the user for an input now and make a move
                    possible_moves = gameboardBlack.generate_moves()
                    user_move = ((-1, -1), (-1, -1))
                    while user_move not in possible_moves:
                        user_from_row = int(input("Row of piece you want to move?"))
                        user_from_col = int(input("Column of piece you want to move?"))
                        user_to_row = int(input("Row to move to?"))
                        user_to_col = int(input("Column to move to?"))
                        user_move = ((user_from_row, user_from_col), (user_to_row, user_to_col))

                    gameboardBlack.make_move(user_move)
                    move_for_red = ((7 - user_move[0][0], 7 - user_move[0][1]),
                                    (7 - user_move[1][0], 7 - user_move[1][1]))
                    gameboardRed.make_move(move_for_red)

                    gameboardBlack.print_board()

            if user_choice == 'pa':
                moves_made_red = []
                moves_made_black = []
                gameboardRed = gb.GameBoard()
                gameboardBlack = gb.GameBoard()
                while len([x for x in gameboardRed.pos if x]) > 0 and len([x for x in gameboardRed.neg if x]) > 0:
                    possible_moves = gameboardRed.generate_moves()
                    if len(possible_moves) == 0:
                        break
                    move_scores = sess.run(blackNetwork.y_hat, feed_dict={
                        blackNetwork.x: [gameboardRed.to_matrix(m) for m in possible_moves]})
                    # With the best move found, move on both red and black boards
                    best_move = possible_moves[np.argmax(move_scores)]
                    moves_made_red.append(gameboardRed.to_matrix(best_move))
                    gameboardRed.make_move(best_move)
                    move_for_black = ((7 - best_move[0][0], 7 - best_move[0][1]),
                                      (7 - best_move[1][0], 7 - best_move[1][1]))
                    gameboardBlack.make_move(move_for_black)

                    print("AI Played piece " + str(move_for_black[0]) + " to " + str(move_for_black[1]))
                    gameboardBlack.print_board()
                    # Ask the user for an input now and make a move
                    possible_moves = gameboardBlack.generate_moves()
                    user_move = ((-1, -1), (-1, -1))
                    while user_move not in possible_moves:
                        user_from_row = int(input("Row of piece you want to move?"))
                        user_from_col = int(input("Column of piece you want to move?"))
                        user_to_row = int(input("Row to move to?"))
                        user_to_col = int(input("Column to move to?"))
                        user_move = ((user_from_row, user_from_col), (user_to_row, user_to_col))

                    gameboardBlack.make_move(user_move)
                    move_for_red = ((7 - user_move[0][0], 7 - user_move[0][1]),
                                    (7 - user_move[1][0], 7 - user_move[1][1]))
                    gameboardRed.make_move(move_for_red)

                    gameboardBlack.print_board()

            if user_choice == 'w':
                moves_made_red = []
                moves_made_black = []
                gameboardRed = gb.GameBoard()
                gameboardBlack = gb.GameBoard()
                while len([x for x in gameboardRed.pos if x]) > 0 and len(
                        [x for x in gameboardRed.neg if x]) > 0:
                    user_input = input("Press enter to see the next move.")
                    possible_moves = gameboardRed.generate_moves()
                    if len(possible_moves) == 0:
                        break
                    move_scores = sess.run(redNetwork.y_hat, feed_dict={
                        redNetwork.x: [gameboardRed.to_matrix(m) for m in
                                  possible_moves]})
                    # With the best move found, move on both red and black boards
                    best_move = possible_moves[np.argmax(move_scores)]
                    moves_made_red.append(gameboardRed.to_matrix(best_move))
                    gameboardRed.make_move(best_move)
                    move_for_black = (
                    (7 - best_move[0][0], 7 - best_move[0][1]),
                    (7 - best_move[1][0], 7 - best_move[1][1]))
                    gameboardBlack.make_move(move_for_black)

                    print("Red played piece " + str(
                        best_move[0]) + " to " + str(best_move[1]))
                    gameboardRed.print_board()

                    user_input = input("Press enter to see the next move.")
                    # Move for black
                    possible_moves = gameboardBlack.generate_moves()
                    if len(possible_moves) == 0:
                        break
                    move_scores = sess.run(blackNetwork.y_hat, feed_dict={
                        blackNetwork.x: [gameboardBlack.to_matrix(m) for m in
                                  possible_moves]})
                    # With the best move found, move on both red and black boards
                    best_move = possible_moves[np.argmax(move_scores)]
                    moves_made_black.append(gameboardBlack.to_matrix(best_move))
                    gameboardBlack.make_move(best_move)
                    move_for_red = (
                        (7 - best_move[0][0], 7 - best_move[0][1]),
                        (7 - best_move[1][0], 7 - best_move[1][1]))
                    gameboardRed.make_move(move_for_red)

                    print("Black side played piece " + str(
                        move_for_red[0]) + " to " + str(move_for_red[1]))
                    gameboardRed.print_board()

