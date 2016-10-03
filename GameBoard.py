"""
- 8x8 board with a maximum of 16 pieces
- Be able to take as input a move by one of the players and update the board state
- Pieces should be captured and removed when appropriate
- Capability to return some statement of winning/losing either by all pieces being eliminated or
whoever has the most pieces
- Should store every move for a game from each player
"""
import numpy as np

class GameBoard:
    #TODO Fix indexing

    def __init__(self):
        self.grid = [[0 for i in range(8)] for j in range(8)]
        self.neg = [(0, x) for x in range(8)]
        self.pos = [(7, x) for x in range(8)]
        for index in range(8):
            self.grid[0][index] = -index - 1 # adding one so that piece values don't drop to zero
            self.grid[7][index] = index + 1 # subtracting one so that piece values don't drop to zero

    def update_board(self, board, pos, neg, update_move):
        piece_val = board[update_move[0][0]][update_move[0][1]]
        board[update_move[0][0]][update_move[0][1]] = 0
        board[update_move[1][0]][update_move[1][1]] = piece_val
        if piece_val > 0:
            pos[piece_val - 1] = update_move[1]
            for capped in self.find_captures(board, update_move[1], 1):
                # The value returned by find captures will be the negative index within the enemy teams list
                cap_location = neg[-1*capped - 1]
                board[cap_location[0]][cap_location[1]] = 0
                neg[-1*capped - 1] = None
        else:
            neg[-piece_val - 1] = update_move[1]
            for capped in self.find_captures(board, update_move[1], -1):
                # The value returned by find captures will be the negative index within the enemy teams list
                cap_location = pos[capped - 1]
                board[cap_location[0]][cap_location[1]] = 0
                pos[capped - 1] = None
        return board, pos, neg

    def find_captures(self, grid, piece_location, team):
        return self.check_capture_direction_NS(grid, piece_location, team, range(piece_location[0]-1, -1, -1)) + \
               self.check_capture_direction_NS(grid, piece_location, team, range(piece_location[0]+1, 8)) + \
               self.check_capture_direction_EW(grid, piece_location, team, range(piece_location[1]-1, -1, -1)) + \
               self.check_capture_direction_EW(grid, piece_location, team, range(piece_location[1]+1, 8))

    def check_capture_direction_NS(self, grid, piece_location, team, search_spaces):
        potential_capture = []
        for i in search_spaces:
            # piece type is the index of the piece, with positivity indicating same team, and negativity the opposite
            piece_type = team*grid[i][piece_location[1]]
            if piece_type < 0:
                potential_capture.append(grid[i][piece_location[1]])
            elif piece_type == 0:
                break
            else:
                return potential_capture
        return []

    def check_capture_direction_EW(self, grid, piece_location, team, search_spaces):
        potential_capture = []
        for i in search_spaces:
            # piece type is the index of the piece, with positivity indicating same team, and negativity the opposite
            piece_type = team*grid[piece_location[0]][i]
            if piece_type < 0:
                potential_capture.append(grid[piece_location[0]][i])
            elif piece_type == 0:
                break
            else:
                return potential_capture
        return []

    def generate_moves(self, side):
        possible_moves = []
        for piece in side:
            if piece is None:
                continue
            # For moving forward
            for i in range(piece[0]-1, -1, -1):
                if self.grid[i][piece[1]] == 0:
                    possible_moves.append((piece, (i, piece[1])))
                else:
                    break
            # For moving backwards
            for i in range(piece[0]+1, 8):
                if self.grid[i][piece[1]] == 0:
                    possible_moves.append((piece, (i, piece[1])))
                else:
                    break
            # For moving left
            for i in range(piece[1]-1, -1, -1):
                if self.grid[piece[0]][i] == 0:
                    possible_moves.append((piece, (piece[0], i)))
                else:
                    break
            # For moving right
            for i in range(piece[1]+1, 8):
                if self.grid[piece[0]][i] == 0:
                    possible_moves.append((piece, (piece[0], i)))
                else:
                    break
        return possible_moves

    def deepcopy_board(self):
        """Used by to_matrix to allow for updating board without actually updating the board"""
        # TODO MAYBE check if this function or copy.deepcopy() is faster
        new_pos = [x for x in self.pos]
        new_neg = [x for x in self.neg]
        new_grid = [[i for i in j] for j in self.grid]
        return new_grid, new_pos, new_neg

    def to_matrix(self, update_move):
        new_grid, new_pos, new_neg = self.deepcopy_board()
        grid_after_move, updated_pos, updated_neg = self.update_board(new_grid, new_pos, new_neg, update_move)
        nn_input = np.zeros(shape=(8,8,4))
        # Numpy indexing expects a list for each dimension. We pass in rows indices by going over the tuples in pos and column
        # indices by doing the same. We know the third dimension we expect.
        nn_input[[x[0] for x in self.pos], [x[1] for x in self.pos], 0] = 1.0
        nn_input[[x[0] for x in self.neg], [x[1] for x in self.neg], 1] = 1.0
        nn_input[[x[0] for x in updated_pos], [x[1] for x in updated_pos], 2] = 1.0
        nn_input[[x[0] for x in updated_neg], [x[1] for x in updated_neg], 3] = 1.0
        return nn_input


    def print_board(self):
        board_str = ""
        for i in range(8):
            line_str = "|"
            for j in range(8):
                if self.grid[i][j] < 0:
                    line_str += " - |"
                elif self.grid[i][j] > 0:
                    line_str += " + |"
                else:
                    line_str += "   |"
            board_str += line_str + "\n"
        print(board_str)





