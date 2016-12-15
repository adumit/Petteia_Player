"""
- 8x8 board with a maximum of 16 pieces
- Be able to take as input a move by one of the players and update the board state
- Pieces should be captured and removed when appropriate
- Capability to return some statement of winning/losing either by all pieces being eliminated or
whoever has the most pieces
- Should store every move for a game from each player
"""
import numpy as np
import copy

class GameBoard:

    def __init__(self):
        self.grid = [[0 for i in range(8)] for j in range(8)]
        self.neg = {(0, x) for x in range(8)}
        self.pos = {(7, x) for x in range(8)}
        for index in range(8):
            self.grid[0][index] = -index - 1  # adding one so that piece values don't drop to zero
            self.grid[7][index] = index + 1  # subtracting one so that piece values don't drop to zero

    def make_move(self, update_move):
        new_board, new_pos, new_neg = self.update_board(self.grid, self.pos, self.neg, update_move)
        self.grid = new_board
        self.pos = new_pos
        self.neg = new_neg

    def update_board(self, game_grid, pos, neg, update_move):
        # TODO this should NOT modify the board that is passed in, but it currently does
        former_loc = (update_move[0][0], update_move[0][1])
        new_loc = (update_move[1][0], update_move[1][1])

        piece_val = game_grid[former_loc[0]][former_loc[1]]
        game_grid[former_loc[0]][former_loc[1]] = 0
        game_grid[new_loc[0]][new_loc[1]] = piece_val
        if piece_val > 0:
            pos.remove(former_loc)
            pos.add(new_loc)
            for capped in self.find_captures(game_grid, new_loc, 1):
                # The value returned by find captures will be the negative index within the enemy teams list
                game_grid[capped[0]][capped[1]] = 0
                neg.remove(capped)
        else:
            neg.remove(former_loc)
            neg.add(new_loc)
            for capped in self.find_captures(game_grid, update_move[1], -1):
                # The value returned by find captures will be the negative index within the enemy teams list
                game_grid[capped[0]][capped[1]] = 0
                pos.remove(capped)
        return game_grid, pos, neg

    def generate_capture_moves(self, move_list):
        """Find all moves that would result in a capture. Assume the team is 'pos'"""
        # possible_moves = self.generate_moves()
        move_locations = [m[1] for m in move_list]
        capture_outcomes = [m for m in move_locations if self.find_captures(self.grid, m, 1)]
        return [m for m in move_list if m[1] in capture_outcomes]


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
                potential_capture.append((i, piece_location[1]))
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
                potential_capture.append((piece_location[0], i))
            elif piece_type == 0:
                break
            else:
                return potential_capture
        return []

    def generate_moves(self):
        """Generate_moves always generates them from the positive side.
        The network will have a board for each team"""
        possible_moves = []
        for piece in self.pos:
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
        new_pos = {x for x in self.pos}
        new_neg = {x for x in self.neg}
        new_grid = [[i for i in j] for j in self.grid]
        return new_grid, new_pos, new_neg

    def move_and_flip_board(self, update_move):

        def reverse_list(l):
            return [l[i] for i in range(len(l) - 1, -1, -1)]

        old_grid = copy.deepcopy(self.grid)
        old_grid = reverse_list([reverse_list(l) for l in self.grid])
        for i in range(len(old_grid)):
            for j in range(len(old_grid[i])):
                old_grid[i][j] = -1 * old_grid[i][j]

        old_pos = copy.deepcopy(self.pos)
        new_neg = set([])
        for pos in old_pos:
            new_neg.add((7 - pos[0], 7 - pos[1]))

        old_neg = copy.deepcopy(self.neg)
        new_pos = set([])
        for neg in old_neg:
            new_pos.add((7 - neg[0], 7 - neg[1]))

        new_gb = GameBoard()
        new_gb.grid = old_grid
        new_gb.pos = new_pos
        new_gb.neg = new_neg

        flipped_update = ((7 - update_move[0][0], 7 - update_move[0][1]), (7 - update_move[1][0], 7 - update_move[1][1]))
        new_gb.make_move(flipped_update)

        return new_gb

    def to_matrix(self, update_move):
        #new_grid, new_pos, new_neg = self.deepcopy_board()
        new_grid = copy.deepcopy(self.grid)
        new_pos = copy.deepcopy(self.pos)
        new_neg = copy.deepcopy(self.neg)
        grid_after_move, updated_pos, updated_neg = self.update_board(new_grid, new_pos, new_neg, update_move)
        board_matrix = np.zeros(shape=(8, 8, 4))
        # Numpy indexing expects a list for each dimension. We pass in rows indices by going over the tuples in
        # pos and column indices by doing the same. We know the third dimension we expect.
        board_matrix[[x[0] for x in self.pos], [x[1] for x in self.pos], 0] = 1.0
        board_matrix[[x[0] for x in self.neg], [x[1] for x in self.neg], 1] = 1.0
        board_matrix[[x[0] for x in updated_pos], [x[1] for x in updated_pos], 2] = 1.0
        board_matrix[[x[0] for x in updated_neg], [x[1] for x in updated_neg], 3] = 1.0
        return board_matrix

    def print_board(self, should_return=False):
        board_str = "    0   1   2   3   4   5   6   7   \n"
        for i in range(8):
            line_str = str(i) + " |"
            for j in range(8):
                if self.grid[i][j] < 0:
                    line_str += " - |"
                elif self.grid[i][j] > 0:
                    line_str += " + |"
                else:
                    line_str += "   |"
            board_str += line_str + "\n"
        if should_return:
            return board_str
        print(board_str)





