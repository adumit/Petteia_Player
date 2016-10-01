"""
- 8x8 board with a maximum of 16 pieces
- Be able to take as input a move by one of the players and update the board state
- Pieces should be captured and removed when appropriate
- Capability to return some statement of winning/losing either by all pieces being eliminated or
whoever has the most pieces
- Should store every move for a game from each player
"""


class GameBoard:

    def __init__(self):
        self.grid = [[0 for i in range(8)] for j in range(8)]
        self.neg = [(0, x) for x in range(8)]
        self.pos = [(7, x) for x in range(8)]
        for index in range(8):
            self.grid[0][index] = -index - 1 # adding one so that piece values don't drop to zero
            self.grid[7][index] = index + 1 # subtracting one so that piece values don't drop to zero

    def update_board(self, update_move):
        piece_val = self.grid[update_move[0][0]][update_move[0][1]]
        self.grid[update_move[0][0]][update_move[0][1]] = 0
        self.grid[update_move[1][0]][update_move[1][1]] = piece_val
        if piece_val > 0:
            self.pos[piece_val - 1] = update_move[1]
            for capped in self.find_captures(update_move[1], 1):
                # The value returned by find captures will be the negative index within the enemy teams list
                cap_location = self.neg[-1*capped - 1]
                self.grid[cap_location[0]][cap_location[1]] = 0
                self.neg[-1*capped - 1] = None
        else:
            self.neg[-piece_val - 1] = update_move[1]
            for capped in self.find_captures(update_move[1], -1):
                # The value returned by find captures will be the negative index within the enemy teams list
                cap_location = self.pos[capped - 1]
                self.grid[cap_location[0]][cap_location[1]] = 0
                self.pos[capped - 1] = None

    def find_captures(self, piece_location, team):
        return self.check_capture_direction_NS(piece_location, team, range(piece_location[0]-1, -1, -1)) + \
               self.check_capture_direction_NS(piece_location, team, range(piece_location[0]+1, 8)) + \
               self.check_capture_direction_EW(piece_location, team, range(piece_location[1]-1, -1, -1)) + \
               self.check_capture_direction_EW(piece_location, team, range(piece_location[1]+1, 8))

    def check_capture_direction_NS(self, piece_location, team, search_spaces):
        potential_capture = []
        for i in search_spaces:
            # piece type is the index of the piece, with positivity indicating same team, and negativity the opposite
            piece_type = team*self.grid[i][piece_location[1]]
            if piece_type < 0:
                potential_capture.append(self.grid[i][piece_location[1]])
            elif piece_type == 0:
                break
            else:
                return potential_capture
        return []

    def check_capture_direction_EW(self, piece_location, team, search_spaces):
        potential_capture = []
        for i in search_spaces:
            # piece type is the index of the piece, with positivity indicating same team, and negativity the opposite
            piece_type = team*self.grid[piece_location[0]][i]
            if piece_type < 0:
                potential_capture.append(self.grid[piece_location[0]][i])
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





