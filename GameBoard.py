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
            self.grid[0][index] = -1
            self.grid[7][index] = 1

    def print_board(self):
        board_str = ""
        for i in range(8):
            line_str = "|"
            for j in range(8):
                if self.grid[i][j] == 1:
                    line_str += " + |"
                elif self.grid[i][j] == -1:
                    line_str += " - |"
                else:
                    line_str += "   |"
            board_str += line_str + "\n"
        print(board_str)





