"""
- Each spot on the board is represented by (1,0), (0,0), or (0,1)
- (1,0) represents that your player is present on that spot
- (0,0) represents that no player is present on a spot
- (0,1) represents that an opponents player is on a spot
- A single move is represented by a 64x4 matrix where the first two columns are the
current state of the board and the next two are the state of the board after a move.
"""

class PossibleMoveGenerator():

    def __init__(self):
        pass

    def generateMoves(self, gameBoard, side):
        possibleMoves = []
        for piece in side:
            # For moving forward
            for i in range(piece[0]-1, 0, -1):
                if gameBoard.grid[i][piece[1]] == 0:
                    possibleMoves.append((piece, (i, piece[1])))
                else:
                    break
            # For moving backwards
            for i in range(piece[0]+1, 8):
                if gameBoard.grid[i][piece[1]] == 0:
                    possibleMoves.append((piece, (i, piece[1])))
                else:
                    break
            # For moving left
            for i in range(piece[1]-1, 0, -1):
                if gameBoard.grid[piece[0]][i] == 0:
                    possibleMoves.append((piece, (piece[0], i)))
                else:
                    break
            # For moving right
            for i in range(piece[1]+1, 8):
                if gameBoard.grid[piece[0]][i] == 0:
                    possibleMoves.append((piece, (piece[0], i)))
                else:
                    break
        return possibleMoves