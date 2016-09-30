"""
- Each spot on the board is represented by (1,0), (0,0), or (0,1)
- (1,0) represents that your player is present on that spot
- (0,0) represents that no player is present on a spot
- (0,1) represents that an opponents player is on a spot
- A single move is represented by a 64x4 matrix where the first two columns are the
current state of the board and the next two are the state of the board after a move.
"""