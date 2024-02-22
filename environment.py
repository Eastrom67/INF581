import numpy as np

class Environment:
    # Playing on black board spaces, the spaces are numbered from top to bottom, left to right.
    # Top left corner is white.

    def __init__(self):
        # International checkers rules except no maximum capture rule
        self.side = 10
        self.half = int(self.side / 2)

        # Number of usable squares
        self.size = int(self.side * self.side / 2)

        # Registering only playable squares, 0 = empty, 1 = white, -1 = black, 2 = white queen, -2 = black queen.
        # Makes it easy to "flip" the board
        self.board = np.array([0 for i in range(self.size)]) 
        self.turn = 0
        self.board[0:20] -= 1
        self.board[30:50] += 1
    
    # Functions to convert between 2D space and np.array indices of squares
    def coordinates(self, square_id):
        y = square_id // self.half
        x = 2 * (square_id % self.half) + (1 - y % 2)
        return x, y
    
    def square_id(self, coordinates):
        x, y = coordinates
        return int((y * self.half) + (x // 2))
    
    # Creates board image for easy matplotlib.pyplot.imshow
    def board_image(self):
        # White board
        image = np.ones((self.side, self.side, 3))

        for i in range(self.size):
            x, y = self.coordinates(i)
            state = self.board[i]

            if state == 0:
                image[y, x] = [0, 0, 0]
            elif state == -1:
                image[y, x] = [1.0, 0, 0]
            elif state == 1:
                image[y, x] = [0, 1.0, 0]
            elif state == -2:
                image[y, x] = [0.3, 0, 0]
            elif state == 2:
                image[y, x] = [0, 0.3, 0]
        return image
    
    # Checks whether a hypothetic move would be valid, supposing piece_type exists at init_position.
    # This assumption allows us to reason on hypothetic move chains without messing the board.
    def is_valid(self, init_position, end_position, piece_type, already_jumped):
        # Piece needs to land in empty spot
        if self.board[end_position] != 0:
            return False, None
        
        valid = False
        jumped_piece = None        
        x1, y1 = self.coordinates(init_position)
        x2, y2 = self.coordinates(end_position)

        # White piece
        if (self.turn == 0) and (piece_type == 1): 
            # If move is first in the chain, one-diagonal, frontwards :
            # Basic move, valid
            if(self.board[init_position] == 1) and (y1 == y2 + 1) and (abs(x1 - x2) == 1):
                valid = True
                jumped_piece = None

            x_enemy = (x1 + x2) / 2
            y_enemy = (y1 + y2) / 2
            enemy_position = self.square_id((x_enemy, y_enemy))
            # If two-diagonal move with enemy piece in the middle, either first move or in chain :
            # Basic take, valid
            if((self.board[init_position] == 1) or len(already_jumped) > 0) and (abs(y1 - y2) == 2) and (abs(x1 - x2) == 2) and (self.board[enemy_position] < 0):
                valid = True
                jumped_piece = enemy_position

        # White queen
        if (self.turn == 0) and (piece_type == 2):
            # Queen move
            # First diagonal
            if ((x1 - x2) == (y1 - y2)):
                min_x = min(x1,x2)
                min_y = min(y1,y2)
                diff = int(abs(x1 - x2))
                pieces_present = 0
                piece_pos = None

                for delta in range(1, diff):
                    square = self.square_id((min_x + delta, min_y + delta))
                    # Enemy piece, only one accepted
                    if self.board[square] < 0:
                        pieces_present += 1
                        piece_pos = square
                    # Friendly piece, none accepted
                    if self.board[square] > 0:
                        pieces_present += 2

                # If taking, and in chain or first move, valid
                if (pieces_present == 1) and (len(already_jumped) > 0 or (self.board[init_position] == 2)):
                    valid = True
                    jumped_piece = piece_pos
                # If moving in first move, valid
                elif (pieces_present == 0) and (self.board[init_position] == 2):
                    valid = True
                else:
                    valid = False

            # Second diagonal
            if ((x1 - x2) == (y2 - y1)):
                min_x = min(x1,x2)
                max_y = max(y1,y2)
                diff = int(abs(x1 - x2))
                pieces_present = 0
                piece_pos = None

                for delta in range(1, diff):
                    square = self.square_id((min_x + delta, max_y - delta))
                    # Enemy piece, only one accepted
                    if self.board[square] < 0:
                        pieces_present += 1
                        piece_pos = square
                    # Friendly piece, none accepted
                    if self.board[square] > 0:
                        pieces_present += 2

                # If taking, and in chain or first move, valid
                if (pieces_present == 1) and (len(already_jumped) > 0 or (self.board[init_position] == 2)):
                    valid = True
                    jumped_piece = piece_pos
                # If moving in first move, valid
                elif (pieces_present == 0) and (self.board[init_position] == 2):
                    valid = True
                else:
                    valid = False

        # Black piece
        if (self.turn == 1) and (piece_type == -1): 
            # If move is first in the chain, one-diagonal, frontwards :
            # Basic move, valid
            if(self.board[init_position] == -1) and (y1 == y2 - 1) and (abs(x1 - x2) == 1):
                valid = True
                jumped_piece = None

            x_enemy = (x1 + x2) / 2
            y_enemy = (y1 + y2) / 2
            enemy_position = self.square_id((x_enemy, y_enemy))
            # If two-diagonal move with enemy piece in the middle, either first move or in chain :
            # Basic take, valid
            if((self.board[init_position] == -1) or len(already_jumped) > 0) and (abs(y1 - y2) == 2) and (abs(x1 - x2) == 2) and (self.board[enemy_position] > 0):
                valid = True
                jumped_piece = enemy_position

        # Black queen
        if (self.turn == 1) and (piece_type == -2):
            # Queen move
            # First diagonal
            if ((x1 - x2) == (y1 - y2)):
                min_x = min(x1,x2)
                min_y = min(y1,y2)
                diff = int(abs(x1 - x2))
                pieces_present = 0
                piece_pos = None
                
                for delta in range(1, diff):
                    square = self.square_id((min_x + delta, min_y + delta))
                    # Enemy piece, only one accepted
                    if self.board[square] > 0:
                        pieces_present += 1
                        piece_pos = square
                    # Friendly piece, none accepted
                    if self.board[square] < 0:
                        pieces_present += 2

                # If taking, and, in chain or first move, valid
                if (pieces_present == 1) and (len(already_jumped) > 0 or (self.board[init_position] == -2)):
                    valid = True
                    jumped_piece = piece_pos
                # If moving in first move, valid
                elif (pieces_present == 0) and (self.board[init_position] == -2):
                    valid = True
                else:
                    valid = False

            # Second diagonal
            if ((x1 - x2) == (y2 - y1)):
                min_x = min(x1,x2)
                max_y = max(y1,y2)
                diff = int(abs(x1 - x2))
                pieces_present = 0
                piece_pos = None

                for delta in range(1, diff):
                    square = self.square_id((min_x + delta, max_y - delta))
                    # Enemy piece, only one accepted
                    if self.board[square] > 0:
                        pieces_present += 1
                        piece_pos = square
                    # Friendly piece, none accepted
                    if self.board[square] < 0:
                        pieces_present += 2

                # If taking, and in chain or first move, valid
                if (pieces_present == 1) and (len(already_jumped) > 0 or (self.board[init_position] == -2)):
                    valid = True
                    jumped_piece = piece_pos
                # If moving in first move, valid
                elif (pieces_present == 0) and (self.board[init_position] == -2):
                    valid = True
                else:
                    valid = False

        # To avoid double captures, we need to account for pieces already taken : already_jumped.
        if jumped_piece in already_jumped:
            return False, None
        
        return valid, jumped_piece
    
    # A move may contain a certain amount of captures.
    # We have to check the validity of the move, and execute it.
    # Finally, we need to crown pieces when they reach the final row.    
    def move(self, square_sequence):
        start_position = square_sequence[0]
        start_state = self.board[start_position]

        if start_state == 0:
            return "Invalid move"
        
        jumped = [] # List of jumped enemies
        for move_id in range(len(square_sequence) - 1):
            init_position = square_sequence[move_id]
            end_position = square_sequence[move_id + 1]
            valid, jumped_enemy = self.is_valid(init_position, end_position, start_state, jumped)
            if not valid:
                return "Invalid move"
            if jumped_enemy is not None:
                jumped.append(jumped_enemy)
        
        # Place piece at the last square of the move
        self.board[square_sequence[-1]] = start_state
        self.board[start_position] = 0

        # Remove pieces taken
        for position in jumped:
            self.board[position] = 0

        # Promote white pieces on top row
        for square in range(self.half):
            if self.board[square] == 1:
                self.board[square] = 2

        # Promote black pieces on bottom row
        for square in range(self.size - self.half, self.size):
            if self.board[square] == -1:
                self.board[square] = -2    

        # Next turn
        self.turn = 1 - self.turn

        return "Valid move"
