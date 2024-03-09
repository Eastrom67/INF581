from board import Board
import numpy as np

def basic_score(board : Board):
        """
        Method that returns the score of the board.
        """
        white_pieces = 0
        white_queens = 0
        black_pieces = 0
        black_queens = 0
        for tile in board.board:
            if tile == -2:
                black_queens += 1
            if tile == -1:
                black_pieces += 1
            if tile == 1:
                white_pieces += 1
            if tile == 2:
                white_queens += 1

        potential_moves_white = len(board.get_allowed_moves())

        board.transpose()
        potential_moves_black = len(board.get_allowed_moves())
        board.transpose()

        score = white_pieces + 5 * white_queens - black_pieces - 5 * black_queens
        
        return score


def q_board_metric(envi):
        """
        Simple evaluation function for the checkers game.
        """
        b = envi.board

        winning_reward = 500
        
        men1 = np.sum((b==1))
        men2 = np.sum((b==-1))
        kings1 = np.sum((b==2))
        kings2 = np.sum((b==-2))
        
        if men2+kings2 == 0:
            return winning_reward
        elif men1+kings1 == 0:
            return -winning_reward
            
        men = men1-men2
        kings = kings1-kings2

        potential_moves_white = len(envi.get_allowed_moves())
        envi.transpose()
        potential_moves_black = len(envi.get_allowed_moves())
        envi.transpose()

        potential = potential_moves_white - potential_moves_black

        score = men + 5*kings #+ potential
        
        return score