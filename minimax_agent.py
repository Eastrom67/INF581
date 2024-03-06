from abstract_agent import Agent
from board import Board
import numpy as np
import copy


class MinimaxAgent(Agent):
    """
    Class that represents a simple agent.
    """

    def __init__(self, nb_future=0, fear=2, hope=2) -> None:
        super().__init__()
        self.nb_future = nb_future
        self.fear = fear
        self.hope = hope
        self.victory = 1000


    def move(self, board : Board):
        """
        Method that returns the move of the agent.
        Implements the minimax algorithm.
        """
        move, _ = self.minimax(board, 0)
        return move

    def minimax(self, board : Board, depth : int):
        moves = board.get_allowed_moves()
        boards = [copy.deepcopy(board) for move in moves]
        for i in range(len(boards)):
            boards[i].move(moves[i])

        if depth == self.nb_future:
            scores = [self.board_score(b) for b in boards]

        else:
            scores = [self.minimax(board, depth + 1)[1] for board in boards]

        if depth %2 == 0:
            best = np.argmax(scores)
        else:
            best = np.argmin(scores)
        return moves[best], scores[best]

    def board_score(self, board : Board):
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

        score = white_pieces + 5 * white_queens - black_pieces - 5 * black_queens + potential_moves_white - potential_moves_black
        
        return score