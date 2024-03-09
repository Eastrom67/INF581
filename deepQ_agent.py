from abstract_agent import Agent
from deepQ_utils import QNetwork
from deepQ_utils import EpsilonScheduler
from torch.optim.lr_scheduler import _LRScheduler
from board import Board
import numpy as np
import torch
import board_metrics

class DeepQAgent(Agent):
    """
    Class that represents an agent.
    """
    def __init__(self, q_network_: QNetwork) -> None:
        self.q_network = q_network_
        pass

    def move(self, board : Board):
        """
        Method that returns the move of the agent.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        moves = board.get_allowed_moves()
        q_values = np.zeros(len(moves))
        for m in range(len(moves)):
            (sequence, taken) = moves[m]
            state_tensor = torch.tensor(self.to_first_layer(board.board, sequence[0], sequence[1]), dtype=torch.float32, device=device).unsqueeze(0)
            q_value = self.q_network(state_tensor)
            q_values[m] = q_value
        action = moves[np.argmax(q_values)]
        return action
    
    def move_random(self, board : Board):
        """
        Method that returns a random possible move.
        """
        moves = board.get_allowed_moves()
        return moves[np.random.randint(len(moves))]

    
    def to_first_layer(self, state: np.array, tile_1: int, tile_2: int) -> np.ndarray:
        """
        Method that converts the state aznd the action to the first layer of the neural network.
        """
        N = len(state)
        output = np.zeros(7*N)
        for i in range(N):
            output[5*i] = (state[i]==0)
            output[5*i+1] = (state[i]==1)
            output[5*i+2] = (state[i]==-1)
            output[5*i+3] = (state[i]==2)
            output[5*i+4] = (state[i]==-2)
        for i in range(N):
            output[5*N + i] = (i==tile_1)
        for i in range(N):
            output[6*N + i] = (i==tile_2)
        return output
    
    def train_imediate_reward(self, 
                              n_games: int, 
                              gamma: float, 
                              epsilon_scheduler: EpsilonScheduler,
                              optimizer: torch.optim.Optimizer,
                              lr_scheduler: _LRScheduler,
                              loss_fn: torch.nn.modules.loss._Loss,
                              device: torch.device
                              ):
        """
        Method that trains the agent on n_games.
        """
        for g in range(n_games):
            board = Board()

            while True:

                # Check if the game is over
                if board.is_final():
                    break

                # Choose the action
                if epsilon_scheduler.random_action():
                    action = self.move_random(board)
                else:
                    action = self.move(board)

                # Compute the q_value
                (sequence, taken) = action
                state_tensor = torch.tensor(self.to_first_layer(board.board, sequence[0], sequence[1]), dtype=torch.float32, device=device).unsqueeze(0)
                q_value = self.q_network(state_tensor)

                # Apply the action, get the reward and transpose the board
                board.move(action)
                reward = board_metrics.q_board_metric(board)
                board.transpose()
                state = board.board

                # Check if the game is over
                if board.is_final():
                    next_q_value = torch.tensor([[0]], dtype=torch.float32, device=device)
                
                    target = reward + gamma * next_q_value

                    # Compute the loss
                    loss = loss_fn(q_value, target)

                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()

                    # Go to the next game
                    break
                else:
                    # Compute the next q_value
                    action_adversary = self.move(board)
                    (sequence, taken) = action_adversary
                    next_state_tensor = torch.tensor(self.to_first_layer(board.board, sequence[0], sequence[1]), dtype=torch.float32, device=device).unsqueeze(0)
                    next_q_value = self.q_network(next_state_tensor)

                    # Adversary plays
                    board.move(action_adversary)
                    board.transpose()

                    # Compute the target
                    target = reward + gamma * next_q_value

                    # Compute the loss
                    loss = loss_fn(q_value, target)

                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()

                   