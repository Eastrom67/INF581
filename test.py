from abstract_agent import Agent
from minimax_agent import MinimaxAgent
from one_step_agent import OneStepAgent
from environment import Environment
from board import Board
from game_runner import GameRunner
from tree_agent import DecisionTreeAgent
from naive_Q_learning_agent import Q_learning_Agent
from deepQ_agent import DeepQAgent
from naive_deep_qlearning import naive_QNetwork
from deepQ_utils import QNetwork
import torch
import time

gr = GameRunner()
side = 10
size = int(side**2/2) # The size of the board
move_length = 2 # The length of a move. We do not consider longer moves for now
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the deep q-network
q_network = QNetwork(7*size, nn_l1=100, nn_l2=100).to(device)
q_network = torch.load(f"q_network.pth").to(device)
deep_q_agent = DeepQAgent(q_network)

# Load the naive q-networks
naive_q_network = naive_QNetwork(5*size, size**move_length, nn_l1=128, nn_l2=128).to(device)
naive_q_network = torch.load(f"naive_q_network20.pth").to(device)
naive_q_agent = Q_learning_Agent(naive_q_network)



agents = {'Random agent': Agent(), 'MinMax': MinimaxAgent(nb_future=2), 'Naive_qnetwork': naive_q_agent, 'Deep_qnetwork': deep_q_agent}
n_games = 100
for A_name, agent_A in agents.items():
    for B_name, agent_B in agents.items():
        before = time.time()
        win_A, win_B, mean_t = gr.compare_agents(agent_A, agent_B, n_games)
        after = time.time()
        print(f'{A_name} won {win_A} times')
        print(f'{B_name} won {win_B} times')
        print(f'Mean time of a game: {mean_t}')

        print(f'Time elapsed per move {round((after - before) / (mean_t * n_games), 3)} seconds')
