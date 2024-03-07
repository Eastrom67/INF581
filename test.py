from abstract_agent import Agent
from minimax_agent import MinimaxAgent
from one_step_agent import OneStepAgent
from environment import Environment
from board import Board
from game_runner import GameRunner
import time

gr = GameRunner()

agent_A = OneStepAgent()
agent_B = MinimaxAgent(nb_future=2)
n_games = 1

# gr.run_and_show(agent_A, agent_B, console = True, gif = True)
before = time.time()
win_A, win_B, mean_t = gr.compare_agents(agent_A, agent_B, n_games)
after = time.time()
print(f'Agent A won {win_A} times')
print(f'Agent B won {win_B} times')
print(f'Mean time of a game: {mean_t}')

print(f'Time elapsed per move {round((after - before) / (mean_t * n_games), 3)} seconds')
