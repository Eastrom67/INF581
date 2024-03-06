from abstract_agent import Agent
from minimax_agent import MinimaxAgent
from one_step_agent import OneStepAgent
from environment import Environment
from board import Board
from game_runner import GameRunner

gr = GameRunner()

agent_A = OneStepAgent()
agent_B = MinimaxAgent(nb_future=1)

# gr.run_and_show(agent_A, agent_B, console = True, gif = True)

win_A, win_B, mean_t = gr.compare_agents(agent_A, agent_B, 5)
print(f'Agent A won {win_A} times')
print(f'Agent B won {win_B} times')
print(f'Mean time of a game: {mean_t}')


