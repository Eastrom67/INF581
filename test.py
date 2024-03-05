from abstract_agent import Agent
from simple_agent import DecisionTreeAgent
from environment import Environment
from board import Board
from game_runner import GameRunner

gr = GameRunner()

agent_A = DecisionTreeAgent(nb_future=1)
agent_B = DecisionTreeAgent(nb_future=2)

# gr.run_and_show(agent_A, agent_B, console = True, gif = True)

win_A, win_B, mean_t = gr.compare_agents(agent_A, agent_B, 10)
print(f'Agent A won {win_A} times')
print(f'Agent B won {win_B} times')
print(f'Mean time of a game: {mean_t}')


