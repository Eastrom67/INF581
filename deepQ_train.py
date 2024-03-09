from deepQ_utils import QNetwork, EpsilonScheduler, MinimumExponentialLR, initialize_q_network
from board import Board
from deepQ_agent import DeepQAgent
from abstract_agent import Agent
from game_runner import GameRunner
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

side = 10
n_games = 50
n_epochs = 20

gm = GameRunner()

q_network = initialize_q_network(side).to(device)
agent = DeepQAgent(q_network)
random_agent = Agent()


success_rates = []

for epoch in range(n_epochs):
    q_network = agent.q_network
    gamma = 0.9
    epsilon_start = 0.9
    epsilon_min = 0.1
    epsilon_decay = 0.99
    epsilon_scheduler = EpsilonScheduler(epsilon_start, epsilon_min, epsilon_decay)
    lr = 0.01
    optimizer = torch.optim.Adam(q_network.parameters(), lr=lr)
    lr_scheduler = MinimumExponentialLR(optimizer, 0.9, min_lr=0.0001)
    loss_fn = torch.nn.MSELoss()

    agent.train_imediate_reward(
        n_games,
        gamma,
        epsilon_scheduler,
        optimizer,
        lr_scheduler,
        loss_fn,
        device
    )

    # Test the agent against a random agent
    A, B, t = gm.compare_agents(agent, random_agent, 100)
    print(f"Epoch {epoch} - Agent won {A} games, Random agent won {B} games")
    success_rates.append(A)


    torch.save(agent.q_network, "q_network.pth")

plt.plot(success_rates)
plt.xlabel("Epoch")
plt.ylabel("Success rate")
plt.show()