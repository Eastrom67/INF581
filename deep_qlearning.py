import environment as env
import matplotlib.pyplot as plt
import numpy as np
import torch

def show_board(envi):
    plt.figure()
    plt.imshow(envi.board_image())
    plt.show()

winning_reward = 1000

def get_metrics(envi, player):
    """
    Simple evaluation function for the checkers game.
    """
    b = envi.board
    
    if player == 0:
        men1 = np.sum((b==1))
        men2 = np.sum((b==-1))
        kings1 = np.sum((b==2))
        kings2 = np.sum((b==-2))
    else:
        men1 = np.sum((b==-1))
        men2 = np.sum((b==1))
        kings1 = np.sum((b==-2))
        kings2 = np.sum((b==2))
    
    if men2+kings2 == 0:
        return winning_reward
    elif men1+kings1 == 0:
        return -winning_reward
        
    men = men1-men2
    kings = kings1-kings2
    score = men + 3*kings
    return score

envi = env.Environment()

state_dim = envi.size
action_dim = envi.size**2

class QNetwork(torch.nn.Module):
    """
    A Q-Network implemented with PyTorch.

    Attributes
    ----------
    layer1 : torch.nn.Linear
        First fully connected layer.
    layer2 : torch.nn.Linear
        Second fully connected layer.
    layer3 : torch.nn.Linear
        Third fully connected layer.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Define the forward pass of the QNetwork.
    """

    def __init__(self, n_observations: int, n_actions: int, nn_l1: int, nn_l2: int):
        """
        Initialize a new instance of QNetwork.

        Parameters
        ----------
        n_observations : int
            The size of the observation space.
        n_actions : int
            The size of the action space.
        nn_l1 : int
            The number of neurons on the first layer.
        nn_l2 : int
            The number of neurons on the second layer.
        """
        super(QNetwork, self).__init__()
        self.layer1 = torch.nn.Linear(n_observations, nn_l1)
        self.layer2 = torch.nn.Linear(nn_l1, nn_l2)
        self.layer3 = torch.nn.Linear(nn_l2, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the QNetwork.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor (state).

        Returns
        -------
        torch.Tensor
            The output tensor (Q-values).
        """

        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = self.layer3(x)

        return x

def qvalues_to_best_possible_move(envi, q_values, player):
    """
    Convert the Q-values to the best possible move.
    """
    
    possible_moves = [possible_move[0] for possible_move in envi.possible_moves(player)]
    possible_moves = list(set([move[0]*envi.size+move[1] for move in possible_moves]))
    
    q_values = q_values.detach().cpu().numpy()[0]
    action = possible_moves[0]
    for possible_move in possible_moves:
        if q_values[possible_move] > q_values[action]:
            action = possible_move

    return [action // envi.size, action % envi.size]

def test_q_network_agent(env, q_network: torch.nn.Module, num_episode: int = 1, render: bool = True):
    """
    Test a naive agent in the given environment using the provided Q-network.

    Parameters
    ----------
    env : gym.Env
        The environment in which to test the agent.
    q_network : torch.nn.Module
        The Q-network to use for decision making.
    num_episode : int, optional
        The number of episodes to run, by default 1.
    render : bool, optional
        Whether to render the environment, by default True.

    Returns
    -------
    List[int]
        A list of rewards per episode.
    """
    episode_reward_list = []

    for episode_id in range(num_episode):

        envi = env.Environment()
        state = envi.board
        done = False
        episode_reward = 0
        player = 0
        action = None

        while not done:
            if render and action is not None:
                print(f"Action: {action}")
                print(f'State: {state}')
                print(f"Episode counting reward: {episode_reward}")
                print()
                print(f"Player {player} move")
            
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            q_values = q_network(state_tensor)
            
            action = qvalues_to_best_possible_move(envi, q_values, player)
            envi.move(action)
            reward = get_metrics(envi, player)
            state = envi.board
            done = np.abs(reward) == winning_reward
            player = 1 - player
            
            episode_reward += reward

        episode_reward_list.append(episode_reward)
        print(f"Episode reward: {episode_reward}")

    return episode_reward_list

"""
moves = [[30,26],[15,20],[26,15]]
player = 0
for move in moves:
    envi.move(move)
    print(get_metrics(envi, player))
    player = 1-player"""
     

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_network = QNetwork(state_dim, action_dim, nn_l1=128, nn_l2=128).to(device)

test_q_network_agent(env, q_network, num_episode=5)