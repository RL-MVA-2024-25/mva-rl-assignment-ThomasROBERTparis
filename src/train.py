from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

import random
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from assignment.DQN_model import DQN

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

class ProjectAgent: #For Q-learning
    def act(self, observation, use_random=False):
        observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        sample = random.random()
        if sample > self.epsilon :
            with torch.no_grad():
                return torch.tensor(self.policy_net(observation).max(1).indices.view(1, 1))
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path, map_location=torch.device('cpu'))

    def load(self):
        path = 'model.pt'
        self.policy_net = DQN().to(device)
        self.policy_net.load_state_dict(torch.load(path), map_location=torch.device('cpu'))
        self.epsilon = 0.01
    
    ### custom methods ###
    def set_policy_net(self, net):
        self.policy_net = net
    
    def set_target_net(self, net):
        self.target_net = net
    
    def set_epslion(self, epsilon):
        self.epsilon = epsilon