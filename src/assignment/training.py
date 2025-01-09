import argparse
import math
import torch
import torch.optim as optim
import torch.nn as nn
import random

from collections import namedtuple
from itertools import count

import sys
import os

# Add the directory containing env_hiv.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

from train import ProjectAgent

from replay import ReplayMemory
from DQN_model import DQN

def get_args():
    parser = argparse.ArgumentParser(description="Hyperparameters for training the model.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for the reward")
    parser.add_argument("--eps_start", type=float, default=0.9, help="Starting value of epsilon for exploration")
    parser.add_argument("--eps_end", type=float, default=0.05, help="Ending value of epsilon for exploration")
    parser.add_argument("--eps_decay", type=int, default=1000, help="Decay rate of epsilon for exploration")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update of target parameters")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer")

    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")
    parser.add_argument("--path", type=str, default="model.pt", help="Path to save the model")

    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to train the model")

    args = parser.parse_args()

    return args

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def optimize_model(memory, agent, optimizer, device, args):
    policy_net = agent.policy_net
    target_net = agent.target_net

    batch_size = args.batch_size
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)

    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * args.gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    print(f"Loss: {loss.item()}", end="\r")

def main(args):
    print('Hello World!')
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Set gymnasium like HIV environment
    env = TimeLimit(
        env=HIVPatient(domain_randomization=False), max_episode_steps=200
    )  # The time wrapper limits the number of steps in an episode at 200.

    #Init policy and target networks
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    #Init replay buffer and optimizer
    memory = ReplayMemory(10000)
    optimizer = optim.AdamW(policy_net.parameters(), lr=args.lr)

    #Init agent
    agent = ProjectAgent()
    agent.set_policy_net(policy_net)
    agent.set_target_net(target_net)

    ### TRAINING LOOP ###
    steps_done = 0
    num_episodes = args.num_episodes

    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        rewards = []
        for t in count():
            agent.set_epslion(args.eps_end + (args.eps_start - args.eps_end) * math.exp(-1. * steps_done / args.eps_decay))
            print(agent.epsilon, end='\r')
            action = agent.act(state)
            steps_done += 1

            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            rewards.append(reward)

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(memory, agent, optimizer, device, args)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*args.tau + target_net_state_dict[key]*(1-args.tau)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                # episode_durations.append(t + 1)
                # plot_durations()
                break
        
        print(sum(rewards)/len(rewards))


    print('Complete')
    agent.save(args.path)

if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    main(args)