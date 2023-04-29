# -*- coding: utf-8 -*-
"""
@author: sruiz2
"""

import gym
from gym import spaces
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from typing import Any, List, Sequence, Tuple
import torch
import random

import torch.nn as nn
import torch.nn.functional as F


# Set seed for experiment reproducibility
seed = 10
tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

#%%
class AdaptiveCruiseControlEnv(gym.Env):
    def __init__(self):
        # Define action and observation space
        self.action_space = spaces.Box(low=-3.0, high=2.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-100.0, -100.0, 0.0]), high=np.array([100.0, 100.0, 100.0]), dtype=np.float32
        )

        # Initialize state variables
        self.velocity_error = 0.0
        self.integral_velocity_error = 0.0
        self.ego_velocity = 0.0
        self.lead_velocity = 0.0
        self.relative_distance = 0.0
        self.target_velocity = 25.0  # Set a default target velocity
        self.prev_action = 0.0

        # Define physical constants
        self.dt = 0.1  # Time step
        self.max_acceleration = 2.0  # Maximum acceleration
        self.max_braking = 4.0  # Maximum braking deceleration
        self.deceleration_target = -4.0  # Target deceleration for following a lead car

    def step(self, action):
        # Saturate the action to the maximum and minimum limits
        action = np.clip(action, -3.0, 2.0)[0]

        # Update state variables based on action
        self.ego_velocity += action * self.dt

        # Update state variables based on dynamics
        self.velocity_error = (self.ego_velocity - self.target_velocity)
        self.integral_velocity_error += self.velocity_error * self.dt

        # Calculate reward
        Mt = 1 if self.ego_velocity ** 2 < 0.25 else 0
        reward = -(0.1 * self.ego_velocity ** 2 + self.prev_action ** 2) + Mt

        # Update previous action
        self.prev_action = action

        # Check if following lead car
        if self.target_velocity < self.ego_velocity:
            reward += self.deceleration_target - self.ego_velocity

        # Check if car has gone off road or collided with lead car
        done = (self.ego_velocity < 0) or (self.relative_distance < 0)

        # Update lead car position
        self.relative_distance += (self.lead_velocity - self.ego_velocity) * self.dt

        # Return observation, reward, done, and info
        observation = np.array([self.velocity_error, self.integral_velocity_error, self.ego_velocity])
        return observation, reward, done, {}

    def reset(self):
        # Reset state variables
        self.velocity_error = 0.0
        self.integral_velocity_error = 0.0
        self.ego_velocity = 0.0
        self.prev_action = 0.0

        # Set a new target velocity
        self.target_velocity = np.random.uniform(20.0, 30.0)

        # Initialize lead car position and velocity
        self.relative_distance = np.random.uniform(50.0, 100.0)
        self.lead_velocity = np.random.uniform(self.target_velocity - 5.0, self.target_velocity + 5.0)

        # Return initial observation
        observation = np.array([self.velocity_error, self.integral_velocity_error, self.ego_velocity])
        return observation


#%% Actor network
class DDPGActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(DDPGActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.max_action
        return x

#%% Critic network
class DDPGCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DDPGCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        x = F.relu(self.fc1(xu))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#%% Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = DDPGActor(state_dim, action_dim, max_action)
        self.actor_target = DDPGActor(state_dim, action_dim, max_action)
        self.critic = DDPGCritic(state_dim, action_dim)
        self.critic_target = DDPGCritic(state_dim, action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.max_action = max_action
        self.loss = nn.MSELoss()
        self.gamma = 0.99
        self.tau = 0.005
        self.memory = []
        self.batch_size = 256

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([transition[0] for transition in batch]))
        actions = torch.FloatTensor(np.array([transition[1] for transition in batch]))
        rewards = torch.FloatTensor(np.array([transition[2] for transition in batch]))
        next_states = torch.FloatTensor(np.array([transition[3] for transition in batch]))
        dones = torch.FloatTensor(np.array([transition[4] for transition in batch]))

        # Critic loss
        next_actions = self.actor_target(next_states)
        target_Q = self.critic_target(next_states, next_actions).detach()
        target_Q = rewards.unsqueeze(1) + (self.gamma * target_Q * (1 - dones.unsqueeze(1)))
        current_Q = self.critic(states, actions)
        critic_loss = self.loss(current_Q, target_Q)

        # Actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad   

#%%
## Train the agent
env = AdaptiveCruiseControlEnv()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = DDPGAgent(state_dim, action_dim, max_action)

num_episodes = 2000
max_steps = 1000

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    done = False
    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        episode_reward += reward
        state = next_state
        agent.train()
        if done:
            break
    print(f"Episode: {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}")

#%%
