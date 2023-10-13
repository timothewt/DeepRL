from typing import Any

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from torch.distributions import Categorical

from algorithms.Algorithm import Algorithm
from models.FCNet import FCNet


class REINFORCE(Algorithm):

	def __init__(self, config: dict[str | Any]):
		"""
		:param config:
			env_name (str) : name of the environment in the Gym registry
			lr (float): learning rate
			gamma (float): discount factor
		"""
		super().__init__(config=config)

		# Device

		self.device = config.get("device", torch.device("cpu"))

		# Environment

		assert config.get("env_name", None) is not None, "No environment provided!"
		self.env: gym.Env = gym.make(config.get("env_name", None))
		self.max_episode_steps = self.env.spec.max_episode_steps

		# Algorithm hyperparameters

		self.lr: float = config.get("lr", .0002)
		self.gamma: float = config.get("gamma", .99)

		# Policy

		assert isinstance(self.env.action_space, spaces.Discrete), "Only discrete action spaces are currently supported"
		assert isinstance(self.env.observation_space, spaces.Box) or \
			isinstance(self.env.observation_space, spaces.Discrete),\
			"Only Box and Discrete observation spaces currently supported"

		if isinstance(self.env.observation_space, spaces.Box):
			input_size = int(np.prod(self.env.observation_space.shape))
		else:
			input_size = self.env.observation_space.n

		policy_config = {
			"input_size": input_size,
			"output_size": self.env.action_space.n,
			"hidden_layers_nb": config.get("hidden_layers_nb", 3),
			"hidden_size": config.get("hidden_size", 32),
			"output_function": nn.Softmax(dim=-1),
		}
		self.policy_network: nn.Module = FCNet(config=policy_config).to(self.device)
		self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.lr)

		# Training stats

		self.rewards = []
		self.losses = []
		self.log_freq = config.get("log_freq", 50)

	def train(self, max_steps: int, plot_training_stats: bool = False) -> None:

		self.rewards = []
		self.losses = []

		steps = 0
		episode = 0

		states = torch.empty((self.max_episode_steps,) + self.env.observation_space.shape, device=self.device, dtype=torch.float)
		actions = torch.empty((self.max_episode_steps,), device=self.device, dtype=torch.float)
		rewards = torch.empty((self.max_episode_steps,), device=self.device, dtype=torch.float)

		print("==== STARTING TRAINING ====")

		while steps <= max_steps:
			self.rewards.append(0)
			self.losses.append(0)

			current_episode_step = 0
			log_probs = torch.empty((self.max_episode_steps,), device=self.device, dtype=torch.float)

			done = False
			new_obs, _ = self.env.reset()
			obs = torch.from_numpy(new_obs).to(self.device).float()

			# Collecting data for an episode

			while not done and current_episode_step < self.max_episode_steps:
				probs = self.policy_network(obs)
				m = Categorical(probs=probs)
				action = m.sample()
				log_probs[current_episode_step] = m.log_prob(action)

				new_obs, reward, done, _, _ = self.env.step(action.item())

				states[current_episode_step] = obs
				actions[current_episode_step] = action
				rewards[current_episode_step] = reward

				obs = torch.from_numpy(new_obs).to(self.device).float()
				current_episode_step += 1

			# Computing G values

			G = torch.zeros((current_episode_step,), device=self.device)
			G[-1] = rewards[current_episode_step - 1]
			for t in range(current_episode_step - 2, -1, -1):
				G[t] = G[t + 1] * self.gamma + rewards[t]
			G = (G - G.mean()) / (G.std() + 1e-9)

			# Updating the network
			self.optimizer.zero_grad()
			loss = - (log_probs[:current_episode_step] * G).mean()
			loss.backward()
			self.optimizer.step()

			self.rewards[episode] = rewards[:current_episode_step].sum().item()
			self.losses[episode] = loss.item()

			if episode % self.log_freq == 0:
				self.log_rewards(rewards=self.rewards, episode=episode, avg_period=10)

			episode += 1
			steps += current_episode_step

		print("==== TRAINING COMPLETE ====")
