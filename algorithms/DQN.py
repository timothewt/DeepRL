from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from gymnasium import spaces
from torch import tensor, nn as nn
import gymnasium as gym
import numpy as np

from algorithms.Algorithm import Algorithm
from models.FCNet import FCNet


class ReplayMemory:

	def __init__(self, max_length: int = 10_000, state_shape: tuple[int] = (1,)):
		self.states = np.empty((max_length,) + state_shape, dtype=float)
		self.actions = np.empty(max_length, dtype=int)
		self.rewards = np.empty(max_length, dtype=float)
		self.next_state = np.empty((max_length,) + state_shape, dtype=float)
		self.is_terminal = np.empty(max_length, dtype=bool)

		self.max_length = max_length
		self.current_length = 0
		self.i = 0

	def push(self, state, action, reward, next_state, is_terminal) -> ReplayMemory:
		self.i %= self.max_length
		self.current_length = min(self.current_length + 1, self.max_length)

		self.states[self.i] = state
		self.actions[self.i] = action
		self.rewards[self.i] = reward
		self.next_state[self.i] = next_state
		self.is_terminal[self.i] = is_terminal
		self.i += 1

	def sample(self, batch_size: int):
		indices = np.random.choice(self.current_length, size=batch_size)
		return self.states[indices], \
			self.actions[indices], \
			self.rewards[indices], \
			self.next_state[indices], \
			self.is_terminal[indices]


class DQN(Algorithm):

	def __init__(self, config: dict):

		super().__init__(config=config)

		# Algorithm hyperparameters

		self.lr: float = config.get("lr", .0004)
		self.gamma: float = config.get("gamma", .99)
		self.eps: float = config.get("eps", 1)
		self.eps_decay: float = config.get("eps_decay", .9995)
		self.eps_min: float = config.get("eps_min", .01)
		self.batch_size: int = config.get("batch_size", 64)
		self.target_network_update_frequency: int = config.get("target_network_update_frequency", 100)

		self.replay_memory: ReplayMemory = ReplayMemory(config.get("max_replay_buffer_size", 10_000), self.env.observation_space.shape)

		# Policies

		self.policy_model: type[nn.Module] = config.get("policy_model", None)
		assert self.policy_model is not None, "No policy architecture provided!"

		assert isinstance(self.env.observation_space, spaces.Box) or isinstance(self.env.observation_space, spaces.Discrete),\
			"Only Box and Discrete spaces currently supported"

		input_size = 0
		if isinstance(self.env.observation_space, spaces.Box):
			input_size = int(np.prod(self.env.observation_space.shape))
		elif isinstance(self.env.observation_space, spaces.Discrete):
			input_size = self.env.observation_space.n

		policy_config = {
			"input_size": input_size,
			"output_size": self.env.action_space.n,
			"hidden_layers_nb": config.get("hidden_layers_nb", 3),
			"hidden_size": config.get("hidden_size", 32),
		}

		self.policy_network: nn.Module = FCNet(config=policy_config).to(self.device)
		self.target_network: nn.Module = FCNet(config=policy_config).to(self.device)
		self.target_network.load_state_dict(self.policy_network.state_dict())

		self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.lr)
		self.loss_fn: nn.MSELoss = nn.MSELoss()

		self.rewards = []
		self.losses = []

	def train(self, max_steps: int) -> None:

		self.rewards = []
		self.losses = []

		steps = 0
		episode = 0
		steps_before_target_network_update = self.target_network_update_frequency

		print("==== STARTING TRAINING ====")

		while steps <= max_steps:
			self.rewards.append(0)
			self.losses.append(0)

			obs, infos = self.env.reset()
			done = False

			while not done:

				if np.random.random() <= self.eps:
					action = self.env.action_space.sample()
				else:
					with torch.no_grad():
						action = torch.argmax(self.policy_network(tensor(obs, device=self.device))).item()

				new_obs, reward, done, _, _ = self.env.step(action)
				self.replay_memory.push(obs, action, reward, new_obs, done)
				self.rewards[episode] += reward

				obs = new_obs

				if self.replay_memory.current_length >= self.batch_size:

					states, actions, rewards, next_states, are_terminals = self.replay_memory.sample(self.batch_size)

					states = torch.from_numpy(states).float().to(self.device)
					actions = torch.from_numpy(actions).long().to(self.device)
					rewards = torch.from_numpy(rewards).float().to(self.device)
					next_states = torch.from_numpy(next_states).float().to(self.device)
					are_terminals = torch.from_numpy(are_terminals).float().to(self.device)

					predictions = self.policy_network(states).gather(dim=1, index=actions.unsqueeze(1))

					with torch.no_grad():
						target = (rewards + self.gamma * self.target_network(next_states).max(1)[0] * (1 - are_terminals)).unsqueeze(1)

					self.optimizer.zero_grad()
					loss = self.loss_fn(predictions, target)
					self.losses.append(loss.item())

					loss.backward()
					self.optimizer.step()

					self.eps = max(self.eps * self.eps_decay, self.eps_min)

					if steps_before_target_network_update == 0:
						self.target_network.load_state_dict(self.policy_network.state_dict())
						steps_before_target_network_update = self.target_network_update_frequency

					steps_before_target_network_update -= 1

				steps += 1

			if episode % self.log_freq == 0:
				self.log_stats(episode=episode, avg_period=10)

			episode += 1

		print("==== TRAINING COMPLETE ====")
