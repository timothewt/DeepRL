from typing import Any

import numpy as np
import torch
from gymnasium.vector.utils import spaces
from torch import nn

from algorithms.Algorithm import Algorithm
from models.FCNet import FCNet


class A2C(Algorithm):

	def __init__(self, config: dict[str | Any]):
		"""
		:param config:
			device: device used by PyTorch
			env: environment instance
			log_freq: episodes interval for logging the current stats of the algorithm
			actor_lr: learning rate of the actor
			critic_lr: learning rate of the critic
			gamma: discount factor
		"""
		super().__init__(config=config)

		# Algorithm hyperparameters

		self.actor_lr: float = config.get("actor_lr", .0002)
		self.critic_lr: float = config.get("critic_lr", .0002)
		self.gamma: float = config.get("gamma", .99)

		# Policy

		assert isinstance(self.env.observation_space, spaces.Box) or \
			isinstance(self.env.observation_space, spaces.Discrete),\
			"Only Box and Discrete spaces currently supported"

		input_size = 0
		if isinstance(self.env.observation_space, spaces.Box):
			input_size = int(np.prod(self.env.observation_space.shape))
		elif isinstance(self.env.observation_space, spaces.Discrete):
			input_size = self.env.observation_space.n

		actor_config = {
			"input_size": input_size,
			"output_size": self.env.action_space.n,
			"hidden_layers_nb": config.get("actor_hidden_layers_nb", 3),
			"hidden_size": config.get("actor_hidden_size", 32),
			"output_function": nn.Softmax(dim=-1),
		}
		self.actor: nn.Module = FCNet(config=actor_config).to(self.device)
		critic_config = {
			"input_size": input_size,
			"output_size": self.env.action_space.n,
			"hidden_layers_nb": config.get("critic_hidden_layers_nb", 3),
			"hidden_size": config.get("critic_hidden_size", 32),
		}
		self.critic: nn.Module = FCNet(config=critic_config).to(self.device)

		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

	def train(self, max_steps: int) -> None:

		self.rewards = []
		self.losses = []

		steps = 0
		episode = 0

		print("==== STARTING TRAINING ====")

		while steps <= max_steps:
			self.rewards.append(0)
			self.losses.append(0)

			current_episode_step = 0
			done = False
			obs, _ = self.env.reset()

			while not done and current_episode_step <= self.max_episode_steps:
				action = self.env.action_space.sample()
				new_obs, reward, done, _, _ = self.env.step(action)
				obs = new_obs

				current_episode_step += 1

			if episode % self.log_freq == 0:
				self.log_stats(episode=episode, avg_period=10)

			episode += 1
			steps += current_episode_step

		print("==== TRAINING COMPLETE ====")
