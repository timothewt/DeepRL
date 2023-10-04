from typing import Any

import numpy as np
import torch
from gymnasium.vector.utils import spaces
from torch import nn, tensor

from algorithms.Algorithm import Algorithm
from models.ActorCritic import ActorCritic


class Buffer:
	def __init__(self, max_len: int = 5, state_shape: tuple[int] = (1,), device: torch.device = torch.device("cpu")):
		self.states = torch.empty((max_len,) + state_shape, device=device, dtype=torch.float)
		self.dones = torch.empty((max_len,), device=device, dtype=torch.float)
		self.actions = torch.empty((max_len,), device=device, dtype=torch.float)
		self.rewards = torch.empty((max_len,), device=device, dtype=torch.float)
		self.values = torch.empty((max_len,), device=device, dtype=torch.float)
		self.log_probs = torch.empty((max_len,), device=device, dtype=torch.float)
		self.entropies = torch.empty((max_len,), device=device, dtype=torch.float)

		self.max_len = max_len
		self.device = device
		self.i = 0

	def is_full(self) -> bool:
		return self.i == self.max_len

	def push(self, state: np.ndarray, done: bool, action: int, reward: float, value: float, log_prob: float, entropy: float) -> None:
		assert self.i < self.max_len, "Buffer is full!"

		self.states[self.i] = state
		self.dones[self.i] = done
		self.actions[self.i] = action
		self.rewards[self.i] = reward
		self.values[self.i] = value
		self.log_probs[self.i] = log_prob
		self.entropies[self.i] = entropy

		self.i += 1

	def get(self, index: int = 0) -> tuple[tensor, tensor, tensor, tensor, tensor, tensor, tensor]:
		return self.states[index],\
			self.dones[index],\
			self.actions[index],\
			self.rewards[index],\
			self.values[index],\
			self.log_probs[index],\
			self.entropies[index]

	def get_all(self) -> tuple[tensor, tensor, tensor, tensor, tensor, tensor, tensor]:
		return self.states, self.dones, self.actions, self.rewards, self.values, self.log_probs, self.entropies

	def reset(self) -> None:
		self.values = torch.empty((self.max_len,), device=self.device, dtype=torch.float)
		self.log_probs = torch.empty((self.max_len,), device=self.device, dtype=torch.float)
		self.entropies = torch.empty((self.max_len,), device=self.device, dtype=torch.float)
		self.i = 0


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
			t_max: steps between each updates
		"""
		super().__init__(config=config)
		self.actor_losses = []
		self.critic_losses = []
		self.entropy = []

		# Algorithm hyperparameters

		self.lr: float = config.get("lr", .0002)
		self.gamma: float = config.get("gamma", .99)
		self.t_max: int = config.get("t_max", 5)
		self.ent_coef: float = config.get("ent_coef", .001)
		self.vf_coef = config.get("vf_coef", .5)

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
		}
		self.network: nn.Module = ActorCritic(config=actor_config).to(self.device)

		self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

	def train(self, max_steps: int, plot_training_stats: bool = False) -> None:
		# From Algorithm S3 : https://arxiv.org/pdf/1602.01783v2.pdf

		self.rewards = []
		self.actor_losses = []
		self.critic_losses = []
		self.entropy = []

		steps = 0
		episode = 0

		buffer = Buffer(self.t_max, self.env.observation_space.shape, self.device)

		print("==== STARTING TRAINING ====")

		while steps <= max_steps:

			current_episode_step = 0
			ep_rewards = 0

			done = truncated = False
			new_obs, _ = self.env.reset()
			obs = torch.from_numpy(new_obs).to(self.device).float()

			# Collecting data for an episode

			while not (done or truncated):
				dist, value = self.network(obs)
				action = dist.sample()

				new_obs, reward, done, truncated, _ = self.env.step(action.item())

				ep_rewards += reward
				obs = torch.from_numpy(new_obs).to(self.device).float()
				current_episode_step += 1

				buffer.push(obs, done or truncated, action, reward, value, dist.log_prob(action), dist.entropy())

				if buffer.is_full():
					self.update_networks(buffer)
					buffer.reset()

			self.rewards.append(ep_rewards)

			if episode % self.log_freq == 0:
				self.log_stats(episode=episode, avg_period=10)

			episode += 1
			steps += current_episode_step

		print("==== TRAINING COMPLETE ====")

		if plot_training_stats:
			self.plot_training_stats([
				("Reward", "Episode", self.rewards),
				("Actor loss", "Update step", self.actor_losses),
				("Critic loss", "Update step", self.critic_losses),
				("Entropy", "Update step", self.entropy),
			])

	def update_networks(
		self,
		buffer: Buffer,
	) -> None:

		_, dones, _, rewards, values, log_probs, entropies = buffer.get_all()

		# Computing advantages
		next_value = values[-1]
		R = torch.zeros((buffer.max_len, 1), device=self.device)

		for t in reversed(range(buffer.max_len)):
			R[t] = rewards[t] + self.gamma * (1 - dones[t]) * next_value
			next_value = values[t]

		advantages = R - values

		# Updating the network
		actor_loss = - (buffer.log_probs * advantages.detach()).mean()
		critic_loss = torch.pow(advantages, 2).mean()
		entropy_loss = buffer.entropies.mean()

		self.optimizer.zero_grad()
		(actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy_loss).backward()
		self.optimizer.step()

		self.actor_losses.append(actor_loss.item())
		self.critic_losses.append(critic_loss.item())
		self.entropy.append(buffer.entropies.mean().item())