from typing import Any

import numpy as np
import torch
from gymnasium.vector.utils import spaces
from torch import nn, tensor
from torch.distributions import Categorical

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
			t_max: steps between each updates
		"""
		super().__init__(config=config)
		self.actor_losses = []
		self.critic_losses = []
		self.entropy = []

		# Algorithm hyperparameters

		self.actor_lr: float = config.get("actor_lr", .0002)
		self.critic_lr: float = config.get("critic_lr", .0002)
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
			"output_function": nn.Softmax(dim=-1),
		}
		self.actor: nn.Module = FCNet(config=actor_config).to(self.device)
		critic_config = {
			"input_size": input_size,
			"output_size": 1,
			"hidden_layers_nb": config.get("critic_hidden_layers_nb", 3),
			"hidden_size": config.get("critic_hidden_size", 32),
		}
		self.critic: nn.Module = FCNet(config=critic_config).to(self.device)

		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

	def train(self, max_steps: int, plot_training_stats: bool = False) -> None:
		# From Algorithm S3 : https://arxiv.org/pdf/1602.01783v2.pdf

		self.rewards = []
		self.actor_losses = []
		self.critic_losses = []
		self.entropy = []

		steps = 0
		episode = 0

		states = torch.empty((self.t_max,) + self.env.observation_space.shape, device=self.device, dtype=torch.float)
		actions = torch.empty((self.t_max,), device=self.device, dtype=torch.float)
		rewards = torch.empty((self.t_max,), device=self.device, dtype=torch.float)
		log_probs = torch.empty((self.t_max,), device=self.device, dtype=torch.float)
		entropy = torch.empty((self.t_max,), device=self.device, dtype=torch.float)

		steps_after_update = 0

		print("==== STARTING TRAINING ====")

		while steps <= max_steps:

			current_episode_step = 0
			ep_rewards = 0

			done = truncated = False
			new_obs, _ = self.env.reset()
			obs = torch.from_numpy(new_obs).to(self.device).float()

			# Collecting data for an episode

			while not done and not truncated:
				probs = self.actor(obs)
				dist = Categorical(probs=probs)
				action = dist.sample()
				log_probs[steps_after_update] = dist.log_prob(action)

				new_obs, reward, done, truncated, _ = self.env.step(action.item())

				states[steps_after_update] = obs
				actions[steps_after_update] = action
				rewards[steps_after_update] = reward
				entropy[steps_after_update] = dist.entropy()
				ep_rewards += reward

				obs = torch.from_numpy(new_obs).to(self.device).float()

				current_episode_step += 1
				steps_after_update += 1

				if steps_after_update == self.t_max or done or truncated:
					self.update_networks(states, log_probs, entropy, rewards, done or truncated, steps_after_update - 1)
					log_probs = torch.empty((self.t_max,), device=self.device, dtype=torch.float)
					entropy = torch.empty((self.t_max,), device=self.device, dtype=torch.float)
					steps_after_update = 0

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
		states: tensor,
		log_probs: tensor,
		entropy: tensor,
		rewards: tensor,
		is_terminal: bool = False,
		terminal_state_index: int = -1
	) -> None:

		T = self.t_max if not is_terminal else terminal_state_index + 1

		returns = torch.zeros((T, 1), device=self.device)
		returns[-1] = rewards[T - 1]
		if is_terminal:
			returns[-1] += self.gamma * self.critic(states[terminal_state_index])

		for t in range(T - 2, -1, -1):
			returns[t] = rewards[t] + self.gamma * returns[t + 1]

		V = self.critic(states[:T])
		R_V = (returns - V).squeeze(1)

		# Updating the network
		actor_loss = - (log_probs[:T] * R_V.detach()).mean()
		critic_loss = self.vf_coef * torch.pow(R_V, 2).mean()
		entropy_loss = (self.ent_coef * entropy).mean()

		self.actor_optimizer.zero_grad()
		self.critic_optimizer.zero_grad()
		(actor_loss + critic_loss - entropy_loss).backward()
		self.actor_optimizer.step()
		self.critic_optimizer.step()

		self.actor_losses.append(actor_loss.item())
		self.critic_losses.append(critic_loss.item())
		self.entropy.append(entropy[:T].mean().item())
