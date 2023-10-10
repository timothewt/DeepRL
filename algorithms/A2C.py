from typing import Any, Tuple

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from torch import nn, tensor
from torch.distributions import Categorical, Normal

from algorithms.Algorithm import Algorithm
from models.ActorContinuous import ActorContinuous
from models.FCNet import FCNet


class Buffer:
	def __init__(
			self,
			num_envs: int,
			max_len: int = 5,
			state_shape: tuple[int] = (1,),
			device: torch.device = torch.device("cpu"),
	):
		self.states = torch.empty((max_len, num_envs) + state_shape, device=device)
		self.next_states = torch.empty((max_len, num_envs) + state_shape, device=device)
		self.dones = torch.empty((max_len, num_envs, 1), device=device)
		self.actions = torch.empty((max_len, num_envs, 1), device=device)
		self.rewards = torch.empty((max_len, num_envs, 1), device=device)
		self.values = torch.empty((max_len, num_envs, 1), device=device)
		self.log_probs = torch.empty((max_len, num_envs, 1), device=device)
		self.entropies = torch.empty((max_len, num_envs, 1), device=device)

		self.num_envs = num_envs
		self.max_len = max_len
		self.device = device
		self.i = 0

	def is_full(self) -> bool:
		return self.i == self.max_len

	def push(
			self,
			states: np.ndarray,
			next_states: np.ndarray,
			dones: bool,
			actions: int,
			rewards: float,
			values: float,
			log_probs: float,
			entropies: float
	) -> None:
		assert self.i < self.max_len, "Buffer is full!"

		self.states[self.i] = states
		self.next_states[self.i] = next_states
		self.dones[self.i] = dones
		self.actions[self.i] = actions
		self.rewards[self.i] = rewards
		self.values[self.i] = values
		self.log_probs[self.i] = log_probs
		self.entropies[self.i] = entropies

		self.i += 1

	def get(self, index: int = 0) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
		return self.states[index], \
			self.next_states[index], \
			self.dones[index], \
			self.actions[index], \
			self.rewards[index], \
			self.values[index], \
			self.log_probs[index], \
			self.entropies[index]

	def get_all(self) -> tuple[tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor]:
		return self.states, self.next_states, self.dones, self.actions, self.rewards, self.values, self.log_probs, self.entropies

	def reset(self) -> None:
		self.values = torch.empty((self.max_len, self.num_envs, 1), device=self.device)
		self.log_probs = torch.empty((self.max_len, self.num_envs, 1), device=self.device)
		self.entropies = torch.empty((self.max_len, self.num_envs, 1), device=self.device)
		self.i = 0


class A2C(Algorithm):

	def __init__(self, config: dict[str | Any]):
		"""
		:param config:
			num_envs: number of environments in parallel
			actor_lr: learning rate of the actor
			critic_lr: learning rate of the critic
			gamma: discount factor
			t_max: steps between each updates
		"""
		super().__init__(config=config)

		# Vectorized envs

		self.num_envs = max(config.get("num_envs", 1), 1)
		self.envs: gym.experimental.vector.VectorEnv = gym.make_vec(config.get("env_name", None), num_envs=self.num_envs)

		# Stats

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

		# Observations
		obs_space = self.envs.single_observation_space
		assert isinstance(obs_space, spaces.Box) or isinstance(obs_space, spaces.Discrete), \
			"Only Box and Discrete observation spaces currently supported"

		input_size = 1
		if isinstance(obs_space, spaces.Discrete):
			input_size = obs_space.n
		elif isinstance(obs_space, spaces.Box):
			input_size = int(np.prod(obs_space.shape))

		# Actions
		act_space = self.envs.single_action_space
		assert isinstance(act_space, spaces.Box) or isinstance(act_space, spaces.Discrete), \
			"Only Box and Discrete action spaces currently supported"

		output_size = 1
		output_function = None
		self.actions_type = "discrete"
		if isinstance(act_space, spaces.Discrete):
			output_size = act_space.n
			output_function = nn.Softmax(dim=-1)
		elif isinstance(act_space, spaces.Box):
			assert np.prod(act_space.shape) == 1, "Only single continuous action currently supported"
			output_size = 2  # network outputs mean and variance
			self.action_space_low, self.action_space_high = act_space.low[0], act_space.high[0]
			self.actions_type = "continuous"

		actor_config = {
			"input_size": input_size,
			"output_size": output_size,
			"hidden_layers_nb": config.get("actor_hidden_layers_nb", 3),
			"hidden_size": config.get("actor_hidden_size", 32),
			"output_function": output_function,
		}

		if self.actions_type == "continuous":
			self.actor: nn.Module = ActorContinuous(config=actor_config).to(self.device)
		else:
			self.actor: nn.Module = FCNet(config=actor_config).to(self.device)
		self.actor_optimizer: torch.optim.Optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

		critic_config = {
			"input_size": input_size,
			"output_size": 1,
			"hidden_layers_nb": config.get("critic_hidden_layers_nb", 3),
			"hidden_size": config.get("critic_hidden_size", 32),
		}
		self.critic: nn.Module = FCNet(config=critic_config).to(self.device)
		self.critic_optimizer: torch.optim.Optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

	def train(self, max_steps: int, plot_training_stats: bool = False) -> None:
		# From Algorithm S3 : https://arxiv.org/pdf/1602.01783v2.pdf

		self.rewards = []
		self.actor_losses = []
		self.critic_losses = []
		self.entropy = []

		steps = 0
		episode = 0

		buffer = Buffer(self.num_envs, self.t_max, self.env.observation_space.shape, self.device)

		print("==== STARTING TRAINING ====")

		obs, _ = self.envs.reset()
		obs = torch.from_numpy(obs).to(self.device)
		first_agent_rewards = 0

		while steps <= max_steps:
			actor_output = self.actor(obs)
			critic_output = self.critic(obs)  # value function

			if self.actions_type == "continuous":
				means, std = actor_output
				dist = Normal(loc=means, scale=std)
				actions = dist.sample()
				actions_to_input = self.scale_to_action_space(actions).cpu().numpy()
				log_probs = dist.log_prob(actions)
				entropies = dist.entropy()
			else:
				probs = actor_output
				dist = Categorical(probs=probs)
				actions = dist.sample()
				actions_to_input = actions.cpu().numpy()
				log_probs = dist.log_prob(actions).unsqueeze(1)
				actions = actions.unsqueeze(1)
				entropies = dist.entropy().unsqueeze(1)

			new_obs, rewards, dones, truncateds, _ = self.envs.step(actions_to_input)
			dones = dones + truncateds  # done or truncate
			new_obs = torch.from_numpy(new_obs).to(self.device)

			buffer.push(
				obs,
				new_obs,
				torch.from_numpy(dones).to(self.device).unsqueeze(1),
				actions,
				torch.from_numpy(rewards).to(self.device).unsqueeze(1),
				critic_output,
				log_probs,
				entropies,
			)

			obs = new_obs

			if buffer.is_full():
				self.update_networks(buffer)
				buffer.reset()

			first_agent_rewards += rewards[0]
			if dones[0]:
				self.rewards.append(first_agent_rewards)
				first_agent_rewards = 0
				if episode % self.log_freq == 0:
					self.log_rewards(episode=episode, avg_period=10)
				episode += 1
			steps += 1

		print("==== TRAINING COMPLETE ====")

		if plot_training_stats:
			self.plot_training_stats([
				("Reward", "Episode", self.rewards),
				("Actor loss", "Update step", self.actor_losses),
				("Critic loss", "Update step", self.critic_losses),
				("Entropy", "Update step", self.entropy),
			], n=max_steps // 1000)

	def update_networks(self, buffer: Buffer) -> None:

		states, next_states, dones, _, rewards, values, log_probs, entropies = buffer.get_all()

		# Computing advantages
		R = self.critic(next_states[-1])  # next_value
		returns = torch.zeros((buffer.max_len, self.num_envs, 1), device=self.device)

		for t in reversed(range(buffer.max_len)):
			R = rewards[t] + self.gamma * R * (1 - dones[t])
			returns[t] = R

		advantages = returns.detach() - values

		# Updating the network
		actor_loss = - (log_probs * advantages.detach()).mean()
		critic_loss = advantages.pow(2).mean()
		entropy_loss = entropies.mean()

		self.actor_optimizer.zero_grad()
		self.critic_optimizer.zero_grad()
		(actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy_loss).backward()
		self.actor_optimizer.step()
		self.critic_optimizer.step()

		self.actor_losses.append(actor_loss.item())
		self.critic_losses.append(critic_loss.item())
		self.entropy.append(entropies.mean().item())

	def scale_to_action_space(self, actions: tensor) -> tensor:
		actions = torch.clamp(actions, 0, 1)
		actions = actions * (self.action_space_high - self.action_space_low) + self.action_space_low
		return actions
