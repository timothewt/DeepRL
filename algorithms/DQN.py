from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch import tensor, nn as nn
from torch.utils.tensorboard import SummaryWriter

from algorithms.Algorithm import Algorithm
from models.FCNet import FCNet


class ReplayMemory:
	"""
	Replay memory storing all the previous steps data, up to a certain length, then replaces the oldest data
	"""
	def __init__(self, max_length: int = 10_000, state_shape: tuple[int] = (1,)):
		"""
		Initializes the memory
		:param max_length: maximum length of the memory after which the oldest values will be replaced
		:param state_shape: shape of the state
		"""
		self.states = np.empty((max_length,) + state_shape, dtype=float)
		self.actions = np.empty(max_length, dtype=int)
		self.rewards = np.empty(max_length, dtype=float)
		self.next_state = np.empty((max_length,) + state_shape, dtype=float)
		self.is_terminal = np.empty(max_length, dtype=bool)

		self.max_length = max_length
		self.current_length = 0
		self.i = 0

	def push(self, state, action, reward, next_state, is_terminal) -> None:
		"""
		Pushes new values in the memory
		:param state: state before the action
		:param action: action taken by the agent
		:param reward: reward given for this action at that state
		:param next_state: next state after doing the action
		:param is_terminal: tells if the action done has terminated the agent
		"""
		self.i %= self.max_length
		self.current_length = min(self.current_length + 1, self.max_length)

		self.states[self.i] = state
		self.actions[self.i] = action
		self.rewards[self.i] = reward
		self.next_state[self.i] = next_state
		self.is_terminal[self.i] = is_terminal
		self.i += 1

	def sample(self, batch_size: int) -> tuple[tensor, tensor, tensor, tensor, tensor]:
		"""
		Samples a random batch in all the replay memory
		:param batch_size: size of the returned batch
		:return: a random batch in the memory
		"""
		indices = np.random.choice(self.current_length, size=batch_size)
		return self.states[indices], \
			self.actions[indices], \
			self.rewards[indices], \
			self.next_state[indices], \
			self.is_terminal[indices]


class DQN(Algorithm):
	"""
	Deep Q Network algorithm
	"""

	def __init__(self, config: dict):
		"""
		:param config:
			env_fn (Callable[[], gymnasium.Env]): function returning a Gymnasium environment
			device (torch.device): device used (cpu, gpu)

			lr (float): learning rate of the agent
			gamma (float): discount factor
			eps (float): epsilon value of the epsilon greedy strategy
			eps_decay (float): decay of the epsilon parameter at each update step
			eps_min (float): minimum value of the epsilon parameter
			batch_size (float): size of the batches used to update the policy
			target_network_update_frequency (int): steps interval before the target network gets replaced by the policy
			network
			max_replay_buffer_size (int): maximum size of the replay buffer

			hidden_layers_nb (int): number of hidden linear layers in the policy network
			hidden_size (int): size of the hidden linear layers
		"""
		super().__init__(config)

		# Device

		self.device = config.get("device", torch.device("cpu"))

		# Stats

		self.writer = None

		# Environment

		self.env_fn = config.get("env_fn", None)
		assert self.env_fn is not None, "No environment function provided!"
		self.env: gym.Env = self.env_fn()

		# Algorithm hyperparameters

		self.lr: float = config.get("lr", .0004)
		self.gamma: float = config.get("gamma", .99)
		self.eps: float = config.get("eps", 1)
		self.eps_decay: float = config.get("eps_decay", .9995)
		self.eps_min: float = config.get("eps_min", .01)
		self.batch_size: int = config.get("batch_size", 64)
		self.target_network_update_frequency: int = config.get("target_network_update_frequency", 100)

		self.replay_memory = ReplayMemory(config.get("max_replay_buffer_size", 10_000), self.env.observation_space.shape)

		# Policies

		assert isinstance(self.env.action_space, spaces.Discrete), "Only discrete action spaces are supported"
		assert isinstance(self.env.observation_space, spaces.Box) or \
			isinstance(self.env.observation_space, spaces.Discrete),\
			"Only Box and Discrete spaces currently supported"

		if isinstance(self.env.observation_space, spaces.Box):
			input_size = int(np.prod(self.env.observation_space.shape))
		else:
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

	def train(self, max_steps: int) -> None:
		"""
		Trains the algorithm on the chosen environment
		:param max_steps: maximum number of steps that can be done
		"""
		self.writer = SummaryWriter()

		steps = 0
		episode = 0
		steps_before_target_network_update = self.target_network_update_frequency

		print("==== STARTING TRAINING ====")

		while steps <= max_steps:

			obs, infos = self.env.reset()
			done = False
			episode_rewards = 0

			while not done:

				if np.random.random() <= self.eps:
					action = self.env.action_space.sample()
				else:
					with torch.no_grad():
						action = torch.argmax(self.policy_network(tensor(obs, device=self.device))).item()

				new_obs, reward, done, _, _ = self.env.step(action)
				self.replay_memory.push(obs, action, reward, new_obs, done)
				obs = new_obs
				episode_rewards += reward
				steps += 1

				if self.replay_memory.current_length < self.batch_size:
					continue

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
				loss.backward()
				self.optimizer.step()

				self.writer.add_scalar("Stats/Loss", loss, episode)
				self.writer.add_scalar("Stats/Epsilon", self.eps, episode)

				self.eps = max(self.eps * self.eps_decay, self.eps_min)

				if steps_before_target_network_update == 0:
					self.target_network.load_state_dict(self.policy_network.state_dict())
					steps_before_target_network_update = self.target_network_update_frequency

				steps_before_target_network_update -= 1

			self.writer.add_scalar("Stats/Reward", episode_rewards, episode)
			episode += 1

		print("==== TRAINING COMPLETE ====")
