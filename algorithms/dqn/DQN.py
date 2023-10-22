from __future__ import annotations

from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch import tensor, nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from algorithms.Algorithm import Algorithm
from algorithms.dqn.ReplayMemory import ReplayMemory
from models import FCNet


class DQN(Algorithm):
	"""
	Deep Q Network
	Used for discrete action spaces only.
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

		# Environment

		self.env_fn = config.get("env_fn", None)
		assert self.env_fn is not None, "No environment function provided!"
		self.env: gym.Env = self.env_fn()
		assert isinstance(self.env, gym.Env), "Only gym.Env environments currently supported!"

		# Stats

		self.writer = None

		# Algorithm hyperparameters

		self.lr: float = config.get("lr", .0004)
		self.gamma: float = config.get("gamma", .99)
		self.eps: float = config.get("eps", 1)
		self.eps_decay: float = config.get("eps_decay", .9995)
		self.eps_min: float = config.get("eps_min", .01)
		self.batch_size: int = config.get("batch_size", 64)
		self.target_network_update_frequency: int = config.get("target_network_update_frequency", 100)
		self.max_replay_buffer_size = config.get("max_replay_buffer_size", 10_000)

		# Policies

		self.env_act_space = self.env.action_space
		assert isinstance(self.env_act_space, spaces.Discrete), "Only discrete action spaces are supported"

		self.env_obs_space = self.env.observation_space
		self.env_flat_obs_space = spaces.flatten_space(self.env_obs_space)

		self.hidden_layers_nb = config.get("hidden_layers_nb", 2)
		self.hidden_size = config.get("hidden_size", 64)
		policy_config = {
			"input_size": np.prod(self.env_flat_obs_space.shape),
			"output_size": self.env_act_space.n,
			"hidden_layers_nb": self.hidden_layers_nb,
			"hidden_size": self.hidden_size,
		}

		self.policy: nn.Module = FCNet(config=policy_config).to(self.device)
		self.policy_target: nn.Module = FCNet(config=policy_config).to(self.device)
		self.policy_target.load_state_dict(self.policy.state_dict())

		self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
		self.loss_fn: nn.MSELoss = nn.MSELoss()

	def train(self, max_steps: int) -> None:
		"""
		Trains the algorithm on the chosen environment
		:param max_steps: maximum number of steps for the whole training process
		"""
		self.writer = SummaryWriter(
			f"runs/DQN-{self.env.metadata.get('name', 'env_')}-{datetime.now().strftime('%d-%m-%y_%Hh%Mm%S')}"
		)
		self.writer.add_text(
			"Hyperparameters/hyperparameters",
			self.dict2mdtable({
				"lr": self.lr,
				"gamma": self.gamma,
				"eps": self.eps,
				"eps_decay": self.eps_decay,
				"eps_min": self.eps_min,
				"batch_size": self.batch_size,
				"target_network_update_frequency": self.target_network_update_frequency,
				"max_replay_buffer_size": self.max_replay_buffer_size,
			})
		)
		self.writer.add_text(
			"Hyperparameters/FC Networks configuration",
			self.dict2mdtable({
				"hidden_layers_nb": self.hidden_layers_nb,
				"hidden_size": self.hidden_size,
			})
		)

		episode = 0
		step = 0
		steps_before_target_network_update = self.target_network_update_frequency
		replay_memory = ReplayMemory(self.max_replay_buffer_size, self.env_flat_obs_space.shape)

		pbar = tqdm(desc="DQN Training", total=max_steps)
		print("==== STARTING TRAINING ====")

		while step <= max_steps:
			obs, _ = self.env.reset()
			obs = self._flatten_obs(obs)
			done = False
			episode_rewards = 0

			while not done:

				if np.random.random() <= self.eps:
					action = self.env.action_space.sample()
				else:
					with torch.no_grad():
						action = torch.argmax(
							self.policy(tensor(obs, device=self.device))
						).item()

				new_obs, reward, terminated, truncated, _ = self.env.step(action)
				new_obs = self._flatten_obs(new_obs)
				done = terminated or truncated
				replay_memory.push(obs, action, reward, new_obs, done)
				obs = new_obs
				episode_rewards += reward
				step += 1
				pbar.update(1)

				if replay_memory.current_length < self.batch_size:
					continue

				self._update_networks(replay_memory, step)

				if steps_before_target_network_update == 0:
					self.policy_target.load_state_dict(self.policy.state_dict())
					steps_before_target_network_update = self.target_network_update_frequency

				steps_before_target_network_update -= 1

			self.writer.add_scalar("Stats/Rewards", episode_rewards, episode)
			episode += 1

		print("==== TRAINING COMPLETE ====")

	def _update_networks(self, replay_memory: ReplayMemory, step: int) -> None:
		"""
		Updates the policy using the experience replay buffer
		:replay_memory: replay memory buffer from which we sample experiences
		:param step: current step
		"""
		states, actions, rewards, next_states, are_terminals = replay_memory.sample(self.batch_size)

		states = torch.from_numpy(states).float().to(self.device)
		actions = torch.from_numpy(actions).long().to(self.device)
		rewards = torch.from_numpy(rewards).float().to(self.device)
		next_states = torch.from_numpy(next_states).float().to(self.device)
		are_terminals = torch.from_numpy(are_terminals).float().to(self.device)

		predictions = self.policy(states).gather(dim=1, index=actions.unsqueeze(1))

		with torch.no_grad():
			target = (
					rewards + self.gamma * self.policy_target(next_states).max(1)[0] * (1 - are_terminals)
			).unsqueeze(1)

		self.optimizer.zero_grad()
		loss = (predictions - target).pow(2).mean()
		loss.backward()
		self.optimizer.step()

		self.eps = max(self.eps * self.eps_decay, self.eps_min)

		self.writer.add_scalar("Stats/Loss", loss, step)
		self.writer.add_scalar("Stats/Epsilon", self.eps, step)
