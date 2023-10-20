from datetime import datetime
from typing import Any

import gymnasium as gym
import numpy as np
import supersuit as ss
import torch
from gymnasium import spaces
from gymnasium.vector.async_vector_env import AsyncVectorEnv
from pettingzoo import ParallelEnv
from torch import nn, tensor
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from algorithms.Algorithm import Algorithm
from models import FCNet


class Buffer:
	"""
	Memory buffer used for PPO
	"""
	def __init__(
			self,
			num_envs: int,
			max_len: int = 5,
			state_shape: tuple[int] = (1,),
			actions_nb: int = 1,
			device: torch.device = torch.device("cpu"),
	):
		"""
		Initialization of the buffer
		:param num_envs: number of parallel environments
		:param max_len: maximum length of the buffer, typically the PPO horizon parameter
		:param state_shape: shape of the state given to the policy
		:param actions_nb: number of possible actions
		:param device: device used by PyTorch
		"""
		self.states = torch.empty((max_len, num_envs, np.prod(state_shape)), device=device)
		self.next_states = torch.empty((max_len, num_envs, np.prod(state_shape)), device=device)
		self.dones = torch.empty((max_len, num_envs, 1), device=device)
		self.actions = torch.empty((max_len, num_envs, actions_nb), device=device)
		self.rewards = torch.empty((max_len, num_envs, 1), device=device)
		self.values = torch.empty((max_len, num_envs, 1), device=device)
		self.log_probs = torch.empty((max_len, num_envs, actions_nb), device=device)

		self.num_envs = num_envs
		self.state_shape = state_shape
		self.max_len = max_len
		self.device = device
		self.i = 0

	def is_full(self) -> bool:
		"""
		Checks if the buffer is full
		:return: True if the buffer is full False otherwise
		"""
		return self.i == self.max_len

	def push(
			self,
			states: tensor,
			next_states: tensor,
			dones: tensor,
			actions: tensor,
			rewards: tensor,
			values: tensor,
			log_probs: tensor,
	) -> None:
		"""
		Pushes new values in the buffer of shape (num_env, data_shape)
		:param states: states of each environment
		:param next_states: next states after this step
		:param dones: if the step led to a termination
		:param actions: actions made by the agents
		:param rewards: rewards given for this action
		:param values: critic policy value
		:param log_probs: log probability of the actions
		"""
		assert self.i < self.max_len, "Buffer is full!"

		self.states[self.i] = states
		self.next_states[self.i] = next_states
		self.dones[self.i] = dones
		self.actions[self.i] = actions
		self.rewards[self.i] = rewards
		self.values[self.i] = values
		self.log_probs[self.i] = log_probs

		self.i += 1

	def get_all(self) -> tuple[tensor, tensor, tensor, tensor, tensor, tensor, tensor]:
		"""
		Gives all the values of the buffer
		:return: all buffer tensors
		"""
		return self.states, self.next_states, self.dones, self.actions, self.rewards, self.values, self.log_probs

	def get_all_flattened(self) -> tuple[tensor, tensor, tensor, tensor, tensor, tensor, tensor]:
		"""
		Gives all the buffer values as flattened tensors
		:return: all buffer tensors flattened
		"""
		return self.states.view((self.max_len * self.num_envs, np.prod(self.state_shape))), \
			self.next_states.view((self.max_len * self.num_envs, np.prod(self.state_shape))), \
			self.dones.flatten(), \
			self.actions.flatten(end_dim=1), \
			self.rewards.flatten(), \
			self.values.flatten(), \
			self.log_probs.flatten(end_dim=1),

	def reset(self) -> None:
		"""
		Resets the iteration variable
		"""
		self.i = 0


class PPO(Algorithm):
	"""
	Proximal Policy Optimization
	Used for discrete action spaces without action mask.
	For continuous action spaces, use PPOContinuous, and for masked	discrete action spaces use PPOMasked.
	Environment can either be a gymnasium.Env or a pettingzoo.ParallelEnv.
	"""
	def __init__(self, config: dict[str: Any]):
		"""
		:param config:
			env_fn (Callable[[], gymnasium.Env]): function returning a Gymnasium environment
			num_envs (int): number of environments in parallel
			device (torch.device): device used (cpu, gpu)

			actor_lr (float): learning rate of the actor
			critic_lr (float): learning rate of the critic
			gamma (float): discount factor
			gae_lambda (float): GAE parameter
			horizon (int): steps number between each update
			num_epochs (int): number of epochs during the policy updates
			ent_coef (float): entropy bonus coefficient
			vf_coef (float): value function loss coefficient
			eps (float): epsilon clip value
			minibatch_size (float): size of the mini-batches used to update the policy
			use_grad_clip (bool): boolean telling if gradient clipping is used
			grad_clip (float): value at which the gradients will be clipped

			actor_hidden_layers_nb (int): number of hidden linear layers in the actor network
			actor_hidden_size (int): size of the hidden linear layers in the actor network
			critic_hidden_layers_nb (int): number of hidden linear layers in the critic network
			critic_hidden_size (int): size of the hidden linear layers in the actor network
		"""
		super().__init__(config=config)

		# Device

		self.device = config.get("device", torch.device("cpu"))

		# Vectorized envs

		self.env_fn = config.get("env_fn", None)
		assert self.env_fn is not None, "No environment function provided!"
		self.env: gym.Env | ParallelEnv = self.env_fn()
		assert isinstance(self.env, gym.Env) or isinstance(self.env, ParallelEnv), \
			"Only gymnasium.Env and pettingzoo.ParallelEnv are currently supported."
		self.is_multi_agents = isinstance(self.env, ParallelEnv)
		self.num_envs = max(config.get("num_envs", 1), 1)
		self.num_agents = 1
		if self.is_multi_agents:
			# pad observations of done agents
			self.num_agents = len(self.env.possible_agents)
			self.envs: ss.ConcatVecEnv = ss.concat_vec_envs_v1(
				ss.pettingzoo_env_to_vec_env_v1(self.env),
				self.num_envs
			)
			self.env_act_space = self.envs.action_space
		else:
			self.envs: AsyncVectorEnv = AsyncVectorEnv([self.env_fn for _ in range(self.num_envs)])
			self.env_act_space = self.envs.single_action_space
		assert isinstance(self.env_act_space, spaces.Discrete), \
			"Only discrete action spaces are supported. For continuous spaces, see PPOContinuous"

		# Stats

		self.writer = None

		# Algorithm hyperparameters

		self.actor_lr: float = config.get("actor_lr", .0001)
		self.critic_lr: float = config.get("critic_lr", .0005)
		self.gamma: float = config.get("gamma", .99)
		self.gae_lambda: float = config.get("gae_lambda", .95)
		self.horizon: int = config.get("horizon", 5)
		self.num_epochs: int = config.get("num_epochs", 5)
		self.ent_coef: float = config.get("ent_coef", .01)
		self.vf_coef = config.get("vf_coef", .5)
		self.eps = config.get("eps", .2)
		self.use_grad_clip = config.get("use_grad_clip", False)
		self.grad_clip = config.get("grad_clip", .5)

		self.batch_size = self.horizon * self.num_envs * self.num_agents
		self.minibatch_size = config.get("minibatch_size", self.batch_size)
		assert self.horizon % self.minibatch_size == 0, \
			"Horizon size must be a multiple of mini-batch size!"
		self.minibatch_nb_per_batch = self.batch_size // self.minibatch_size

		# Policies

		if self.is_multi_agents:
			self.env_obs_space = self.envs.observation_space
		else:
			self.env_obs_space = self.envs.single_observation_space
		self.env_flat_obs_space = gym.spaces.utils.flatten_space(self.env_obs_space)
		self.actions_nb = 1

		self.actor_hidden_layers_nb = config.get("actor_hidden_layers_nb", 3)
		self.actor_hidden_size = config.get("actor_hidden_size", 64)
		self.critic_hidden_layers_nb = config.get("critic_hidden_layers_nb", 3)
		self.critic_hidden_size = config.get("critic_hidden_size", 64)

		actor_config = {
			"input_size": np.prod(self.env_flat_obs_space.shape),
			"hidden_layers_nb": self.actor_hidden_layers_nb, "hidden_size": self.actor_hidden_size,
			"output_layer_std": .01, "output_size": self.env_act_space.n,
			"output_function": nn.Softmax(dim=-1)
		}

		self.actor: nn.Module = FCNet(config=actor_config).to(self.device)

		self.actor_optimizer: torch.optim.Optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

		critic_config = {
			"input_size": int(np.prod(self.env_flat_obs_space.shape)),
			"output_size": 1,
			"hidden_layers_nb": self.critic_hidden_layers_nb,
			"hidden_size": self.critic_hidden_size,
			"output_layer_std": 1,
		}
		self.critic: nn.Module = FCNet(config=critic_config).to(self.device)
		self.critic_optimizer: torch.optim.Optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

		self.mse = nn.MSELoss()

	def train(self, max_steps: int) -> None:
		"""
		Trains the algorithm on the chosen environment
		From https://arxiv.org/pdf/1707.06347.pdf and https://arxiv.org/pdf/2205.09123.pdf
		:param max_steps: maximum number of steps for the whole training process
		"""
		self.writer = SummaryWriter(
			f"runs/{self.env.metadata.get('name', 'env_')}-{datetime.now().strftime('%d-%m-%y_%Hh%Mm%S')}"
		)
		self.writer.add_text(
			"Hyperparameters/hyperparameters",
			self.dict2mdtable({
				"num_envs": self.num_envs,
				"actor_lr": self.actor_lr,
				"critic_lr": self.critic_lr,
				"gamma": self.gamma,
				"gae_lambda": self.gae_lambda,
				"horizon": self.horizon,
				"num_epochs": self.num_epochs,
				"ent_coef": self.ent_coef,
				"vf_coef": self.vf_coef,
				"eps": self.eps,
				"minibatch_size": self.minibatch_size,
				"use_grad_clip": self.use_grad_clip,
				"grad_clip": self.grad_clip,
			})
		)
		self.writer.add_text(
			"Hyperparameters/FC Networks configuration",
			self.dict2mdtable({
				"actor_hidden_layers_nb": self.actor_hidden_layers_nb,
				"actor_hidden_size": self.actor_hidden_size,
				"critic_hidden_layers_nb": self.critic_hidden_layers_nb,
				"critic_hidden_size": self.critic_hidden_size,
			})
		)

		episode = 0

		buffer = Buffer(self.num_envs * self.num_agents, self.horizon, self.env_flat_obs_space.shape, self.actions_nb, self.device)

		print("==== STARTING TRAINING ====")

		obs, infos = self.envs.reset()
		obs = torch.from_numpy(self._flatten_obs(obs)).float().to(self.device)
		first_agent_rewards = 0

		for _ in tqdm(range(max_steps), desc="PPO Training"):
			critic_output = self.critic(obs)  # value function

			probs = self.actor(obs)
			dist = Categorical(probs=probs)
			actions = dist.sample()
			actions_to_input = actions.cpu().numpy()
			log_probs = dist.log_prob(actions).unsqueeze(1)
			actions = actions.unsqueeze(1)

			new_obs, rewards, dones, truncateds, new_infos = self.envs.step(actions_to_input)
			dones = dones + truncateds  # done or truncate
			new_obs = torch.from_numpy(self._flatten_obs(new_obs)).float().to(self.device)

			buffer.push(
				obs,
				new_obs,
				torch.from_numpy(dones).float().to(self.device).unsqueeze(1),
				actions,
				torch.from_numpy(rewards).float().to(self.device).unsqueeze(1),
				critic_output,
				log_probs,
			)

			obs = new_obs

			if buffer.is_full():
				self._update_networks(buffer)
				buffer.reset()

			first_agent_rewards += rewards[0]
			if dones[0]:
				self.writer.add_scalar("Rewards", first_agent_rewards, episode)
				first_agent_rewards = 0
				episode += 1

		print("==== TRAINING COMPLETE ====")

	def _update_networks(self, buffer: Buffer) -> None:
		"""
		Updates the actor and critic networks according to the PPO paper
		:param buffer: complete buffer of experiences
		"""
		states, _, _, actions, rewards, values, old_log_probs = buffer.get_all_flattened()
		values, old_log_probs = values.detach().view(self.batch_size, 1), old_log_probs.detach()
		actions = actions.view(self.batch_size)

		advantages = self._compute_advantages(buffer, self.gamma, self.gae_lambda).flatten(end_dim=1)
		advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
		returns = advantages + values

		for _ in range(self.num_epochs):
			indices = torch.randperm(self.batch_size)

			for m in range(self.minibatch_nb_per_batch):
				start = m * self.minibatch_size
				end = start + self.minibatch_size
				minibatch_indices = indices[start:end]

				probs = self.actor(states[minibatch_indices])
				new_dist = Categorical(probs=probs)

				new_log_probs = new_dist.log_prob(actions[minibatch_indices]).view(self.minibatch_size, self.actions_nb)
				new_entropy = new_dist.entropy()
				new_values = self.critic(states[minibatch_indices])

				r = torch.exp(new_log_probs - old_log_probs[minibatch_indices])  # policy ratio
				L_clip = torch.min(
					r * advantages[minibatch_indices],
					torch.clamp(r, 1 - self.eps, 1 + self.eps) * advantages[minibatch_indices]
				).mean()
				L_vf = self.mse(new_values, returns[minibatch_indices])
				L_S = new_entropy.mean()

				# Updating the network
				self.actor_optimizer.zero_grad()
				self.critic_optimizer.zero_grad()
				(- L_clip + self.vf_coef * L_vf - self.ent_coef * L_S).backward()
				if self.use_grad_clip:
					nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
					nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
				self.actor_optimizer.step()
				self.critic_optimizer.step()

				self.writer.add_scalar("Loss/Actor_Loss", L_clip.item())
				self.writer.add_scalar("Loss/Critic_Loss", L_vf.item())
				self.writer.add_scalar("Loss/Entropy", L_S.item())

	def _compute_advantages(self, buffer: Buffer, gamma: float, gae_lambda: float) -> tensor:
		"""
		Computes the advantages for all steps of the buffer
		:param buffer: complete buffer of experiences
		:param gamma: rewards discount rate
		:param gae_lambda: lambda parameter of the GAE
		:return: the advantages for each timestep as a tensor
		"""
		_, next_states, dones, _, rewards, values, _ = buffer.get_all()

		next_values = values.roll(-1, dims=0)
		next_values[-1] = self.critic(next_states[-1])

		deltas = (rewards + gamma * next_values - values).detach()

		advantages = torch.zeros(deltas.shape, device=self.device)
		last_advantage = advantages[-1]
		next_step_terminates = dones[-1]  # should be the dones of the next step however cannot reach it
		for t in reversed(range(buffer.max_len)):
			advantages[t] = last_advantage = deltas[t] + gamma * gae_lambda * last_advantage * (1 - next_step_terminates)
			next_step_terminates = dones[t]

		return advantages

	def _extract_action_mask_from_infos(self, infos: dict | list) -> tensor:
		if self.is_multi_agents:
			# Issue: no infos on dead agent => KeyError
			return torch.from_numpy(np.array(
				[agent_info["action_mask"] for agent_info in infos]
			)).float().to(self.device)
		else:
			return torch.from_numpy(
				np.stack(infos["action_mask"])
			).float().to(self.device)
