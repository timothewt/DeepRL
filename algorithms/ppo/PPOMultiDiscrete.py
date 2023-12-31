from datetime import datetime
from typing import Any

import os
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
from algorithms.ppo.Buffer import Buffer
from models import FCNet, FCNetMultiHead


class PPOMultiDiscrete(Algorithm):
	"""
	Proximal Policy Optimization
	Used for multi-discrete action spaces without action mask.
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
			critic_hidden_size (int): size of the hidden linear layers in the critic network
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
		assert isinstance(self.env_act_space, spaces.MultiDiscrete), \
			"Only multi-discrete action spaces are supported. For [space_type] spaces, see PPO[space_type]"

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
		self.actions_nb = np.prod(self.env_act_space.shape)

		self.actor_hidden_layers_nb = config.get("actor_hidden_layers_nb", 3)
		self.actor_hidden_size = config.get("actor_hidden_size", 64)
		self.critic_hidden_layers_nb = config.get("critic_hidden_layers_nb", 3)
		self.critic_hidden_size = config.get("critic_hidden_size", 64)

		actor_config = {
			"input_size": np.prod(self.env_flat_obs_space.shape),
			"hidden_layers_nb": self.actor_hidden_layers_nb,
			"hidden_size": self.actor_hidden_size,
			"output_layer_std": .01,
			"heads_nb": self.actions_nb,
			"output_sizes": self.env_act_space.nvec,
			"output_function": nn.Softmax(dim=-1)
		}
		self.actor: nn.Module = FCNetMultiHead(config=actor_config).to(self.device)
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

	def train(self, max_steps: int, save_models: bool = False, checkpoints: bool = False, save_freq: int = 1_000) -> None:
		"""
		Trains the algorithm on the chosen environment
		From https://arxiv.org/pdf/1707.06347.pdf and https://arxiv.org/pdf/2205.09123.pdf
		:param max_steps: maximum number of steps for the whole training process
		:param save_models: indicates if the models should be saved at the end of the training
		:param checkpoints: indicates if the models should be saved at regular intervals
		:param save_freq: frequency at which the models should be saved
		"""
		exp_name = f"PPO-{self.env.metadata.get('name', 'env_')}-{datetime.now().strftime('%d-%m-%y_%Hh%Mm%S')}"
		self.writer = SummaryWriter(
			f"runs/{exp_name}",
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

		for step in tqdm(range(max_steps), desc="PPO Training"):
			critic_output = self.critic(obs)  # value function

			probs = self.actor(obs)
			dists = [
				Categorical(probs=probs[i]) for i in range(self.actions_nb)
			]
			actions = torch.stack([
				dist.sample() for dist in dists
			]).to(self.device)
			actions_to_input = actions.T.cpu().numpy()

			log_probs = torch.stack([
				dists[i].log_prob(actions[i]) for i in range(self.actions_nb)
			]).T

			new_obs, rewards, terminateds, truncateds, new_infos = self.envs.step(actions_to_input)
			dones = terminateds + truncateds  # done or truncate
			new_obs = torch.from_numpy(self._flatten_obs(new_obs)).float().to(self.device)

			buffer.push(
				obs,
				new_obs,
				torch.from_numpy(dones).float().to(self.device).unsqueeze(1),
				actions.T,  # needs to be transposed to be of shape (num_envs, actions_nb) instead of (actions_nb, num_envs)
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

			if save_models and checkpoints and step % save_freq == 0:
				self.save_models(f"{exp_name}/step{step}", [("actor", self.actor), ("critic", self.critic)])

		print("==== TRAINING COMPLETE ====")
		if save_models:
			self.save_models(exp_name, [("actor", self.actor), ("critic", self.critic)])

	def _update_networks(self, buffer: Buffer) -> None:
		"""
		Updates the actor and critic networks according to the PPO paper
		:param buffer: complete buffer of experiences
		"""
		states, _, _, actions, rewards, values, old_log_probs, _ = buffer.get_all_flattened()
		values, old_log_probs = values.detach().view(self.batch_size, 1), old_log_probs.detach()

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
				new_dists = [
					Categorical(probs=probs[i]) for i in range(self.actions_nb)
				]
				new_log_probs = torch.stack([
					new_dists[i].log_prob(actions[minibatch_indices].T[i]) for i in range(self.actions_nb)
				]).T.view(self.minibatch_size, self.actions_nb)
				new_entropy = torch.stack([
					new_dist.entropy() for new_dist in new_dists
				])
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
		_, next_states, dones, _, rewards, values, _, _ = buffer.get_all()

		next_values = values.roll(-1, dims=0)
		next_values[-1] = self.critic(next_states[-1])

		deltas = (rewards + gamma * next_values - values).detach()

		advantages = torch.zeros(deltas.shape, device=self.device)
		last_advantage = advantages[-1]
		next_step_terminates = 0  # should be the dones of the next step after last step of the batch however cannot reach it
		for t in reversed(range(buffer.max_len)):
			advantages[t] = last_advantage = deltas[t] + gamma * gae_lambda * last_advantage * (1 - next_step_terminates)
			next_step_terminates = dones[t]

		return advantages

	def compute_single_action(self, obs: np.ndarray, infos: dict) -> int | np.ndarray:
		"""
		Computes one action for the given observation
		:param obs: observation to compute the action from
		:param infos: infos given by the environment
		:return: the action
		"""
		if not self.is_multi_agents:
			obs = obs.reshape(1, -1)
		obs = torch.from_numpy(self._flatten_obs(obs)).float().to(self.device)
		probs = self.actor(obs)
		dists = [
			Categorical(probs=probs[i]) for i in range(self.actions_nb)
		]
		actions = torch.stack([
			dist.sample() for dist in dists
		]).view(self.num_agents,self.actions_nb)
		return actions.detach().numpy()
