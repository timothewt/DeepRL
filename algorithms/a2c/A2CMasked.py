from datetime import datetime
from typing import Any

import gymnasium as gym
import numpy as np
import supersuit as ss
import torch
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv
from pettingzoo import ParallelEnv
from torch import nn, tensor
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from algorithms.Algorithm import Algorithm
from algorithms.a2c.Buffer import Buffer
from models import FCNet
from models import FCNetMasked


class A2CMasked(Algorithm):
	"""
	Synchronous Advantage Actor-Critic
	Used for discrete action spaces with an action mask. The action mask has to be in the
	infos dict, with the "action_mask" key.
	Environment can either be a gymnasium.Env or a pettingzoo.ParallelEnv.
	"""
	def __init__(self, config: dict[str | Any]):
		"""
		:param config:
			env_fn (Callable[[], gymnasium.Env]): function returning a Gymnasium environment
			num_envs (int): number of environments in parallel
			device (torch.device): device used (cpu, gpu)

			actor_lr (float): learning rate of the actor
			critic_lr (float): learning rate of the critic
			gamma (float): discount factor
			t_max (int): steps between each update
			ent_coef (float): entropy bonus coefficient
			vf_coef (float): value function loss coefficient

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

		# Stats

		self.writer = None

		# Algorithm hyperparameters

		self.actor_lr: float = config.get("actor_lr", .0002)
		self.critic_lr: float = config.get("critic_lr", .0002)
		self.gamma: float = config.get("gamma", .99)
		self.t_max: int = config.get("t_max", 5)
		self.ent_coef: float = config.get("ent_coef", .001)
		self.vf_coef: float = config.get("vf_coef", .5)

		# Policies

		assert isinstance(self.env_act_space, spaces.Discrete), "Action mask only supported in Discrete action spaces!"
		self.action_mask_size = self.env_act_space.n
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

		self.actor: nn.Module = FCNetMasked(config=actor_config).to(self.device)

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

	def train(self, max_steps: int) -> None:
		"""
		From Algorithm S3 : https://arxiv.org/pdf/1602.01783v2.pdf
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
				"t_max": self.t_max,
				"ent_coef": self.ent_coef,
				"vf_coef": self.vf_coef,
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

		buffer = Buffer(self.num_envs, self.t_max, self.env_obs_space.shape, self.actions_nb, self.device)

		print("==== STARTING TRAINING ====")

		obs, infos = self.envs.reset()
		masks = self._extract_action_mask_from_infos(infos)
		obs = torch.from_numpy(self._flatten_obs(obs)).float().to(self.device)
		first_agent_rewards = 0

		for _ in tqdm(range(max_steps), desc="A2C Training"):
			critic_output = self.critic(obs)  # value function

			probs = self.actor(obs, masks)
			dist = Categorical(probs=probs)
			actions = dist.sample()
			actions_to_input = actions.cpu().numpy()
			log_probs = dist.log_prob(actions).unsqueeze(1)
			actions = actions.unsqueeze(1)
			entropies = dist.entropy().unsqueeze(1)

			new_obs, rewards, dones, truncateds, new_infos = self.envs.step(actions_to_input)
			dones = dones + truncateds  # done or truncate
			new_masks = self._extract_action_mask_from_infos(new_infos)
			new_obs = torch.from_numpy(self._flatten_obs(new_obs)).float().to(self.device)

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
			masks = new_masks

			if buffer.is_full():
				self.update_networks(buffer)
				buffer.reset()

			first_agent_rewards += rewards[0]
			if dones[0]:
				self.writer.add_scalar("Rewards", first_agent_rewards, episode)
				first_agent_rewards = 0
				episode += 1

		print("==== TRAINING COMPLETE ====")

	def update_networks(self, buffer: Buffer) -> None:
		"""
		Updates the actor and critic networks according to the A2C paper
		:param buffer: complete buffer of experiences
		"""
		advantages = self._compute_advantages(buffer, self.gamma).flatten(end_dim=1)

		states, next_states, dones, _, rewards, values, log_probs, entropies = buffer.get_all_flattened()

		# Updating the network
		actor_loss = - (log_probs * advantages.detach()).mean()
		critic_loss = advantages.pow(2).mean()
		entropy_loss = entropies.mean()

		self.actor_optimizer.zero_grad()
		self.critic_optimizer.zero_grad()
		(actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy_loss).backward()
		self.actor_optimizer.step()
		self.critic_optimizer.step()

		self.writer.add_scalar("Loss/Actor_Loss", actor_loss.item())
		self.writer.add_scalar("Loss/Critic_Loss", critic_loss.item())
		self.writer.add_scalar("Loss/Entropy", entropy_loss.item())

	def _compute_advantages(self, buffer: Buffer, gamma: float) -> tensor:
		"""
		Computes the advantages for all steps of the buffer
		:param buffer: complete buffer of experiences
		:param gamma: rewards discount rate
		:return: the advantages for each timesteps as a tensor
		"""
		_, next_states, dones, _, rewards, values, _, _ = buffer.get_all()

		R = self.critic(next_states[-1])  # next_value
		returns = torch.zeros((buffer.max_len, self.num_envs, 1), device=self.device)

		for t in reversed(range(buffer.max_len)):
			R = rewards[t] + gamma * R * (1 - dones[t])
			returns[t] = R

		return returns.detach() - values
