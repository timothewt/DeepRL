from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv
from torch import nn, tensor
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical, Normal
from tqdm import tqdm

from algorithms.Algorithm import Algorithm
from models.ActorContinuous import ActorContinuous
from models.FCNet import FCNet
from models.MaskedFCNet import MaskedFCNet


class Buffer:
	"""
	Memory buffer used for A2C
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
		:param max_len: maximum length of the buffer, typically the t_max value for A2C
		:param state_shape: shape of the state given to the policy
		:param actions_nb: number of possible actions (1 for discrete and n for continuous)
		:param device: device used by PyTorch
		"""
		self.states = torch.empty((max_len, num_envs) + state_shape, device=device)
		self.next_states = torch.empty((max_len, num_envs) + state_shape, device=device)
		self.dones = torch.empty((max_len, num_envs, 1), device=device)
		self.actions = torch.empty((max_len, num_envs, actions_nb), device=device)
		self.rewards = torch.empty((max_len, num_envs, 1), device=device)
		self.values = torch.empty((max_len, num_envs, 1), device=device)
		self.log_probs = torch.empty((max_len, num_envs, actions_nb), device=device)
		self.entropies = torch.empty((max_len, num_envs, actions_nb), device=device)

		self.num_envs = num_envs
		self.state_shape = state_shape
		self.actions_nb = actions_nb
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
			states: np.ndarray,
			next_states: np.ndarray,
			dones: bool,
			actions: int,
			rewards: float,
			values: float,
			log_probs: float,
			entropies: float,
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
		:param entropies: entropies of the distributions
		"""
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

	def get_all(self) -> tuple[tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor]:
		"""
		Gives all the values of the buffer
		:return: all buffer tensors
		"""
		return self.states, self.next_states, self.dones, self.actions, self.rewards, self.values, self.log_probs, self.entropies

	def get_all_flattened(self) -> tuple[tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor]:
		"""
		Gives all the buffer values as flattened tensors
		:return: all buffer tensors flattened
		"""
		return self.states.view((self.max_len * self.num_envs,) + self.state_shape), \
			self.next_states.view((self.max_len * self.num_envs,) + self.state_shape), \
			self.dones.flatten(), \
			self.actions.flatten(end_dim=1), \
			self.rewards.flatten(), \
			self.values.flatten(), \
			self.log_probs.flatten(end_dim=1), \
			self.entropies.flatten(end_dim=1)

	def reset(self) -> None:
		self.values = torch.empty((self.max_len, self.num_envs, 1), device=self.device)
		self.log_probs = torch.empty((self.max_len, self.num_envs, self.actions_nb), device=self.device)
		self.entropies = torch.empty((self.max_len, self.num_envs, self.actions_nb), device=self.device)
		self.i = 0


class A2C(Algorithm):

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
		self.num_envs = max(config.get("num_envs", 1), 1)
		self.envs: AsyncVectorEnv = AsyncVectorEnv([self.env_fn for _ in range(self.num_envs)])

		self.env_uses_action_mask = config.get("env_uses_action_mask", False)

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

		self.env_act_space = self.envs.single_action_space
		self.action_mask_size = 1
		if self.env_uses_action_mask:
			assert isinstance(self.env_act_space, spaces.Discrete)
			self.env_obs_space = self.envs.single_observation_space["real_obs"]
			self.action_mask_size = self.env_act_space.n
		else:
			self.env_obs_space = self.envs.single_observation_space
		self.env_flat_obs_space = gym.spaces.utils.flatten_space(self.env_obs_space)
		self.actions_nb = 1

		actor_config = {
			"input_size": int(np.prod(self.env_flat_obs_space.shape)),
			"hidden_layers_nb": config.get("actor_hidden_layers_nb", 3),
			"hidden_size": config.get("actor_hidden_size", 64),
			"output_layer_std": .01,
		}

		if isinstance(self.env_act_space, spaces.Discrete):
			self.actions_type = "discrete"
			actor_config["output_size"] = self.env_act_space.n
			actor_config["output_function"] = nn.Softmax(dim=-1)
			if self.env_uses_action_mask:
				self.actor: nn.Module = MaskedFCNet(config=actor_config).to(self.device)
			else:
				self.actor: nn.Module = FCNet(config=actor_config).to(self.device)
		elif isinstance(self.env_act_space, spaces.Box):
			self.actions_type = "continuous"
			self.action_space_low = torch.from_numpy(self.env_act_space.low).to(self.device)
			self.action_space_high = torch.from_numpy(self.env_act_space.high).to(self.device)
			self.action_space_intervals = (self.action_space_high - self.action_space_low)
			actor_config["actions_nb"] = self.actions_nb = int(np.prod(self.env_act_space.shape))
			self.actor: nn.Module = ActorContinuous(config=actor_config).to(self.device)
		else:
			raise NotImplementedError("Only Discrete or Box action spaces currently supported.")

		self.actor_optimizer: torch.optim.Optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

		critic_config = {
			"input_size": int(np.prod(self.env_flat_obs_space.shape)),
			"output_size": 1,
			"hidden_layers_nb": config.get("critic_hidden_layers_nb", 3),
			"hidden_size": config.get("critic_hidden_size", 64),
			"output_layer_std": 1,
		}
		self.critic: nn.Module = FCNet(config=critic_config).to(self.device)
		self.critic_optimizer: torch.optim.Optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

	def train(self, max_steps: int) -> None:
		# From Algorithm S3 : https://arxiv.org/pdf/1602.01783v2.pdf

		self.writer = SummaryWriter()

		episode = 0

		buffer = Buffer(self.num_envs, self.t_max, self.env_obs_space.shape, self.actions_nb, self.device)

		print("==== STARTING TRAINING ====")

		obs, _ = self.envs.reset()
		masks = new_masks = None
		if self.env_uses_action_mask:
			obs, masks = self._extract_mask_from_obs(obs)
			masks = torch.from_numpy(masks).to(self.device)
		obs = torch.from_numpy(obs).to(self.device)
		first_agent_rewards = 0

		for _ in tqdm(range(max_steps)):
			if self.env_uses_action_mask:
				actor_output = self.actor(obs, masks)
			else:
				actor_output = self.actor(obs)
			critic_output = self.critic(obs)  # value function

			if self.actions_type == "continuous":
				means, std = actor_output
				dist = Normal(loc=means, scale=std)
				actions = dist.sample()
				actions_to_input = self._scale_to_action_space(actions).cpu().numpy()
				log_probs = dist.log_prob(actions)
				entropies = dist.entropy()
			else:  # discrete
				probs = actor_output
				dist = Categorical(probs=probs)
				actions = dist.sample()
				actions_to_input = actions.cpu().numpy()
				log_probs = dist.log_prob(actions).unsqueeze(1)
				actions = actions.unsqueeze(1)
				entropies = dist.entropy().unsqueeze(1)

			new_obs, rewards, dones, truncateds, _ = self.envs.step(actions_to_input)
			dones = dones + truncateds  # done or truncate
			if self.env_uses_action_mask:
				new_obs, new_masks = self._extract_mask_from_obs(new_obs)
				new_masks = torch.from_numpy(new_masks).to(self.device)
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

	def _scale_to_action_space(self, actions: tensor) -> tensor:
		"""
		For continuous action spaces, scales the action given by the distribution to the action spaces
		:param actions: actions given by the distribution
		:return: the scaled actions
		"""
		actions = torch.clamp(actions, 0, 1)
		actions = actions * self.action_space_intervals + self.action_space_low
		return actions

	@staticmethod
	def _extract_mask_from_obs(obs: dict[str: np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
		"""
		Used to take out the mask from the observation when environment requires action masking
		:param obs: raw observation containing the action mask and the real observation
		:return: a tuple of the real observation and the masks
		"""
		real_obs = obs["real_obs"]
		masks = obs["action_mask"]
		return real_obs, masks

