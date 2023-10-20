import numpy as np
import torch
from torch import tensor


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
			action_mask_size: int = 1
	):
		"""
		Initialization of the buffer
		:param num_envs: number of parallel environments
		:param max_len: maximum length of the buffer, typically the PPO horizon parameter
		:param state_shape: shape of the state given to the policy
		:param actions_nb: number of possible actions (1 for discrete and n for continuous)
		:param device: device used by PyTorch
		:param action_mask_size: size of the action mask, corresponding to the size of the Discrete action space
		"""
		self.states = torch.empty((max_len, num_envs, np.prod(state_shape)), device=device)
		self.next_states = torch.empty((max_len, num_envs, np.prod(state_shape)), device=device)
		self.dones = torch.empty((max_len, num_envs, 1), device=device)
		self.actions = torch.empty((max_len, num_envs, actions_nb), device=device)
		self.rewards = torch.empty((max_len, num_envs, 1), device=device)
		self.values = torch.empty((max_len, num_envs, 1), device=device)
		self.log_probs = torch.empty((max_len, num_envs, actions_nb), device=device)
		self.action_masks = torch.empty((max_len, num_envs, action_mask_size), device=device)

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
			action_mask: tensor = None,
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
		:param action_mask: action mask used in some environments
		"""
		assert self.i < self.max_len, "Buffer is full!"

		self.states[self.i] = states
		self.next_states[self.i] = next_states
		self.dones[self.i] = dones
		self.actions[self.i] = actions
		self.rewards[self.i] = rewards
		self.values[self.i] = values
		self.log_probs[self.i] = log_probs
		if action_mask is not None:
			self.action_masks[self.i] = action_mask

		self.i += 1

	def get_all(self) -> tuple[tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor]:
		"""
		Gives all the values of the buffer
		:return: all buffer tensors
		"""
		return self.states, self.next_states, self.dones, self.actions, self.rewards, self.values, self.log_probs, self.action_masks

	def get_all_flattened(self) -> tuple[tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor]:
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
			self.log_probs.flatten(end_dim=1), \
			self.action_masks.flatten(end_dim=1)

	def reset(self) -> None:
		"""
		Resets the iteration variable
		"""
		self.i = 0
