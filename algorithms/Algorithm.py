from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from torch import tensor


class Algorithm:

	def __init__(self, config: dict[str | Any]):
		self.config = config

		self.num_envs = 1
		self.action_space_low = tensor([0])
		self.action_space_high = tensor([0])
		self.env_obs_space = spaces.Space()

	def train(self, max_steps: int) -> None:
		raise NotImplementedError

	@property
	def action_space_intervals(self) -> tensor:
		return self.action_space_high - self.action_space_low

	def _scale_to_action_space(self, actions: tensor) -> tensor:
		"""
		For continuous action spaces, scales the action given by the distribution to the action spaces
		:param actions: actions given by the distribution
		:return: the scaled actions
		"""
		actions = torch.clamp(actions, 0, 1)
		actions = actions * self.action_space_intervals + self.action_space_low
		return actions

	def _flatten_obs(self, obs: np.ndarray) -> np.ndarray:
		"""
		Used to flatten the observations before passing it to the policies and the buffer
		:param obs: observation to flatten
		:return: the observation in one dimension
		"""
		if self.num_envs == 1:
			return spaces.flatten(self.env_obs_space, obs)
		else:
			return np.array([
				spaces.flatten(self.env_obs_space, value) for value in obs
			])

	@staticmethod
	def dict2mdtable(d: dict[str: float], key: str = 'Name', val: str = 'Value'):
		"""
		Used to log hyperparameters in tensorboard
		From https://github.com/tensorflow/tensorboard/issues/46#issuecomment-1331147757
		:param d: dict mapping name to values
		:param key: key in table header
		:param val: value in table header
		:return:
		"""
		rows = [f'| {key} | {val} |']
		rows += ['|--|--|']
		rows += [f'| {k} | {v} |' for k, v in d.items()]
		return "  \n".join(rows)
