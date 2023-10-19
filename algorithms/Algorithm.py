from typing import Any

import numpy as np
import torch
from torch import tensor


class Algorithm:

	def __init__(self, config: dict[str | Any]):
		self.config = config

		self.action_space_low = tensor([0])
		self.action_space_high = tensor([0])

	def train(self, max_steps: int) -> None:
		raise NotImplementedError

	@property
	def action_space_intervals(self) -> tensor:
		return self.action_space_high - self.action_space_low

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
