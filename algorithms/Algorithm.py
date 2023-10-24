import os
from typing import Any

import glob
import numpy as np
import torch
from gymnasium import spaces
from torch import tensor, nn


class Algorithm:

	def __init__(self, config: dict[str | Any]):
		self.config = config

		self.num_envs = 1
		self.is_multi_agents = False
		self.device = torch.device("cpu")
		self.action_space_low = tensor([0])
		self.action_space_high = tensor([0])
		self.env_obs_space = spaces.Space()

	def train(self, max_steps: int, save_model: bool = False, save_freq: int = 1_000) -> None:
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
			return np.array([
				spaces.flatten(self.env_obs_space, obs)
			])
		else:
			return np.array([
				spaces.flatten(self.env_obs_space, value) for value in obs
			])

	def _extract_action_mask_from_infos(self, infos: dict) -> tensor:
		if self.is_multi_agents:
			# Issue: no infos on dead agent => KeyError
			return torch.from_numpy(np.array(
				[agent_info["action_mask"] for agent_info in infos]
			)).float().to(self.device)
		else:
			return torch.from_numpy(
				np.stack(infos["action_mask"])
			).float().to(self.device)

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

	@staticmethod
	def save_models(exp_name: str, models: list[tuple[str, nn.Module]]) -> None:
		"""
		Saves the models in the given directory
		:param models: list of tuples of the algorithm's models (name, model)
		:param exp_name: name of the experiment
		"""
		os.makedirs(f"saved_models/{exp_name}", exist_ok=True)
		for name, model in models:
			torch.save(model.state_dict(), f"saved_models/{exp_name}/{name}.pt")

	def load_models(self, dir_path: str) -> None:
		"""
		Loads the models from the given directory
		:param dir_path: directory path
		"""
		for saved_model in glob.glob(f"{dir_path}/*.pt"):
			self.__getattribute__(saved_model.split("\\")[-1].split(".")[0]).load_state_dict(torch.load(saved_model))

	def compute_single_action(self, obs: np.ndarray, infos: dict) -> int:
		raise NotImplementedError
