from random import randint
from time import time
from typing import Any

import pygame as pg
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GridWorldEnv(gym.Env):
	metadata = {"render_modes": ["ansi", "human"], "render_fps": 5, "name": "GridWorldEnv-v0"}

	def __init__(self, width: int = 12, height: int = 12, render_mode: str | None = None):
		super().__init__()

		self.width: int = width
		self.height: int = height

		self.agent_x: int = 0
		self.agent_y: int = 0

		self.target_x: int = 0
		self.target_y: int = 0

		self.action_space: spaces.MultiDiscrete = spaces.MultiDiscrete([3, 3])  # (up, steady, down), (left, steady, right)
		self.real_obs_space: spaces.Dict = spaces.Dict({
			"agent": spaces.Box(0, np.array([self.width, self.height]), (2,), dtype=np.float32),
			"target": spaces.Box(0, np.array([self.width, self.height]), (2,), dtype=np.float32),
		})
		self.observation_space: spaces.Box = spaces.flatten_space(self.real_obs_space)

		assert render_mode in self.metadata["render_modes"] or render_mode is None
		self.render_mode: str | None = render_mode
		if self.render_mode == "human":
			pg.init()
			self.cell_size = 24
			self.screen = pg.display.set_mode((self.width * self.cell_size, self.height * self.cell_size))
			self.last_render_update = time()

	def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict[str, Any]]:
		self.agent_x, self.agent_y = randint(0, self.width - 1), randint(0, self.height - 1)
		self.target_x, self.target_y = randint(0, self.width - 1), randint(0, self.height - 1)

		self.render()

		return self._get_obs(), self._get_infos()

	def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
		prev_dist_to_target = self._get_dist_to_target()
		self.agent_y += action[0] - 1
		self.agent_x += action[1] - 1
		dist_to_target = self._get_dist_to_target()

		terminated = self.agent_x == self.target_x and self.agent_y == self.target_y
		truncated = self.agent_x < 0 or self.agent_x >= self.width or self.agent_y < 0 or self.agent_y >= self.height
		reward = self._get_reward(dist_to_target < prev_dist_to_target, terminated, truncated)

		self.render()

		return self._get_obs(), reward, terminated, truncated, self._get_infos()

	def _get_dist_to_target(self):
		return abs(self.agent_x - self.target_x) + abs(self.agent_y - self.target_y)

	def render(self):
		if self.render_mode == "ansi":
			str_grid = ""
			for y in range(self.height):
				for x in range(self.width):
					if x == self.agent_x and y == self.agent_y:
						str_grid += "A "
					elif x == self.target_x and y == self.target_y:
						str_grid += "T "
					else:
						str_grid += ". "
				str_grid += "\n"
			return str_grid
		elif self.render_mode == "human":
			while time() - self.last_render_update < (1 / self.metadata["render_fps"]):
				pass
			self.last_render_update = time()

			self.screen.fill((20, 20, 100))
			pg.draw.rect(self.screen, (255, 0, 0), (self.target_x * self.cell_size, self.target_y * self.cell_size, self.cell_size, self.cell_size))
			pg.draw.circle(self.screen, (0, 0, 255), (self.agent_x * self.cell_size + 12, self.agent_y * self.cell_size + 12), 12)

			for y in range(self.height):
				pg.draw.line(self.screen, (0, 0, 0), (0, y * self.cell_size), (self.width * self.cell_size, y * self.cell_size))
			for x in range(self.width):
				pg.draw.line(self.screen, (0, 0, 0), (x * self.cell_size, 0), (x * self.cell_size, self.height * self.cell_size))

			pg.display.flip()

	def _get_obs(self):
		obs = {
			"agent": np.array([self.agent_x, self.agent_y]),
			"target": np.array([self.target_x, self.target_y]),
		}
		return spaces.flatten(self.real_obs_space, obs)

	@staticmethod
	def _get_infos():
		return {}

	@staticmethod
	def _get_reward(got_closer: bool, terminated: bool, truncated: bool) -> float:
		if truncated:
			return -100
		elif terminated:
			return 10
		elif got_closer:
			return .5
		else:
			return -1
