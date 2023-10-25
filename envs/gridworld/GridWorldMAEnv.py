from random import randint
from time import time
from typing import Any

import pygame as pg
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


class GridWorldMAEnv(ParallelEnv):
	"""
	Multi-agents gridworld environment, multiple agents trying to reach the target, when one reaches, respawns somewhere else.
	"""
	metadata = {"render_modes": ["ansi", "human"], "render_fps": 5, "name": "GridWorldMAEnv-v0"}

	def __init__(self, agents_nb: int = 1, width: int = 12, height: int = 12, render_mode: str | None = None):
		super().__init__()

		self.width: int = width
		self.height: int = height

		self.max_steps: int = width * height // 2
		self.steps: int = 0

		self.possible_agents = list(range(agents_nb))
		self.agents = self.possible_agents[:]

		self.agents_xs: np.ndarray = np.array([0] * agents_nb)
		self.agents_ys: np.ndarray = np.array([0] * agents_nb)
		self.agents_steps: np.ndarray = np.array([0] * agents_nb)

		# sets random colors for each agent
		self.colors: list[tuple[int, int, int]] = [
			(randint(0, 255), randint(0, 255), randint(0, 255)) for _ in range(agents_nb)
		]

		self.target_x: int = 0
		self.target_y: int = 0

		self.action_spaces: dict[int: spaces.MultiDiscrete] = {
			agent: spaces.MultiDiscrete((3, 3)) for agent in self.agents
		}  # (up, steady, down), (left, steady, right)
		self.real_obs_space: spaces.Dict = spaces.Dict({
			"agent": spaces.Box(0, np.array([self.width, self.height]), (2,), dtype=np.float32),
			"target": spaces.Box(0, np.array([self.width, self.height]), (2,), dtype=np.float32),
		})
		self.observation_spaces = {agent: spaces.flatten_space(self.real_obs_space) for agent in self.agents}

		assert render_mode in self.metadata["render_modes"] or render_mode is None
		self.render_mode: str | None = render_mode
		if self.render_mode == "human":
			pg.init()
			self.cell_size = 24
			self.screen = pg.display.set_mode((self.width * self.cell_size, self.height * self.cell_size))
			self.last_render_update = time()

	def reset(self, seed=None, options=None) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
		self.agents_xs = np.random.randint(0, self.width - 1, size=self.num_agents)
		self.agents_ys = np.random.randint(0, self.height - 1, size=self.num_agents)
		self.agents_steps = np.array([0] * self.num_agents)
		self.target_x, self.target_y = randint(0, self.width - 1), randint(0, self.height - 1)
		self.steps = 0

		self.render()

		return self._get_obs(), self._get_infos()

	def step(self, actions: dict[int: np.ndarray]) -> tuple[
		dict[int: np.ndarray],
		dict[int: float],
		dict[int: bool],
		dict[int: bool],
		dict[str, Any]
	]:
		dx = np.array([actions[agent][1] for agent in self.agents]) - 1
		dy = np.array([actions[agent][0] for agent in self.agents]) - 1

		prev_dists_to_target = self._get_dists_to_target()
		self.agents_ys += dy
		self.agents_xs += dx
		self.agents_steps += 1
		dists_to_target = self._get_dists_to_target()

		got_closer = dists_to_target < prev_dists_to_target
		terminateds = {
			agent: self.agents_xs[agent] == self.target_x and self.agents_ys[agent] == self.target_y
			for agent in self.agents
		}
		truncateds = {
			agent: self.agents_xs[agent] < 0 or self.agents_xs[agent] >= self.width or self.agents_ys[agent] < 0 or self.agents_ys[agent] >= self.height or self.agents_steps[agent] >= self.max_steps
			for agent in self.agents
		}
		rewards = {
			agent: self._get_reward(got_closer[agent], terminateds[agent], truncateds[agent])
			for agent in self.agents
		}

		for agent in self.agents:
			if terminateds[agent] or truncateds[agent]:
				self.agents_xs[agent] = randint(0, self.width - 1)
				self.agents_ys[agent] = randint(0, self.height - 1)
				self.agents_steps[agent] = 0

		if self.steps >= self.max_steps:
			self.target_x, self.target_y = randint(0, self.width - 1), randint(0, self.height - 1)
			self.steps = 0

		self.steps += 1

		self.render()

		return self._get_obs(), rewards, terminateds, truncateds, self._get_infos()

	def _get_dists_to_target(self) -> np.ndarray:
		return abs(self.agents_xs - self.target_x) + abs(self.agents_ys - self.target_y)

	def render(self):
		if self.render_mode == "ansi":
			str_grid = ""
			for y in range(self.height):
				for x in range(self.width):
					if x in self.agents_xs and y in self.agents_ys:
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

			for y in range(self.height):
				pg.draw.line(self.screen, (0, 0, 0), (0, y * self.cell_size), (self.width * self.cell_size, y * self.cell_size))
			for x in range(self.width):
				pg.draw.line(self.screen, (0, 0, 0), (x * self.cell_size, 0), (x * self.cell_size, self.height * self.cell_size))

			pg.draw.rect(self.screen, (255, 0, 0), (self.target_x * self.cell_size, self.target_y * self.cell_size, self.cell_size, self.cell_size))
			for i in range(self.num_agents):
				pg.draw.circle(
					self.screen,
					self.colors[i],
					(self.agents_xs[i] * self.cell_size + self.cell_size // 2, self.agents_ys[i] * self.cell_size + self.cell_size // 2),
					self.cell_size // 2
				)

			pg.display.flip()

	def observation_space(self, agent: int) -> spaces.Space:
		return self.observation_spaces[agent]

	def action_space(self, agent: int) -> spaces.Space:
		return self.action_spaces[agent]

	def _get_obs(self):
		obs = {}
		for agent in self.agents:
			agent_obs = {
				"agent": np.array([self.agents_xs[agent], self.agents_ys[agent]]),
				"target": np.array([self.target_x, self.target_y]),
			}
			obs[agent] = spaces.flatten(self.real_obs_space, agent_obs)
		return obs

	def _get_infos(self):
		return {
			agent: {
				"action_mask": self._get_action_mask(agent)
			} for agent in self.agents
		}

	def _get_action_mask(self, agent: int) -> tuple[np.ndarray, np.ndarray]:
		vertical_mask = np.ones(3, dtype=np.float32)
		horizontal_mask = np.ones(3, dtype=np.float32)
		if self.agents_xs[agent] == 0:
			horizontal_mask[0] = 0
		elif self.agents_xs[agent] == self.width - 1:
			horizontal_mask[2] = 0
		if self.agents_ys[agent] == 0:
			vertical_mask[0] = 0
		elif self.agents_ys[agent] == self.height - 1:
			vertical_mask[2] = 0
		return vertical_mask, horizontal_mask

	@staticmethod
	def _get_reward(got_closer: bool, terminated: bool, truncated: bool) -> float:
		if truncated:
			return -100
		elif terminated:
			return 20
		elif got_closer:
			return .1
		else:
			return -1
