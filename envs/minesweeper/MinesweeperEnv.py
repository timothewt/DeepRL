from random import randint
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete, Dict


class MinesweeperEnv(gym.Env):
	metadata = {"render_modes": ["ansi"], "render_fps": 1, "name": "MinesweeperEnv-v0"}

	def __init__(self, grid_width: int = 9, grid_height: int = 9, bombs_nb: int = 12, render_mode: str | None = None):
		super().__init__()

		self.grid_width: int = grid_width
		self.grid_height: int = grid_height
		self.grid_dimension: int = self.grid_height * self.grid_width
		self.bombs_nb: int = bombs_nb
		assert self.bombs_nb < self.grid_height * self.grid_width, "Too many bombs!"

		self.grid: np.ndarray = np.full((self.grid_height, self.grid_width), -1., dtype=np.float32)
		self.bombs: set[tuple[int, int]] = set()
		self.discovered_cells_nb: int = 0

		self.action_space = Discrete(self.grid_width * self.grid_height)
		self.observation_space = Box(-1, 9, (self.grid_dimension,), dtype=np.float32)

		self.render_mode: str | None = render_mode

	def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict[str, Any]]:
		self.grid = np.full((self.grid_height, self.grid_width), -1., dtype=np.float32)
		self.bombs = set()
		self.discovered_cells_nb = 0

		for i in range(self.bombs_nb):
			while True:
				if (new_bomb := (randint(0, self.grid_width - 1), randint(0, self.grid_height - 1))) not in self.bombs:
					self.bombs.add(new_bomb)
					break

		return self._get_obs(), {"action_mask": self._get_action_mask()}

	def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
		row, col = action // self.grid_width, action % self.grid_width

		terminated = truncated = False

		if (row, col) in self.bombs:
			self.grid[row, col] = 9
			self.discovered_cells_nb += 1
			truncated = True
		else:
			self._discover_cell(row, col)
			terminated = (self.grid < 0).sum() == self.bombs_nb

		reward = self._get_reward(terminated, truncated)
		return self._get_obs(), reward, terminated, truncated, {"action_mask": self._get_action_mask()}

	def render(self) -> None | str:
		if self.render_mode == "ansi":
			str_grid = ""
			for row in self.grid:
				for cell in row:
					if cell == -1:
						str_grid += f"■  "
					elif cell == 9:
						str_grid += f"💣 "
					else:
						str_grid += f"{int(cell)}  "
				str_grid += "\n"
			return str_grid

	def close(self):
		pass

	def _get_undiscovered_neighbors(self, row: int, col: int) -> set[tuple[int, int]]:
		neighbors = {
			(row - 1, col - 1),
			(row - 1, col),
			(row - 1, col + 1),
			(row, col - 1),
			(row, col + 1),
			(row + 1, col + 1),
			(row + 1, col),
			(row + 1, col - 1),
		}
		return {
			cell for cell in neighbors if
			0 <= cell[0] < self.grid_height and 0 <= cell[1] < self.grid_width and self.grid[cell[0], cell[1]] == -1
		}

	def _discover_cell(self, row, col) -> None:
		neighbors = self._get_undiscovered_neighbors(row, col)
		if (neighbour_bombs_nb := len(neighbors.intersection(self.bombs))) > 0:
			self.grid[row, col] = neighbour_bombs_nb
		else:
			self.grid[row, col] = 0
			for neighbour in neighbors:
				self._discover_cell(neighbour[0], neighbour[1])

	@staticmethod
	def _get_reward(terminated, truncated) -> float:
		if terminated:
			return 10
		elif truncated:
			return -50
		else:
			return .1

	def _get_obs(self) -> dict[str: np.ndarray]:
		return self.grid.flatten()

	def _get_action_mask(self) -> np.ndarray:
		return (self.grid == -1).flatten().astype(np.float32)
