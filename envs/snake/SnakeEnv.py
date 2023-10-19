from random import randint
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete, Dict


class SnakeEnv(gym.Env):
	metadata = {"render_modes": ["ansi"], "render_fps": 1, "name": "MinesweeperEnv-v0"}

	def __init__(self, width: int = 12, height: int = 12, render_mode: str | None = None):
		super().__init__()

		self.width: int = width
		self.height: int = height
		self.grid_dimension: int = self.width * self.height

		self.food: tuple[int, int] = (0, 0)
		self.snake: list[tuple[int, int]] = [(0, 0)]  # snake's head is the first element of the list
		self.score: int = 0

		self.action_space = Discrete(4)  # up, right, down, left
		self.observation_space = Dict({
			"action_mask": Box(0, 1, (4,), dtype=np.float32),
			"real_obs": Dict({
				"head": Box(0, np.array([self.width, self.height]), (2,), dtype=np.float32),
				"food": Box(0, np.array([self.width, self.height]), (2,), dtype=np.float32),
				"distances_to_closest_obstacles": Box(0, np.sqrt(2) * max(self.width, self.height), (8,), dtype=np.float32),
			})
		})

		self.render_mode: str | None = render_mode

	def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict[str, Any]]:
		self.snake = [(randint(0, self.width - 1), randint(0, self.height - 1))]
		self._place_food()

		return self._get_obs(), {}

	def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:

		curr_x, curr_y = self.snake[0][0], self.snake[0][1]
		match action:
			case 0:
				new_head = (curr_x, curr_y - 1)
			case 1:
				new_head = (curr_x + 1, curr_y)
			case 2:
				new_head = (curr_x, curr_y + 1)
			case _:
				new_head = (curr_x - 1, curr_y)

		truncated = new_head in self.snake

		ate_food = new_head == self.food
		if ate_food:
			self._place_food()
		else:
			self.snake.pop()
		self.snake.insert(0, new_head)

		obs = self._get_obs()
		reward = self._get_reward(ate_food, truncated)
		terminated = len(self.snake) == self.grid_dimension

		return obs, reward, terminated, truncated, {}

	def render(self) -> None | str:
		if self.render_mode == "ansi":
			str_grid = ""
			for row in range(self.height):
				for col in range(self.width):
					if (col, row) == self.food:
						str_grid += "O  "
					elif (col, row) in self.snake:
						str_grid += "■  "
					else:
						str_grid += ".  "
				str_grid += "\n"
			return str_grid

	def close(self):
		pass

	@staticmethod
	def _get_reward(truncated: bool, ate_food: bool) -> float:
		if ate_food:
			return 10
		elif truncated:
			return -50
		else:
			return .5

	def _get_obs(self) -> dict[str: np.ndarray]:
		return {
			"action_mask": self._get_action_mask(),
			"real_obs": {
				"head": np.array(self.snake[0]),
				"food": np.array(self.food),
				"closest_obstacles": np.array(self._get_distances_to_closest_obstacles(), dtype=np.float32),
			}
		}

	def _get_distances_to_closest_obstacles(self) -> tuple[float, float, float, float]:
		curr_x, curr_y = self.snake[0][0], self.snake[0][1]
		up, right, down, left = curr_y + 1, self.width - curr_x, self.height - curr_y, curr_x + 1

		for i in range(1, curr_y + 1):
			if (curr_x, curr_y - i) in self.snake:
				up = i
				break

		for i in range(curr_x + 1, self.width):
			if (i, curr_y) in self.snake:
				right = i
				break

		for i in range(curr_y + 1, self.height):
			if (curr_x, i) in self.snake:
				down = i
				break

		for i in range(1, curr_x + 1):
			if (curr_x - i, curr_y) in self.snake:
				left = i
				break

		return up, right, down, left

	def _get_action_mask(self) -> np.ndarray:
		curr_x, curr_y = self.snake[0][0], self.snake[0][1]
		mask = np.empty((4,), dtype=np.float32)
		mask[0] = curr_y > 0 and (curr_x, curr_y - 1) not in self.snake
		mask[1] = curr_x < self.width - 1 and (curr_x + 1, curr_y) not in self.snake
		mask[2] = curr_y < self.height - 1 and (curr_x, curr_y + 1) not in self.snake
		mask[3] = curr_x > 0 and (curr_x - 1, curr_y) not in self.snake
		return mask

	def _place_food(self) -> None:
		while True:
			if (food_position := (randint(0, self.width - 1), randint(0, self.height - 1))) not in self.snake:
				self.food = food_position
				break
