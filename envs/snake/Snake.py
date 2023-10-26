from random import randint
from time import time
from typing import Any

import pygame as pg
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Snake(gym.Env):
	metadata = {"render_modes": ["ansi", "human"], "render_fps": 15, "name": "Snake-v0"}

	def __init__(self, width: int = 12, height: int = 12, render_mode: str | None = None):
		super().__init__()

		self.width: int = width
		self.height: int = height
		self.grid_dimension: int = self.width * self.height

		self.food: tuple[int, int] = (0, 0)
		self.snake: list[tuple[int, int]] = [(0, 0)]  # snake's head is the first element of the list
		self.score: int = 0

		self.action_space = spaces.Discrete(4)  # up, right, down, left
		self.real_obs_space = spaces.Dict({
			"head": spaces.Box(0, np.array([self.width, self.height]), (2,), dtype=np.float32),
			"food": spaces.Box(0, np.array([self.width, self.height]), (2,), dtype=np.float32),
			"closest_obstacles": spaces.Box(0, max(self.width, self.height), (4,), dtype=np.float32),
		})
		self.observation_space = spaces.flatten_space(self.real_obs_space)

		assert self.render_mode in self.metadata["render_modes"] or self.render_mode is None
		self.render_mode: str | None = render_mode
		if self.render_mode == "human":
			pg.init()
			self.pg_clock = pg.time.Clock()
			self.cell_size = 40
			self.screen = pg.display.set_mode((self.width * self.cell_size, self.height * self.cell_size))
			self.last_render_update = time()

	def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict[str, Any]]:
		self.snake = [(randint(0, self.width - 1), randint(0, self.height - 1))]
		self._place_food()
		self.score = 0

		self.render()

		return self._get_obs(), {"action_mask": self._get_action_mask()}

	def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:

		curr_x, curr_y = self.snake[0][0], self.snake[0][1]
		prev_distance_to_food = np.linalg.norm(np.array([curr_x - self.food[0], curr_y - self.food[1]]))
		match action:
			case 0:
				new_head = (curr_x, curr_y - 1)
			case 1:
				new_head = (curr_x + 1, curr_y)
			case 2:
				new_head = (curr_x, curr_y + 1)
			case _:
				new_head = (curr_x - 1, curr_y)
		distance_to_food = np.linalg.norm(np.array([new_head[0] - self.food[0], new_head[1] - self.food[1]]))

		truncated = new_head in self.snake or not 0 <= new_head[0] < self.width or not 0 <= new_head[1] < self.width

		ate_food = new_head == self.food
		got_closer_to_food = prev_distance_to_food > distance_to_food
		if ate_food:
			self.score += 1
			self._place_food()
		else:
			self.snake.pop()
		self.snake.insert(0, new_head)

		obs = self._get_obs()
		reward = self._get_reward(truncated, ate_food, got_closer_to_food)
		terminated = len(self.snake) == self.grid_dimension

		self.render()

		return obs, reward, terminated, truncated, {"action_mask": self._get_action_mask()}

	def render(self) -> None | str:
		if self.render_mode == "ansi":
			str_grid = ""
			for row in range(self.height):
				for col in range(self.width):
					if (col, row) == self.food:
						str_grid += "O  "
					elif (col, row) == self.snake[0]:
						str_grid += "▣  "
					elif (col, row) in self.snake:
						str_grid += "■  "
					else:
						str_grid += "⬝  "
				str_grid += "\n"
			return str_grid
		elif self.render_mode == "human":
			while time() - self.last_render_update < (1 / self.metadata["render_fps"]):
				pass
			self.last_render_update = time()
			self.pg_clock.tick()

			self.screen.fill((20, 20, 80))
			pg.draw.rect(self.screen, (255, 0, 0), (self.food[0] * self.cell_size, self.food[1] * self.cell_size, self.cell_size, self.cell_size))
			pg.draw.rect(self.screen, (0, 200, 0), (self.snake[0][0] * self.cell_size, self.snake[0][1] * self.cell_size, self.cell_size, self.cell_size))
			for x, y in self.snake[1:]:
				pg.draw.rect(self.screen, (0, 255, 0), (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
			pg.display.set_caption(f"SnakeEnv-v0 | Score: {self.score}")
			self.screen.blit(pg.font.SysFont("gothambold", 30).render(f"Score: {self.score}", True, (255, 255, 255)), (5, 5))
			pg.display.flip()

	def close(self):
		pass

	@staticmethod
	def _get_reward(truncated: bool, ate_food: bool, got_closer_to_food: bool) -> float:
		if truncated:
			return -50
		elif ate_food:
			return 10
		elif got_closer_to_food:
			return .2
		else:
			return -.2

	def _get_obs(self) -> dict[str: np.ndarray]:
		return spaces.flatten(self.real_obs_space, {
			"head": np.array(self.snake[0]),
			"food": np.array(self.food),
			"closest_obstacles": np.array(self._get_distances_to_closest_obstacles(), dtype=np.float32),
		})

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
		mask = np.ones((4,), dtype=np.float32)
		if len(self.snake) == 1:
			return mask
		mask[0] = (curr_x, curr_y - 1) != self.snake[1]
		mask[1] = (curr_x + 1, curr_y) != self.snake[1]
		mask[2] = (curr_x, curr_y + 1) != self.snake[1]
		mask[3] = (curr_x - 1, curr_y) != self.snake[1]
		return mask

	def _place_food(self) -> None:
		while True:
			if (food_position := (randint(0, self.width - 1), randint(0, self.height - 1))) not in self.snake:
				self.food = food_position
				break
