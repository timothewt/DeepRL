from typing import Any

import gymnasium as gym
import torch
from matplotlib import pyplot as plt


class Algorithm:

	def __init__(self, config: dict[str | Any]):
		"""
		:param config:
			device: device used by PyTorch
			env_name: name of the environment in the gym register
			log_freq: episodes interval for logging the current stats of the algorithm
		"""
		# Device

		self.device = config.get("device", torch.device("cpu"))

		# Environment

		self.env: gym.Env = gym.make(config.get("env_name", None))
		assert self.env is not None, \
			"No environment provided!"
		self.max_episode_steps = self.env.spec.max_episode_steps

		# Training stats

		self.rewards = []
		self.losses = []
		self.log_freq = config.get("log_freq", 50)

	def train(self, max_steps: int) -> None:
		# Typical train function

		self.rewards = []
		self.losses = []

		steps = 0
		episode = 0

		print("==== STARTING TRAINING ====")

		while steps <= max_steps:
			self.rewards.append(0)
			self.losses.append(0)

			current_episode_step = 0
			done = False
			obs, _ = self.env.reset()

			while not done and current_episode_step <= self.max_episode_steps:
				action = self.env.action_space.sample()
				new_obs, reward, done, _, _ = self.env.step(action)
				obs = new_obs

				current_episode_step += 1

			if episode % self.log_freq == 0:
				self.log_rewards(episode=episode, avg_period=10)

			episode += 1
			steps += current_episode_step

		print("==== TRAINING COMPLETE ====")

	def log_rewards(self, episode: int, avg_period: int) -> None:
		print(f"--- Episode {episode} ---\n"
			f"\tAverage reward (last {avg_period} episodes): "
			f"{round(sum(self.rewards[max(0, episode - avg_period + 1): episode + 1]) / min(episode + 1, avg_period), 1)}")

	def plot_training_stats(self, stats: list[tuple[str | str | list[float]]], n: int = 20) -> None:
		"""
		:param stats: list of tuples of "y-axis title", "x-axis title", "values"
		:param n: average interval
		"""
		cols = len(stats) // 2 + len(stats) % 2
		fig, axs = plt.subplots(2, cols, figsize=(36 // cols, 8))
		n += n % 2  # make it an even number
		n_2 = n // 2

		for i, (y, x, values) in enumerate(stats):
			axs[i % 2][i // cols].plot(values, label="Real value")
			data_for_avg = [values[max(0, j - n_2):j + n_2] for j in range(len(values))]
			axs[i % 2][i // cols].plot([sum(data) / len(data) for data in data_for_avg], label=f"Average on {n}")
			axs[i % 2][i // cols].set(xlabel=x, ylabel=y)
			axs[i % 2][i // cols].legend(loc="upper right")

		fig.tight_layout(pad=.2)
		plt.show()
