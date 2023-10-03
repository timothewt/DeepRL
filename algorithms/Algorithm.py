from typing import Any

import gymnasium as gym
import torch
from matplotlib import pyplot as plt


class Algorithm:

	def __init__(self, config: dict[str | Any]):
		"""
		:param config:
			device: device used by PyTorch
			env: environment instance
			log_freq: episodes interval for logging the current stats of the algorithm
		"""
		# Device

		self.device = config.get("device", torch.device("cpu"))

		# Environment

		self.env: gym.Env = config.get("env", None)
		assert self.env is not None, \
			"No environment provided!"
		assert isinstance(self.env.action_space, gym.spaces.Discrete), \
			"Only discrete action spaces are currently supported!"
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
				self.log_stats(episode=episode, avg_period=10)

			episode += 1
			steps += current_episode_step

		print("==== TRAINING COMPLETE ====")

	def log_stats(self, episode: int, avg_period: int) -> None:
		print(f"--- Episode {episode} ---\n"
			f"\tAverage reward (last {avg_period} episodes): "
			f"{round(sum(self.rewards[episode - avg_period: episode]) / avg_period, 1)}\n"
			f"\tAverage loss (last {avg_period} episodes): "
			f"{round(sum(self.losses[episode - avg_period: episode]) / avg_period, 3)}")

	def plot_training_stats(self) -> None:
		fig, axs = plt.subplots(2)

		axs[0].plot(self.rewards)
		axs[0].set(xlabel="Episode", ylabel="Reward")
		axs[1].plot(self.losses)
		axs[1].set(xlabel="Episode", ylabel="Loss")

		plt.show()
