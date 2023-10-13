from typing import Any

from matplotlib import pyplot as plt


class Algorithm:

	def __init__(self, config: dict[str | Any]):
		self.rewards = []
		pass

	def train(self, max_steps: int, plot_training_stats: bool = False) -> None:
		raise NotImplementedError

	@staticmethod
	def log_rewards(rewards: list[float], episode: int, avg_period: int) -> None:
		print(f"--- Episode {episode} ---\n"
			f"\tAverage reward (last {avg_period} episodes): "
			f"{round(sum(rewards[max(0, episode - avg_period + 1): episode + 1]) / min(episode + 1, avg_period), 1)}")

	@staticmethod
	def plot_training_stats(stats: list[tuple[str, str, list[float]]], n: int = 20) -> None:
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
			axs[i % 2][i // cols].legend(loc="lower right")

		fig.tight_layout(pad=.2)
		plt.show()
