from typing import Any


class Algorithm:

	def __init__(self, config: dict[str | Any]):
		self.config = config

	def train(self, max_steps: int) -> None:
		raise NotImplementedError
