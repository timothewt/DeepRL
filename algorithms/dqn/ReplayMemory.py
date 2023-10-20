from __future__ import annotations

import numpy as np


class ReplayMemory:
	"""
	Replay memory storing all the previous steps data, up to a certain length, then replaces the oldest data
	"""
	def __init__(self, max_length: int = 10_000, state_shape: tuple[int] = (1,), action_size: int = 1):
		"""
		Initializes the memory
		:param max_length: maximum length of the memory after which the oldest values will be replaced
		:param state_shape: shape of the state
		:param action_size: size of the discrete action space, used in action masking
		"""
		self.states = np.empty((max_length,) + state_shape, dtype=float)
		self.actions = np.empty((max_length,), dtype=int)
		self.rewards = np.empty((max_length,), dtype=float)
		self.next_state = np.empty((max_length,) + state_shape, dtype=float)
		self.is_terminal = np.empty((max_length,), dtype=bool)
		self.masks = np.empty((max_length, action_size), dtype=float)
		self.next_masks = np.empty((max_length, action_size), dtype=float)

		self.max_length = max_length
		self.current_length = 0
		self.i = 0

	def push(
			self,
			state: np.ndarray,
			action: np.ndarray,
			reward: np.ndarray,
			next_state: np.ndarray,
			is_terminal: np.ndarray,
			mask: np.ndarray = None,
			next_mask: np.ndarray = None
	) -> None:
		"""
		Pushes new values in the memory
		:param state: state before the action
		:param action: action taken by the agent
		:param reward: reward given for this action at that state
		:param next_state: next state after doing the action
		:param is_terminal: tells if the action done has terminated the agent
		:param mask: mask used to choose the action
		:param next_mask: action masked used at the next step
		"""
		self.i %= self.max_length
		self.current_length = min(self.current_length + 1, self.max_length)

		self.states[self.i] = state
		self.actions[self.i] = action
		self.rewards[self.i] = reward
		self.next_state[self.i] = next_state
		self.is_terminal[self.i] = is_terminal
		if mask is not None:
			self.masks[self.i] = mask
		if next_mask is not None:
			self.masks[self.i] = mask
		self.i += 1

	def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		"""
		Samples a random batch in all the replay memory
		:param batch_size: size of the returned batch
		:return: a random batch in the memory
		"""
		indices = np.random.choice(self.current_length, size=batch_size)
		return self.states[indices], \
			self.actions[indices], \
			self.rewards[indices], \
			self.next_state[indices], \
			self.is_terminal[indices], \
			self.masks[indices], \
			self.next_masks[indices]
