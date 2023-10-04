from typing import Any

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from models.FCNet import FCNet


class ActorCritic(nn.Module):
	def __init__(self, config: dict[str: Any]):
		"""
		:param config: config of the network
			input_size: number of inputs of the neural net
			output_size: number of outputs of the neural net
			hidden_layers_nb: number of layers between the input and output layers
			hidden_size: size of the hidden layers
			activation_function: activation function used between the layers
		"""
		super(ActorCritic, self).__init__()

		self.fc = FCNet(config=config)
		self.actor_output = nn.Softmax(dim=-1)
		self.critic_output = nn.Linear(config["output_size"], 1)

	def forward(self, x):
		x = self.fc(x)
		probs = self.actor_output(x)
		dist = Categorical(probs=probs)
		value = self.critic_output(x)

		return dist, value
