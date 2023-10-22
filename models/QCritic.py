from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor

from models.FCNet import FCNet


class QCritic(nn.Module):
	"""
	Used for DDPG as the Q network
	"""
	def __init__(self, config: dict[str: Any]):
		"""
		:param config: config of the network
			input_size: number of inputs of the neural net
			output_size: number of outputs of the neural net
			hidden_layers_nb: number of layers between the input and output layers
			hidden_size: size of the hidden layers
			actions_nb: number of actions
			activation_function: activation function used between the layers
		"""
		super(QCritic, self).__init__()

		input_size: int = config.get("input_size", 1)
		hidden_size: int = config.get("hidden_size", 1)
		self.activation_function = config.get("activation_function", F.relu)
		self.fc_layer = nn.Linear(input_size, hidden_size)
		config["input_size"] = hidden_size + config.get("actions_nb", 1)
		self.fc = FCNet(config=config)

	def forward(self, x: tensor, actions: tensor) -> tensor:
		x = self.activation_function(self.fc_layer(x))
		x = torch.cat((x, actions))
		return self.fc(x)
