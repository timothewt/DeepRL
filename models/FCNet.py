from typing import Any

import torch.nn as nn
import torch.nn.functional as F


class FCNet(nn.Module):

	def __init__(self, config: dict[str: Any]):
		"""
		:param config: config of the network
			input_size: number of inputs of the neural net
			output_size: number of outputs of the neural net
			hidden_layers_nb: number of layers between the input and output layers
			hidden_size: size of the hidden layers
			activation_function: activation function used between the layers
			output_function: function used at the output of the network
		"""
		super(FCNet, self).__init__()

		input_size: int = config.get("input_size", 1)
		output_size: int = config.get("output_size", 1)
		hidden_layers_nb: int = config.get("hidden_layers_nb", 2)
		hidden_size: int = config.get("hidden_size", 32)
		activation_function = config.get("activation_function", F.relu)
		output_function = config.get("output_function", None)

		self.fc = nn.ModuleList(
			[nn.Linear(input_size, hidden_size)] +
			[nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers_nb)] +
			[nn.Linear(hidden_size, output_size)]
		)
		self.activation_function = activation_function
		if output_function is None:
			def output_function(x): return x
		self.output_function = output_function

	def forward(self, x):
		for layer in self.fc[:-1]:
			x = self.activation_function(layer(x))
		x = self.fc[-1](x)
		return self.output_function(x)
