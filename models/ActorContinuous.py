from typing import Any

import torch.nn as nn
from models.FCNet import FCNet


class ActorContinuous(nn.Module):

	def __init__(self, config: dict[str: Any]):
		"""
		:param config: config of the network
			input_size: number of inputs of the neural net
			hidden_layers_nb: number of layers between the input and output layers
			hidden_size: size of the hidden layers
			activation_function: activation function used between the layers
		"""
		super(ActorContinuous, self).__init__()

		config["output_size"] = 128
		config["output_function"] = None

		self.fc = FCNet(config)

		self.mean_layer = nn.Sequential(
			nn.ReLU(),
			nn.Linear(128, 1),
			nn.Tanh(),
		)

		self.var_layer = nn.Sequential(
			nn.ReLU(),
			nn.Linear(128, 1),
			nn.Softplus(),
		)

	def forward(self, x):
		x = self.fc(x)
		means, vars = self.mean_layer(x), self.var_layer(x)
		return means, vars


