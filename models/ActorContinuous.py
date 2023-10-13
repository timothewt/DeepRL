from typing import Any

import torch
import torch.nn as nn
from models.FCNet import FCNet


class ActorContinuous(nn.Module):

	def __init__(self, config: dict[str: Any]):
		"""
		:param config: config of the network
			input_size: number of inputs of the neural net
			hidden_layers_nb: number of layers between the input and output layers
			hidden_size: size of the hidden layers
		"""
		super(ActorContinuous, self).__init__()

		self.actions_nb = config.get("actions_nb", 1)
		config["output_size"] = self.actions_nb
		config["output_function"] = nn.Sigmoid()

		self.means_net = FCNet(config)

		self.log_stds = nn.Parameter(torch.zeros(1, self.actions_nb))

	def forward(self, x):
		means = self.means_net(x)
		log_stds = self.log_stds.expand_as(means)
		stds = torch.exp(log_stds)
		return means, stds


