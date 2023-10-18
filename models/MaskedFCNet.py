from typing import Any

import torch
from torch import tensor, nn

from models.FCNet import FCNet


class MaskedFCNet(nn.Module):

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
		super(MaskedFCNet, self).__init__()

		output_function = config.get("output_function", None)
		if output_function is None:
			def output_function(x): return x
		self.output_function = output_function
		config["output_function"] = None
		self.fc = FCNet(config=config)

	def forward(self, x, mask: tensor):
		return self.output_function(
			self.fc(x) + torch.clamp(torch.log(mask), min=-3.4e38)
		)
