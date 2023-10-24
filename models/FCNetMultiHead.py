from typing import Any

import torch
from torch import tensor, nn

from models.FCNet import FCNet


class FCNetMultiHead(nn.Module):

	def __init__(self, config: dict[str: Any]):
		"""
		:param config: config of the network
			input_size: number of inputs of the neural net
			output_sizes: number of outputs of the neural net for each head (i.e. [2, 3] for 2 heads with 2 and 3 outputs)
			hidden_layers_nb: number of layers between the input and output layers
			hidden_size: size of the hidden layers
			activation_function: activation function used between the layers
			output_function: function used at the output of the network
			heads_nb: number of heads
		"""
		super(FCNetMultiHead, self).__init__()

		output_function = config.get("output_function", None)
		heads_nb = config.get("heads_nb", 1)
		output_sizes = config.get("output_sizes", [1 for _ in range(heads_nb)])
		assert len(output_sizes) == heads_nb, "output_sizes must have a length of heads_nb"
		hidden_size = config.get("hidden_size", 1)
		if output_function is None:
			def output_function(x): return x
		self.output_function = output_function
		config["output_function"] = config.get("activation_function", None)
		config["output_size"] = hidden_size
		config["hidden_layers_nb"] = config.get("hidden_layers_nb", 1) - 1
		self.fc = FCNet(config=config)

		self.output_layers = nn.ModuleList(
			[nn.Linear(hidden_size, output_sizes[i]) for i in range(heads_nb)]
		)

	def forward(self, x: tensor) -> list[tensor]:
		x = self.fc(x)
		return [
			self.output_function(self.output_layers[i](x)) for i in range(len(self.output_layers))
		]
