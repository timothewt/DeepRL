import torch

from algorithms.A2C import A2C
from algorithms.DQN import DQN
from algorithms.REINFORCE import REINFORCE


if __name__ == "__main__":
	env_name = "CartPole-v1"

	algo = A2C(config={
		"env_name": env_name,
		"num_envs": 4,
		"device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
		"actor_hidden_layers_nb": 2,
		"actor_hidden_size": 64,
		"critic_hidden_layers_nb": 2,
		"critic_hidden_size": 64,
		"gamma": .99,
		"actor_lr": .0003,
		"critic_lr": .0008,
		"log_freq": 50,
		"t_max": 5,
		"ent_coef": .001,
	})

	algo.train(max_steps=50_000, plot_training_stats=True)

	algo.env.close()
