from torch import device, cuda

from algorithms.A2C import A2C
from algorithms.DQN import DQN
from algorithms.REINFORCE import REINFORCE


if __name__ == "__main__":
	env_name = "CartPole-v1"

	algo = A2C(config={
		"env_name": env_name,
		"num_envs": 8,
		"device": device("cuda:0" if cuda.is_available() else "cpu"),
		"actor_hidden_layers_nb": 2,
		"actor_hidden_size": 64,
		"critic_hidden_layers_nb": 2,
		"critic_hidden_size": 64,
		"gamma": .999,
		"actor_lr": .0003,
		"critic_lr": .001,
		"log_freq": 10,
		"t_max": 5,
		"ent_coef": .001,
	})

	algo.train(max_steps=50_000, plot_training_stats=True)

	algo.env.close()
