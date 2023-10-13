from torch import device, cuda

from algorithms.A2C import A2C
from algorithms.PPO import PPO
from algorithms.DQN import DQN
from algorithms.REINFORCE import REINFORCE


if __name__ == "__main__":
	env_name = "BipedalWalker-v3"

	algo = PPO(config={
		"env_name": env_name,
		"num_envs": 8,
		# "device": device("cuda:0" if cuda.is_available() else "cpu"),
		"actor_hidden_layers_nb": 3,
		"actor_hidden_size": 256,
		"critic_hidden_layers_nb": 3,
		"critic_hidden_size": 256,
		"gamma": .999,
		"gae_lambda": .95,
		"actor_lr": .0003,
		"critic_lr": .0006,
		"horizon": 2048,
		"ent_coef": .01,
		"minibatch_size": 64,
		"log_freq": 5,
	})

	algo.train(max_steps=50_000, plot_training_stats=True)

	algo.env.close()
