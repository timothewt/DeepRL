from torch import device, cuda

from algorithms.A2C import A2C
from algorithms.PPO import PPO
from algorithms.DQN import DQN
from algorithms.REINFORCE import REINFORCE


if __name__ == "__main__":
	env_name = "CartPole-v1"

	algo = PPO(config={
		"env_name": env_name,
		"num_envs": 8,
		# "device": device("cuda:0" if cuda.is_available() else "cpu"),
		"actor_hidden_layers_nb": 2,
		"actor_hidden_size": 32,
		"critic_hidden_layers_nb": 2,
		"critic_hidden_size": 32,
		"gamma": .999,
		"lambda": .95,
		"actor_lr": .00015,
		"critic_lr": .0005,
		"horizon": 256,
		"ent_coef": .01,
		"minibatch_size": 16,
		"log_freq": 10,
	})

	algo.train(max_steps=50_000, plot_training_stats=True)

	algo.env.close()
