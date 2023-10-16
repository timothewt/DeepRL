from torch import device, cuda

from algorithms.A2C import A2C
from algorithms.PPO import PPO
from algorithms.DQN import DQN


if __name__ == "__main__":
	env_name = "BipedalWalker-v3"

	algo = PPO(config={
		"env_name": env_name,
		"num_envs": 16,
		"device": device("cuda:0" if cuda.is_available() else "cpu"),
		"actor_hidden_layers_nb": 3,
		"actor_hidden_size": 64,
		"critic_hidden_layers_nb": 3,
		"critic_hidden_size": 64,
		"gamma": .999,
		"gae_lambda": .95,
		"actor_lr": .00015,
		"critic_lr": .0005,
		"horizon": 1024,
		"ent_coef": .01,
		"minibatch_size": 32,
	})

	algo.train(max_steps=20_000)

	if isinstance(algo, PPO) or isinstance(algo, A2C):
		algo.envs.close()
