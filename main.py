import gymnasium as gym
import torch

from algorithms.A2C import A2C
from algorithms.REINFORCE import REINFORCE
from algorithms.DQN import DQN


if __name__ == "__main__":
	env = gym.make("CartPole-v1")

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	algo = A2C(config={
		"env": env,
		"device": device,
		"actor_hidden_layers_nb": 2,
		"actor_hidden_size": 32,
		"critic_hidden_layers_nb": 2,
		"critic_hidden_size": 32,
		"gamma": .99,
		"actor_lr": .0003,
		"critic_lr": .0003,
		"log_freq": 100,
		"t_max": 5,
		"ent_coef": .0005,
	})

	algo.train(max_steps=20_000, plot_training_stats=True)

	env.close()
