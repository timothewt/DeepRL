import gymnasium as gym
import numpy as np
from torch import device, cuda

from algorithms.A2C import A2C
from algorithms.PPO import PPO
from algorithms.DQN import DQN
from envs.minesweeper.MinesweeperEnv import MinesweeperEnv
from envs.snake.SnakeEnv import SnakeEnv

if __name__ == "__main__":

	algo = PPO(config={
		# "env_fn": lambda: gym.make(env_name),
		# "env_fn": lambda: MinesweeperEnv(grid_width=4, grid_height=4),
		"env_fn": lambda: SnakeEnv(),
		"num_envs": 16,
		# "device": device("cuda:0" if cuda.is_available() else "cpu"),
		"actor_hidden_layers_nb": 3,
		"actor_hidden_size": 64,
		"critic_hidden_layers_nb": 3,
		"critic_hidden_size": 64,
		"gamma": .999,
		"gae_lambda": .95,
		"actor_lr": .00015,
		"critic_lr": .0005,
		"horizon": 128,
		"ent_coef": .01,
		"minibatch_size": 32,
	})

	algo.train(max_steps=20_000)
