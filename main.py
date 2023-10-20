import gymnasium as gym
import numpy as np
from torch import device, cuda

from algorithms.A2C import A2C
from algorithms.PPO import PPO
from algorithms.DQN import DQN
from algorithms.PPOContinuous import PPOContinuous
from algorithms.PPOMasked import PPOMasked
from envs.airport.AirportEnv import AirportEnv
from envs.minesweeper.MinesweeperEnv import MinesweeperEnv
from envs.snake.SnakeEnv import SnakeEnv

if __name__ == "__main__":

	algo = PPOMasked(config={
		# "env_fn": lambda: gym.make("Pendulum-v1"),
		# "env_fn": lambda: MinesweeperEnv(grid_width=4, grid_height=4),
		"env_fn": lambda: SnakeEnv(),
		# "env_fn": lambda: AirportEnv(airport_icao="CYTZ", agents_nb=3, predetermined_path_agents_nb=0),
		"num_envs": 8,
		# "device": device("cuda:0" if cuda.is_available() else "cpu"),
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
		"minibatch_size": 128,
	})

	algo.train(max_steps=50_000)
