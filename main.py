import gymnasium as gym
import numpy as np
from torch import device, cuda

from algorithms.A2C import A2C
from algorithms.PPO import PPO
from algorithms.DQN import DQN

if __name__ == "__main__":
	gym.register("Minesweeper-v0", entry_point='envs.minesweeper.MinesweeperEnv:MinesweeperEnv')

	env_name = "Minesweeper-v0"

	algo = PPO(config={
		"env_name": env_name,
		"num_envs": 16,
		# "device": device("cuda:0" if cuda.is_available() else "cpu"),
		"actor_hidden_layers_nb": 3,
		"actor_hidden_size": 128,
		"critic_hidden_layers_nb": 3,
		"critic_hidden_size": 128,
		"gamma": .999,
		"gae_lambda": .95,
		"actor_lr": .00015,
		"critic_lr": .0005,
		"horizon": 256,
		"ent_coef": .01,
		"minibatch_size": 32,
		"env_uses_action_mask": True,
		"env_kwargs": {
			"grid_width": 9,
			"grid_height": 9,
			"render_mode": "ansi",
		}
	})

	algo.train(max_steps=50_000)
