import gymnasium as gym
import numpy as np
import supersuit as ss
from tqdm import tqdm

from algorithms.ppo import PPO, PPOMasked, PPOMultiDiscrete, PPOMultiDiscreteMasked
from envs.gridworld import GridWorld, GridWorldMA
from envs.snake import Snake

if __name__ == "__main__":

	algo = PPOMultiDiscreteMasked(config={
		"env_fn": lambda: GridWorldMA(agents_nb=2),
		"num_envs": 8,
		# "device": device("cuda:0" if cuda.is_available() else "cpu"),
		"actor_hidden_layers_nb": 3,
		"actor_hidden_size": 256,
		"critic_hidden_layers_nb": 3,
		"critic_hidden_size": 256,
		"gamma": .999,
		"gae_lambda": .95,
		"actor_lr": .00003,
		"critic_lr": .00008,
		"horizon": 1024,
		"ent_coef": .01,
		"minibatch_size": 256,
	})

	algo.train(max_steps=2_000_000, save_models=False, checkpoints=False, save_freq=100_000)
