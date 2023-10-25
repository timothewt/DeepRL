import gymnasium as gym
import numpy as np
import supersuit as ss

from algorithms.ppo import PPO, PPOMasked, PPOMultiDiscrete, PPOMultiDiscreteMasked
from envs.airport import AirportEnv
from envs.gridworld import GridWorldEnv, GridWorldMAEnv
from envs.snake import SnakeEnv

if __name__ == "__main__":

	algo = PPOMultiDiscreteMasked(config={
		# "env_fn": lambda: gym.make("CartPole-v1"),
		# "env_fn": lambda: gym.make("Pendulum-v1"),
		# "env_fn": lambda: MinesweeperEnv(grid_width=4, grid_height=4),
		# "env_fn": lambda: SnakeEnv(),
		# "env_fn": lambda: AirportEnv(agents_nb=5),
		"env_fn": lambda: GridWorldEnv(width=20, height=20),
		"num_envs": 8,
		# "device": device("cuda:0" if cuda.is_available() else "cpu"),
		"actor_hidden_layers_nb": 2,
		"actor_hidden_size": 64,
		"critic_hidden_layers_nb": 2,
		"critic_hidden_size": 64,
		"gamma": .999,
		"gae_lambda": .95,
		"actor_lr": .00015,
		"critic_lr": .0005,
		"horizon": 128,
		"ent_coef": .01,
		"minibatch_size": 64,
	})

	# algo.load_models(r"C:\Users\AT82790\Documents\DeepRL\saved_models\PPO-GridWorldEnv-v0-25-10-23_09h06m13")

	algo.train(max_steps=25_000, save_models=True, checkpoints=False)

	# algo.load_models(r"C:\Users\AT82790\Documents\DeepRL\saved_models\PPO-GridWorldMAEnv-v0-24-10-23_13h55m00")
	# #
	# # env = ss.pettingzoo_env_to_vec_env_v1(AirportEnv(agents_nb=5, render_mode="human"))
	# # env = gym.make("CartPole-v1", render_mode="human")
	# env = GridWorldEnv(width=20, height=20, render_mode="human")
	# # env = ss.pettingzoo_env_to_vec_env_v1(GridWorldMAEnv(width=20, height=20, agents_nb=4, render_mode="human"))
	# # env = SnakeEnv(render_mode="human")
	# obs, infos = env.reset()
	#
	# while True:
	# 	actions = algo.compute_single_action(obs, infos)
	# 	obs, _, term, trunc, infos = env.step(actions)
	# 	if term or trunc:
	# 		obs, infos = env.reset()
