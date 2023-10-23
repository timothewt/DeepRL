import gymnasium as gym
import numpy as np
import supersuit as ss
from torch import cuda, device

from algorithms.dqn import DQN
from algorithms.ppo import PPO, PPOContinuous, PPOMasked
from algorithms.a2c import A2C, A2CContinuous, A2CMasked
from envs.airport import AirportEnv

if __name__ == "__main__":

	algo = PPOMasked(config={
		# "env_fn": lambda: gym.make("CartPole-v1"),
		# "env_fn": lambda: gym.make("Pendulum-v1"),
		# "env_fn": lambda: MinesweeperEnv(grid_width=4, grid_height=4),
		# "env_fn": lambda: SnakeEnv(),
		"env_fn": lambda: AirportEnv(agents_nb=5),
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
		"horizon": 256,
		"ent_coef": .01,
		"minibatch_size": 64,
	})

	# algo.train(30_000, True)
	#
	algo.load_models(r"C:\Users\AT82790\Documents\DeepRL\saved_models\PPOMaskedAirportEnv_v0_23-10-23_13h12m17")

	env = ss.pettingzoo_env_to_vec_env_v1(AirportEnv(agents_nb=5, render_mode="human"))
	obs, infos = env.reset()

	while True:
		actions = algo.compute_single_action(obs, infos)
		obs, _, _, _, infos = env.step(actions)
