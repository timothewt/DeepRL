import gymnasium as gym
import numpy as np
import supersuit as ss
from torch import cuda, device

from algorithms.dqn import DQN
from algorithms.ppo import PPO, PPOContinuous, PPOMasked
from algorithms.a2c import A2C, A2CContinuous, A2CMasked


if __name__ == "__main__":

	algo = PPOContinuous(config={
		# "env_fn": lambda: gym.make("CartPole-v1"),
		"env_fn": lambda: gym.make("Pendulum-v1"),
		# "env_fn": lambda: MinesweeperEnv(grid_width=4, grid_height=4),
		# "env_fn": lambda: SnakeEnv(),
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
	algo.load_models(r"C:\Users\AT82790\Documents\DeepRL\saved_models\PPOContinuous_env__23-10-23_13h29m06")

	env = gym.make("Pendulum-v1", render_mode="human")
	obs, infos = env.reset()

	while True:
		actions = algo.compute_single_action(obs, infos)
		obs, _, _, _, infos = env.step(actions)
