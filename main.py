import gymnasium as gym
from torch import device, cuda

from algorithms.A2C import A2C
from algorithms.PPO import PPO
from algorithms.DQN import DQN
from envs.airport.AirportEnv import AirportEnv
from envs.minesweeper.MinesweeperEnv import MinesweeperEnv

if __name__ == "__main__":

	algo = PPO(config={
		# "env_fn": lambda: gym.make("CartPole-v1"),
		# "env_fn": lambda: AirportEnv(airport_icao="CYTZ", agents_nb=4, predetermined_path_agents_nb=0),
		"env_fn": lambda: MinesweeperEnv(),
		"num_envs": 4,
		"device": device("cuda:0" if cuda.is_available() else "cpu"),
		"actor_hidden_layers_nb": 3,
		"actor_hidden_size": 256,
		"critic_hidden_layers_nb": 3,
		"critic_hidden_size": 256,
		"gamma": .999,
		"gae_lambda": .95,
		"actor_lr": .00005,
		"critic_lr": .00015,
		"horizon": 256,
		"ent_coef": .01,
		"minibatch_size": 32,
	})

	algo.train(max_steps=500_000)
