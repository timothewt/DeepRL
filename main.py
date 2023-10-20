import gymnasium as gym

from algorithms.dqn import DQN, DQNMasked
from algorithms.ppo import PPO
from envs.snake import SnakeEnv

if __name__ == "__main__":

	algo = DQNMasked(config={
		# "env_fn": lambda: gym.make("CartPole-v0"),
		"env_fn": lambda: SnakeEnv(),
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
		"minibatch_size": 32,
	})

	algo.train(max_steps=25_000)
