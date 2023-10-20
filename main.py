import gymnasium as gym
from algorithms.ppo import PPO, PPOContinuous, PPOMasked
from algorithms.a2c import A2C, A2CContinuous, A2CMasked
from pettingzoo.sisl import waterworld_v4

if __name__ == "__main__":

	algo = A2CContinuous(config={
		# "env_fn": lambda: gym.make("CartPole-v1"),
		"env_fn": lambda: gym.make("Pendulum-v1"),
		# "env_fn": lambda: MinesweeperEnv(grid_width=4, grid_height=4),
		# "env_fn": lambda: SnakeEnv(),
		# "env_fn": lambda: waterworld_v4.parallel_env(),
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
		"horizon": 1024,
		"ent_coef": .01,
		"minibatch_size": 256,
	})

	algo.train(max_steps=50_000)
