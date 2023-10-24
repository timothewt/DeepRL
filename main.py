import gymnasium as gym

from algorithms.ppo import PPO, PPOMasked, PPOMultiDiscrete
from envs.airport import AirportEnv
from envs.gridworld import GridWorldEnv

if __name__ == "__main__":

	algo = PPOMultiDiscrete(config={
		# "env_fn": lambda: gym.make("CartPole-v1"),
		# "env_fn": lambda: gym.make("Pendulum-v1"),
		# "env_fn": lambda: MinesweeperEnv(grid_width=4, grid_height=4),
		# "env_fn": lambda: SnakeEnv(),
		# "env_fn": lambda: AirportEnv(agents_nb=5),
		"env_fn": lambda: GridWorldEnv(width=10, height=10),
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
	#
	# algo.load_models(r"C:\Users\AT82790\Documents\DeepRL\saved_models\PPOMaskedAirportEnv_v0_24-10-23_08h09m25")
	#
	# algo.train(max_steps=20_000, save_models=True, save_freq=30_000)
	#
	# algo.load_models(r"C:\Users\AT82790\Documents\DeepRL\saved_models\PPO-GridWorldEnv-v0-24-10-23_11h40m33")
	#
	# env = ss.pettingzoo_env_to_vec_env_v1(AirportEnv(agents_nb=5, render_mode="human"))
	# env = gym.make("CartPole-v1", render_mode="human")
	# env = GridWorldEnv(width=10, height=10, render_mode="human")
	# obs, infos = env.reset()
	#
	# while True:
	# 	actions = algo.compute_single_action(obs, infos)
	# 	obs, _, term, trunc, infos = env.step(actions)
	# 	if term or trunc:
	# 		obs, infos = env.reset()
