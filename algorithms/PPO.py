from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch import nn, tensor
from torch.distributions import Categorical, Normal

from algorithms.Algorithm import Algorithm
from models.ActorContinuous import ActorContinuous
from models.FCNet import FCNet


class Buffer:
    def __init__(
            self,
            num_envs: int,
            max_len: int = 5,
            state_shape: tuple[int] = (1,),
            device: torch.device = torch.device("cpu"),
    ):
        self.states = torch.empty((max_len, num_envs) + state_shape, device=device)
        self.next_states = torch.empty((max_len, num_envs) + state_shape, device=device)
        self.dones = torch.empty((max_len, num_envs, 1), device=device)
        self.actions = torch.empty((max_len, num_envs, 1), device=device)
        self.rewards = torch.empty((max_len, num_envs, 1), device=device)
        self.values = torch.empty((max_len, num_envs, 1), device=device)
        self.log_probs = torch.empty((max_len, num_envs, 1), device=device)

        self.num_envs = num_envs
        self.state_shape = state_shape
        self.max_len = max_len
        self.device = device
        self.i = 0

    def is_full(self) -> bool:
        return self.i == self.max_len

    def push(
            self,
            states: tensor,
            next_states: tensor,
            dones: tensor,
            actions: tensor,
            rewards: tensor,
            values: tensor,
            log_probs: tensor
    ) -> None:
        assert self.i < self.max_len, "Buffer is full!"

        self.states[self.i] = states
        self.next_states[self.i] = next_states
        self.dones[self.i] = dones
        self.actions[self.i] = actions
        self.rewards[self.i] = rewards
        self.values[self.i] = values
        self.log_probs[self.i] = log_probs

        self.i += 1

    def get_all(self) -> tuple[tensor, tensor, tensor, tensor, tensor, tensor, tensor]:
        return self.states, self.next_states, self.dones, self.actions, self.rewards, self.values, self.log_probs

    def get_all_flattened(self) -> tuple[tensor, tensor, tensor, tensor, tensor, tensor, tensor]:
        return self.states.view((self.max_len * self.num_envs,) + self.state_shape), \
            self.next_states.view((self.max_len * self.num_envs,) + self.state_shape), \
            self.dones.flatten(), \
            self.actions.flatten(), \
            self.rewards.flatten(), \
            self.values.flatten(), \
            self.log_probs.flatten()

    def reset(self) -> None:
        self.values = torch.empty((self.max_len, self.num_envs, 1), device=self.device)
        self.log_probs = torch.empty((self.max_len, self.num_envs, 1), device=self.device)
        self.i = 0


class PPO(Algorithm):

    def __init__(self, config: dict[str: Any]):
        """
		:param config:
			num_envs: number of environments in parallel
			actor_lr: learning rate of the actor
			critic_lr: learning rate of the critic
			gamma: discount factor
			horizon: steps between each updates
		"""
        super().__init__(config=config)

        # Vectorized envs

        self.num_envs = max(config.get("num_envs", 1), 1)
        self.envs: gym.experimental.vector.VectorEnv = gym.make_vec(config.get("env_name", None),
                                                                    num_envs=self.num_envs)

        # Stats

        self.actor_losses = []
        self.critic_losses = []
        self.entropy = []

        # Algorithm hyperparameters

        self.actor_lr: float = config.get("actor_lr", .0002)
        self.critic_lr: float = config.get("critic_lr", .0002)
        self.gamma: float = config.get("gamma", .99)
        self.lam: float = config.get("lambda", .95)
        self.horizon: int = config.get("horizon", 5)
        self.num_epochs: int = config.get("num_epochs", 5)
        self.ent_coef: float = config.get("ent_coef", .001)
        self.vf_coef = config.get("vf_coef", .5)
        self.eps = config.get("eps", .2)

        self.batch_size = self.horizon * self.num_envs
        self.minibatch_size = config.get("minibatch_size", self.batch_size)
        assert self.batch_size % self.minibatch_size == 0, \
            "Batch size (num_envs * horizon) must be a multiple of mini-batch size!"
        self.minibatch_nb_per_batch = self.batch_size // self.minibatch_size

        self.mse = nn.MSELoss()
        self.grad_clip = .5

        # Policy

        # Observations
        self.env_obs_space = self.envs.single_observation_space
        self.flat_env_obs_space = gym.spaces.utils.flatten_space(self.env_obs_space)
        input_size = int(np.prod(self.flat_env_obs_space.shape))

        # Actions
        self.env_act_space = self.envs.single_action_space

        output_size = 1
        output_function = None
        if isinstance(self.env_act_space, spaces.Discrete):
            self.actions_type = "discrete"
            output_size = self.env_act_space.n
            output_function = nn.Softmax(dim=-1)
        elif isinstance(self.env_act_space, spaces.Box):
            assert np.prod(self.env_act_space.shape) == 1, "Only single continuous action currently supported"
            self.action_space_low, self.action_space_high = self.env_act_space.low[0], self.env_act_space.high[0]
            self.actions_type = "continuous"

        actor_config = {
            "input_size": input_size,
            "output_size": output_size,
            "hidden_layers_nb": config.get("actor_hidden_layers_nb", 3),
            "hidden_size": config.get("actor_hidden_size", 32),
            "output_function": output_function,
            "output_layer_std": .01,
        }

        if self.actions_type == "continuous":
            self.actor: nn.Module = ActorContinuous(config=actor_config).to(self.device)
        else:
            self.actor: nn.Module = FCNet(config=actor_config).to(self.device)
        self.actor_optimizer: torch.optim.Optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        critic_config = {
            "input_size": input_size,
            "output_size": 1,
            "hidden_layers_nb": config.get("critic_hidden_layers_nb", 3),
            "hidden_size": config.get("critic_hidden_size", 32),
            "output_layer_std": 1,
        }
        self.critic: nn.Module = FCNet(config=critic_config).to(self.device)
        self.critic_optimizer: torch.optim.Optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def train(self, max_steps: int, plot_training_stats: bool = False) -> None:
        # From https://arxiv.org/pdf/1707.06347.pdf and https://arxiv.org/pdf/2205.09123.pdf

        self.rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropy = []

        steps = 0
        episode = 0

        buffer = Buffer(self.num_envs, self.horizon, self.env.observation_space.shape, self.device)

        print("==== STARTING TRAINING ====")

        obs, _ = self.envs.reset()
        obs = torch.from_numpy(obs).to(self.device)
        first_agent_rewards = 0

        while steps <= max_steps:
            actor_output = self.actor(obs)
            critic_output = self.critic(obs)  # value function

            if self.actions_type == "continuous":
                means, std = actor_output
                dist = Normal(loc=means, scale=std)
                actions = dist.sample()
                actions_to_input = self._scale_to_action_space(actions).cpu().numpy()
                log_probs = dist.log_prob(actions)
            else:
                probs = actor_output
                dist = Categorical(probs=probs)
                actions = dist.sample()
                actions_to_input = actions.cpu().numpy()
                log_probs = dist.log_prob(actions).unsqueeze(1)
                actions = actions.unsqueeze(1)

            new_obs, rewards, dones, truncateds, _ = self.envs.step(actions_to_input)
            dones = dones + truncateds  # done or truncate
            new_obs = torch.from_numpy(new_obs).to(self.device)

            buffer.push(
                obs,
                new_obs,
                torch.from_numpy(dones).to(self.device).unsqueeze(1),
                actions,
                torch.from_numpy(rewards).to(self.device).unsqueeze(1),
                critic_output,
                log_probs,
            )

            obs = new_obs

            if buffer.is_full():
                self._update_networks(buffer)
                buffer.reset()

            first_agent_rewards += rewards[0]
            if dones[0]:
                self.rewards.append(first_agent_rewards)
                first_agent_rewards = 0
                if episode % self.log_freq == 0:
                    self.log_rewards(episode=episode, avg_period=10)
                episode += 1
            steps += 1

        print("==== TRAINING COMPLETE ====")

        if plot_training_stats:
            self.plot_training_stats([
                ("Reward", "Episode", self.rewards),
                ("Actor loss", "Update step", self.actor_losses),
                ("Critic loss", "Update step", self.critic_losses),
                ("Entropy", "Update step", self.entropy),
            ], n=max_steps // 1000)

    def _update_networks(self, buffer: Buffer) -> None:
        states, _, _, actions, rewards, values, old_log_probs = buffer.get_all_flattened()
        values, old_log_probs = values.detach(), old_log_probs.detach()

        advantages = self._compute_advantages(buffer, self.gamma, self.lam).flatten()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values

        for _ in range(self.num_epochs):
            indices = torch.randperm(self.batch_size)
            for m in range(self.minibatch_nb_per_batch):
                start = m * self.minibatch_size
                end = start + self.minibatch_size
                minibatch_indices = indices[start:end]

                actor_output = self.actor(states[minibatch_indices])
                if self.actions_type == "continuous":
                    means, stds = actor_output
                    means, stds = means.view(self.minibatch_size), stds.view(self.minibatch_size)
                    new_dist = Normal(loc=means, scale=stds)
                else:
                    new_dist = Categorical(probs=actor_output)

                new_log_probs = new_dist.log_prob(actions[minibatch_indices])
                new_entropy = new_dist.entropy()
                new_values = self.critic(states[minibatch_indices])

                r = torch.exp(new_log_probs - old_log_probs[minibatch_indices])
                L_clip = torch.min(
                    r * advantages[minibatch_indices],
                    torch.clamp(r, 1 - self.eps, 1 + self.eps) * advantages[minibatch_indices]
                ).mean()
                L_vf = self.mse(new_values.squeeze(1), returns[minibatch_indices])
                L_S = new_entropy.mean()

                # Updating the network
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                (- L_clip + self.vf_coef * L_vf - self.ent_coef * L_S).backward()
                # nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
                # nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                self.actor_losses.append(L_clip.item())
                self.critic_losses.append(L_vf.item())
                self.entropy.append(L_S.item())

    def _compute_advantages(self, buffer: Buffer, gamma: float, lam: float) -> tensor:
        _, next_states, dones, _, rewards, values, _ = buffer.get_all()

        next_values = values.roll(-1, dims=0)
        next_values[-1] = self.critic(next_states[-1])

        deltas = (rewards + gamma * next_values - values).detach()

        advantages = torch.zeros(deltas.shape, device=self.device)
        last_advantage = advantages[-1]
        next_step_terminates = dones[-1]  # should be the dones of the next step however cannot reach it
        for t in reversed(range(buffer.max_len)):
            advantages[t] = last_advantage = deltas[t] + gamma * lam * last_advantage * (1 - next_step_terminates)
            next_step_terminates = dones[t]

        return advantages

    def _scale_to_action_space(self, actions: tensor) -> tensor:
        actions = torch.clamp(actions, 0, 1)
        actions = actions * (self.action_space_high - self.action_space_low) + self.action_space_low
        return actions
