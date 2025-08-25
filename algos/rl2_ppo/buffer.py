import torch
from gymnasium.spaces import Box, Discrete


class RolloutBuffer:
    def __init__(
        self,
        size,
        obs_space,
        action_space,
        gamma=0.99,
        gae_lambda=0.95,
        device=None,
    ):
        self.device = device
        obs_dim = obs_space.shape[0]

        if isinstance(action_space, Box):
            action_dim = action_space.shape[0]
            action_dim = (size, action_dim)

        elif isinstance(action_space, Discrete):
            action_dim = action_space.n
            action_dim = size

        self.data = {
            "obs": torch.zeros((size, obs_dim)).to(device),
            "action": torch.zeros(action_dim).to(device),
            "advantage": torch.zeros((size)).to(device),
            "reward": torch.zeros((size)).to(device),
            "return": torch.zeros((size)).to(device),
            "value": torch.zeros((size)).to(device),
            "prev_reward": torch.zeros((size)).to(device),
            "prev_action": torch.zeros((size)).to(device),
            "log_prob": torch.zeros((size)).to(device),
            "termination": torch.zeros((size)).to(device),
        }
        self.episodes = []

        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.max_size, self.size = size, 0

    def store(
        self,
        obs,
        action,
        reward,
        prev_action,
        prev_reward,
        term,
        value,
        log_prob,
    ):
        assert self.size <= self.max_size
        self.data["obs"][self.size] = obs
        self.data["action"][self.size] = action
        self.data["reward"][self.size] = torch.tensor(reward).to(self.device)
        self.data["termination"][self.size] = torch.tensor(term).to(self.device)
        self.data["prev_action"][self.size] = prev_action
        self.data["prev_reward"][self.size] = prev_reward
        self.data["value"][self.size] = value
        self.data["log_prob"][self.size] = log_prob
        self.size += 1

    def reset(self):
        self.empty_buffer()
        self.episodes = []

    def empty_buffer(self):
        for key, val in self.data.items():
            self.data[key] = torch.zeros_like(val)
        self.size = 0

    def finish_path(self, last_value, last_termination):

        # Calculate advantages
        prev_advantage = 0
        for step in reversed(range(self.max_size)):

            # The "value" argument should be 0 if the trajectory ended
            # because the agent reached a terminal state (died).
            if step == self.max_size - 1:
                next_non_terminal = 1.0 - last_termination
                next_value = last_value
            else:
                # Otherwise it should be V(s_t), the value function estimated for the
                # last state. This allows us to bootstrap the reward-to-go calculation
                # to account. for timesteps beyond the arbitrary episode horizon.
                next_non_terminal = 1.0 - self.data["termination"][step + 1]
                next_value = self.data["value"][step + 1]

            delta = (
                self.data["reward"][step]
                + self.gamma * next_value * next_non_terminal
                - self.data["value"][step]
            )
            self.data["advantage"][step] = (
                delta
                + self.gamma
                * self.gae_lambda
                * next_non_terminal
                * prev_advantage
            )
            prev_advantage = self.data["advantage"][step]

        self.data["return"] = self.data["advantage"] + self.data["value"]

        # Add to episode list
        episode = {k: v.clone() for k, v in self.data.items()}
        self.episodes.append(episode)

        # Empty episode buffer
        self.empty_buffer()

    def get(self):
        # format the experience to (batch_size, horizon, ...) length
        batch = {
            k: torch.stack([ep[k] for ep in self.episodes])
            for k in self.data.keys()
        }

        return batch, len(self.episodes)


class MyRolloutBuffer:
    def __init__(
        self,
        size,
        obs_space,
        action_space,
        gamma=0.99,
        gae_lambda=0.95,
        device=None,
    ):
        self.device = device
        obs_dim = obs_space.shape

        if isinstance(action_space, Box):
            action_dim = action_space.shape
            action_dim = (size, *action_dim)

        elif isinstance(action_space, Discrete):
            action_dim = action_space.n
            action_dim = size
        self.data = {
            "obs": torch.zeros((size, *obs_dim)).to(device),
            "action": torch.zeros(action_dim).to(device),
            "advantage": torch.zeros((size)).to(device),
            "reward": torch.zeros((size)).to(device),
            "return": torch.zeros((size)).to(device),
            "value": torch.zeros((size)).to(device),
            "prev_reward": torch.zeros((size, *(1, 1))).to(device),
            "prev_action": torch.zeros(action_dim).to(device),
            "log_prob": torch.zeros((size)).to(device),
            "termination": torch.zeros((size)).to(device),
        }
        self.episodes = []

        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.max_size, self.size = size, 0

    def store(
        self,
        obs,
        action,
        reward,
        prev_action,
        prev_reward,
        term,
        value,
        log_prob,
    ):
        assert self.size <= self.max_size
        self.data["obs"][self.size] = obs
        self.data["action"][self.size] = action
        self.data["reward"][self.size] = torch.tensor(reward).to(self.device)
        self.data["termination"][self.size] = torch.tensor(term).to(self.device)
        self.data["prev_action"][self.size] = prev_action
        self.data["prev_reward"][self.size] = prev_reward
        self.data["value"][self.size] = value
        self.data["log_prob"][self.size] = log_prob
        self.size += 1

    def reset(self):
        self.empty_buffer()
        self.episodes = []

    def empty_buffer(self):
        for key, val in self.data.items():
            self.data[key] = torch.zeros_like(val)
        self.size = 0

    def finish_path(self, last_value, last_termination):

        # Calculate advantages
        prev_advantage = 0
        for step in reversed(range(self.max_size)):

            # The "value" argument should be 0 if the trajectory ended
            # because the agent reached a terminal state (died).
            if step == self.max_size - 1:
                next_non_terminal = ~last_termination
                next_value = last_value
            else:
                # Otherwise it should be V(s_t), the value function estimated for the
                # last state. This allows us to bootstrap the reward-to-go calculation
                # to account. for timesteps beyond the arbitrary episode horizon.
                next_non_terminal = 1.0 - self.data["termination"][step + 1]
                next_value = self.data["value"][step + 1]

            delta = (
                self.data["reward"][step]
                + self.gamma * next_value * next_non_terminal
                - self.data["value"][step]
            )
            self.data["advantage"][step] = (
                delta
                + self.gamma
                * self.gae_lambda
                * next_non_terminal
                * prev_advantage
            )
            prev_advantage = self.data["advantage"][step]

        self.data["return"] = self.data["advantage"] + self.data["value"]

        # Add to episode list
        episode = {k: v.clone() for k, v in self.data.items()}
        self.episodes.append(episode)

        # Empty episode buffer
        self.empty_buffer()

    def get(self):
        # format the experience to (batch_size, horizon, ...) length
        batch = {
            k: torch.stack([ep[k] for ep in self.episodes])
            for k in self.data.keys()
        }

        return batch, len(self.episodes)


# 支持n_envs>1的buffer
class VecEnvRolloutBuffer:
    def __init__(
        self,
        size,           # 每个环境的轨迹长度
        obs_space,
        action_space,
        gamma=0.99,
        gae_lambda=0.95,
        device=None,
    ):
        self.device = device
        n_envs = obs_space.shape[0]
        obs_dim = obs_space.shape[1]

        self.n_envs = n_envs
        self.max_size = size
        self.size = 0

        # obs和action维度一般大于1
        if isinstance(action_space, Box):
            action_dim = action_space.shape[1]
            self.data = {
                "obs": torch.zeros((size, n_envs, obs_dim), device=device),
                "action": torch.zeros((size, n_envs, action_dim), device=device),
            }
        elif isinstance(action_space, Discrete):
            self.data = {
                "obs": torch.zeros((size, n_envs, obs_dim), device=device),
                "action": torch.zeros((size, n_envs), device=device),
            }

        # 其它变量, dim维度一般为1
        self.data.update({
            "advantage": torch.zeros((size, n_envs), device=device),
            "reward": torch.zeros((size, n_envs), device=device),
            "return": torch.zeros((size, n_envs), device=device),
            "value": torch.zeros((size, n_envs), device=device),
            "prev_reward": torch.zeros((size, n_envs, 1), device=device),  # for rnn input
            "prev_action": torch.zeros((size, n_envs, action_dim), device=device) if isinstance(action_space, Box) else torch.zeros((size, n_envs), device=device),
            "log_prob": torch.zeros((size, n_envs), device=device),
            "termination": torch.zeros((size, n_envs), device=device),
        })
        self.episodes = []

        self.gamma, self.gae_lambda = gamma, gae_lambda

    def store(
        self,
        obs,            # [n_envs, obs_dim]
        action,         # [n_envs, action_dim] or [n_envs]
        reward,         # [n_envs]
        prev_action,    # [n_envs, action_dim] or [n_envs]
        prev_reward,    # [n_envs, 1]
        term,           # [n_envs]
        value,          # [n_envs]
        log_prob,       # [n_envs]
    ):
        assert self.size < self.max_size
        self.data["obs"][self.size] = obs
        self.data["action"][self.size] = action
        self.data["reward"][self.size] = torch.tensor(reward).to(self.device)
        self.data["termination"][self.size] = torch.tensor(term).to(self.device)
        self.data["prev_action"][self.size] = prev_action
        self.data["prev_reward"][self.size] = prev_reward
        self.data["value"][self.size] = value
        self.data["log_prob"][self.size] = log_prob
        self.size += 1

    def reset(self):
        self.empty_buffer()
        self.episodes = []

    def empty_buffer(self):
        for key, val in self.data.items():
            self.data[key].zero_()
        self.size = 0

    def finish_path(self, last_value, last_termination):
        # last_value, last_termination shape: [n_envs]
        # 按每个环境分别计算 GAE
        for env in range(self.n_envs):
            prev_advantage = 0
            for step in reversed(range(self.max_size)):
                if step == self.max_size - 1:
                    next_non_terminal = 1.0 - float(last_termination[env])
                    next_value = last_value[env]
                else:
                    next_non_terminal = 1.0 - self.data["termination"][step + 1, env].float()
                    next_value = self.data["value"][step + 1, env]

                delta = (
                    self.data["reward"][step, env]
                    + self.gamma * next_value * next_non_terminal
                    - self.data["value"][step, env]
                )
                self.data["advantage"][step, env] = (
                    delta
                    + self.gamma
                    * self.gae_lambda
                    * next_non_terminal
                    * prev_advantage
                )
                prev_advantage = self.data["advantage"][step, env]

        self.data["return"] = self.data["advantage"] + self.data["value"]

        # 保存整个 episode
        episode = {k: v.clone() for k, v in self.data.items()}
        self.episodes.append(episode)

        self.empty_buffer()

    def get(self):
        # 最终返回 shape: [num_episodes*n_envs, size, ...] 和 num_episodes*n_envs
        ## stack 后 shape: [num_episodes, size, n_envs, ...]
        batch = {
            k: torch.stack([ep[k] for ep in self.episodes])
            for k in self.data.keys()
        }
        ## 转换为 [num_episodes, n_envs, size, ...]
        batch = {
            k: v.permute(0, 2, 1, *range(3, v.ndim)) if v.ndim >= 3 else v
            for k, v in batch.items()
        }
        # return batch, len(self.episodes)
        # 展平为 [num_episodes * n_envs, size, ...]
        batch = {
            k: v.reshape(-1, *v.shape[2:]) if v.ndim >= 3 else v.reshape(-1)
            for k, v in batch.items()
        }
        return batch, batch["obs"].shape[0]
