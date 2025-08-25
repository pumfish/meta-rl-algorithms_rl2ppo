import logging

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

from utils.misc import make_gif
from algos.rl2_ppo.agent import PPO
from algos.rl2_ppo.buffer import RolloutBuffer, MyRolloutBuffer, VecEnvRolloutBuffer


def train_rl2_ppo(
    config,
    envs: list[gym.Env],
    test_envs: list[gym.Env] = None,
    writer: SummaryWriter = None,
):

    # Define the agent and rollout buffer
    agent = PPO(
        obs_space=envs[0].observation_space,
        action_space=envs[0].action_space,
        writer=writer,
        device=config["device"],
        ac_kwargs=config["actor_critic"],
        **config["ppo"],
    )
    buffer = RolloutBuffer(
        obs_space=envs[0].observation_space,
        action_space=envs[0].action_space,
        device=config["device"],
        size=config["max_episode_steps"],
        gae_lambda=config["ppo"]["gae_lambda"],
    )

    global_step = 0
    for meta_epoch in range(config["meta_epochs"]):

        # Sample new meta-training environment
        env = np.random.choice(envs, 1)[0]
        avg_return, avg_ep_len = [], []

        # RL^2 variables
        rnn_state = agent.actor_critic.initialize_state(batch_size=1)
        prev_action = (
            torch.tensor(env.action_space.sample())
            .to(config["device"])
            .view(-1)
        )
        prev_rew = torch.tensor(0).to(config["device"]).view(1, 1)
        ###

        # Iterate for number of episodes
        for epoch in range(config["episodes"]):
            termination, truncated = False, False
            obs, _ = env.reset()

            while not (termination or truncated):
                obs = (
                    torch.tensor(obs).to(config["device"]).float().unsqueeze(0)
                )
                action, value, log_prob, rnn_state = agent.act(
                    obs, prev_action, prev_rew, rnn_state
                )
                next_obs, rew, termination, truncated, info = env.step(
                    action.cpu().numpy()[0]
                )

                # termination
                buffer.store(
                    obs,
                    action,
                    rew,
                    prev_action,
                    prev_rew,
                    termination,
                    value,
                    log_prob,
                )

                # Update the observation
                obs = next_obs

                # Set previous action and reward tensors
                prev_action = action.detach()
                prev_rew = torch.tensor(rew).to(config["device"]).view(1, 1)

                if termination or truncated:
                    obs = (
                        torch.tensor(obs)
                        .to(config["device"])
                        .float()
                        .unsqueeze(0)
                    )
                    _, value, _, _ = agent.act(
                        obs, prev_action, prev_rew, rnn_state
                    )
                    buffer.finish_path(value, termination)

                    # Update every n episodes
                    if epoch % config["update_every_n"] == 0 and epoch != 0:
                        batch, batch_size = buffer.get()
                        agent.optimize(
                            batch,
                            config["update_epochs"],
                            batch_size,
                            global_step,
                        )
                        buffer.reset()

                    # Log final episode statistics
                    writer.add_scalar(
                        "env/ep_return", info["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "env/ep_length", info["episode"]["l"], global_step
                    )

                    avg_return.append(info["episode"]["r"])
                    avg_ep_len.append(info["episode"]["l"])

                global_step += 1

        # Store the meta-weights of the agent
        if meta_epoch % config["log_every_n"] == 0 and meta_epoch != 0:

            test_env = np.random.choice(test_envs, 1)[0]
            if (
                meta_epoch % (config["log_every_n"] * 5) == 0
                and not test_env.is_debug
            ):
                make_gif(agent, test_env, meta_epoch, config)

            # Save the weights
            if not config["debug"]:
                agent.save_weights(config["path"], meta_epoch)

            # Log test statistics
            test_return, test_ep_len = evaluate_policy(agent, test_env, config)
            writer.add_scalar("env/test_ep_return", test_return, global_step)
            writer.add_scalar("env/test_ep_length", test_ep_len, global_step)

            logging.info(
                f"meta-epoch #: {meta_epoch} "
                f"meta-train - episode return, length: ({np.mean(avg_return):.3f}, "
                f" {np.mean(avg_ep_len):.0f}) "
                f"meta-test - episode return, length: ({np.mean(test_return):.3f}, "
                f"{np.mean(test_ep_len):.0f})"
            )


def train_rl2_ppo_meta_reset(
    config,
    envs: list,  # 支持 list[GymEnv]
    test_envs: list = None,
    writer: SummaryWriter = None,
):
    """
    RL²-PPO 训练函数，适配 Domain Randomization 环境，使用 env.meta_reset() 来采样不同任务。
    """

    # -------------------------
    # 1. 初始化 agent 和 buffer
    # -------------------------
    agent = PPO(
        obs_space=envs[0].observation_space,
        action_space=envs[0].action_space,
        writer=writer,
        device=config["device"],
        ac_kwargs=config["actor_critic"],
        **config["ppo"],
    )

    buffer = MyRolloutBuffer(
        obs_space=envs[0].observation_space,
        action_space=envs[0].action_space,
        device=config["device"],
        size=config["max_episode_steps"],
        gae_lambda=config["ppo"]["gae_lambda"],
    )

    global_step = 0

    # -------------------------
    # 2. meta-training loop
    # -------------------------
    for meta_epoch in range(config["meta_epochs"]):

        # -------------------------
        # 采样新任务（使用 meta_reset）
        # -------------------------
        env = np.random.choice(envs, 1)[0]
        env.unwrapped.meta_reset()

        avg_return, avg_ep_len, avg_success = [], [], []

        # 初始化 RL² 隐状态和 prev_action/reward
        rnn_state = agent.actor_critic.initialize_state(batch_size=1)
        prev_action = (
            torch.tensor(env.action_space.sample())
            .to(config["device"])
        )
        prev_rew = torch.tensor(0.0, device=config["device"]).view(1, 1)
        # -------------------------
        # 迭代若干 episodes
        # -------------------------
        for epoch in range(config["episodes"]):
            termination, truncated = False, False
            obs, _ = env.reset()

            ep_return, ep_success = 0.0, 0
            while not (termination or truncated):
                obs_tensor = (
                    torch.tensor(obs).to(config["device"]).float()
                )

                # RL² action + value + log_prob
                action, value, log_prob, rnn_state = agent.act(
                    obs_tensor, prev_action, prev_rew, rnn_state
                )

                # 执行动作
                next_obs, reward, termination, truncated, info = env.step(
                    action
                )

                # 存储到 buffer
                buffer.store(
                    obs_tensor,
                    action,
                    reward,
                    prev_action,
                    prev_rew,
                    termination,
                    value,
                    log_prob
                )

                # 更新 obs、prev_action/reward
                obs = next_obs
                prev_action = action.detach()
                prev_rew = torch.tensor(reward).to(config["device"]).view(1, 1)

                # 从info检查是否成功
                ep_return += reward
                if "is_success" in info:
                    ep_success += int(info["is_success"])

                # episode 结束
                if termination or truncated:
                    obs_tensor = (
                        torch.tensor(obs)
                        .to(config["device"])
                        .float()
                    )
                    _, value, _, _ = agent.act(
                        obs_tensor, prev_action, prev_rew, rnn_state
                    )
                    buffer.finish_path(value, termination)

                    # 每 n episode 更新一次 PPO
                    if epoch % config["update_every_n"] == 0 and epoch != 0:
                        batch, batch_size = buffer.get()
                        agent.optimize(
                            batch,
                            config["update_epochs"],
                            batch_size,
                            global_step
                        )
                        buffer.reset()
                    avg_return.append(ep_return.cpu().numpy())
                    avg_ep_len.append(info.get("episode_length", 0))
                    avg_success.append(ep_success / max(1, info.get("episode_length", 1)))

                global_step += 1

        # -------------------------
        # 3. meta-testing & 保存模型
        # -------------------------
        if meta_epoch % config["log_every_n"] == 0 and meta_epoch != 0:
            if test_envs is not None:
                test_env = np.random.choice(test_envs, 1)[0]
                test_return, test_ep_len = meta_evaluate_policy(agent, test_env, config)
                logging.info(
                    f"[Meta-Epoch {meta_epoch}] "
                    f"Train: return={np.mean(avg_return):.2f}, length={np.mean(avg_ep_len):.1f}, success={np.mean(avg_success):.2f} | "
                    f"Test: return={test_return:.2f}, length={test_ep_len:.1f}"
                )
                if writer is not None:
                    writer.add_scalar("train/return", np.mean(avg_return), meta_epoch)
                    writer.add_scalar("train/success_rate", np.mean(avg_success), meta_epoch)
                    writer.add_scalar("test/return", test_return, meta_epoch)
                    writer.add_scalar("test/length", test_ep_len, meta_epoch)

            # 保存模型
            if not config["debug"]:
                agent.save_weights(config["path"], meta_epoch)


#TODO: debug
# Meta-RL support n_envs>1 的并行环境
def train_rl2_ppo_meta_reset_vecenv(
    config,
    envs: list,  # list[0], vector env: gym.vector.SyncVectorEnv
    test_envs: list = None,
    writer: SummaryWriter = None,
):
    """
    支持 n_envs>1 的 RL²-PPO 训练函数，适配 VecEnvRolloutBuffer。
    """

    # 这里的envs是一个只包含单个元素的list
    n_envs = envs[0].unwrapped.num_envs

    agent = PPO(
        obs_space=envs[0].observation_space,
        action_space=envs[0].action_space,
        writer=writer,
        device=config["device"],
        ac_kwargs=config["actor_critic"],
        **config["ppo"],
    )

    buffer = VecEnvRolloutBuffer(
        size=config["max_episode_steps"],
        obs_space=envs[0].observation_space,  # 注意这里是env，不是envs[0]
        action_space=envs[0].action_space,
        device=config["device"],
        gae_lambda=config["ppo"]["gae_lambda"],
    )

    global_step = 0

    # -------------------------
    # 2. meta-training loop
    # -------------------------
    for meta_epoch in range(config["meta_epochs"]):
        # 采样新任务
        env = np.random.choice(envs, 1)[0]
        env.unwrapped.meta_reset()

        avg_return, avg_ep_len, avg_success = [], [], []

        # 初始化 RL² 隐状态和 prev_action/reward
        rnn_state = agent.actor_critic.initialize_state(batch_size=n_envs)
        prev_action = (
            torch.tensor(env.action_space.sample())
              .to(config["device"])
        )
        # for rnn input
        prev_rew = torch.zeros((n_envs), device=config["device"]).view(n_envs, 1)

        obs, _ = env.reset()
        termination = np.array([False] * n_envs)
        truncated = np.array([False] * n_envs)

        ep_return = np.zeros(n_envs)
        ep_length = np.zeros(n_envs, dtype=int)

        for step in range(config["max_episode_steps"]):
            obs_tensor = torch.tensor(obs).to(config["device"]).float()
            action, value, log_prob, rnn_state = agent.act(
                obs_tensor, prev_action, prev_rew, rnn_state
            )
            # step环境, shappe: (n_envs, ...)
            next_obs, reward, term, trunc, info = env.step(action)
            # 存储交互的数据
            buffer.store(
                obs_tensor,      # [n_envs, obs_dim]
                action,          # [n_envs, action_dim] or [n_envs]
                reward,          # [n_envs]
                prev_action,     # [n_envs, action_dim] or [n_envs]
                prev_rew,        # [n_envs, 1]
                term,            # [n_envs]
                value,           # [n_envs]
                log_prob,        # [n_envs]
            )
            ep_return += reward.cpu().numpy()
            ep_length += 1

            # 更新 obs、prev_action/reward
            obs = next_obs
            prev_action = action.detach()
            prev_rew = torch.tensor(reward).to(config["device"]).view(n_envs, 1)
            termination = term
            truncated = trunc

            # 如果所有环境都结束则 break
            if termination.all() and truncated.all():
                break

        # 结束时，计算每个环境的 last_value 和 last_termination
        obs_tensor = torch.tensor(obs).to(config["device"]).float()
        _, last_value, _, _ = agent.act(
            obs_tensor, prev_action, prev_rew, rnn_state
        )
        buffer.finish_path(
            last_value.detach().cpu().numpy(),
            termination,
        )

        # PPO优化
        batch, batch_size = buffer.get()
        agent.optimize(
            batch,
            config["update_epochs"],
            batch_size,
            global_step
        )
        buffer.reset()

        # 日志统计
        avg_return.append(np.mean(ep_return))
        avg_ep_len.append(np.mean(ep_length))
        success_rate = info['log'].get('Metrics/object_pose/success', 0.0)
        avg_success.append(success_rate)

        global_step += n_envs * config["max_episode_steps"]

        # meta-testing & 保存模型
        if meta_epoch % config["log_every_n"] == 0 and meta_epoch != 0:
            if test_envs is not None:
                test_env = np.random.choice(test_envs, 1)[0]
                test_return, test_ep_len = meta_evaluate_policy_vecenv(agent, test_env, config)
                logging.info(
                    f"[Meta-Epoch {meta_epoch}] "
                    f"Train: return={np.mean(avg_return):.2f}, length={np.mean(avg_ep_len):.1f}, success={np.mean(avg_success):.2f} | "
                    f"Test: return={test_return:.2f}, length={test_ep_len:.1f}"
                )
                if writer is not None:
                    writer.add_scalar("train/return", np.mean(avg_return), meta_epoch)
                    writer.add_scalar("train/success_rate", np.mean(avg_success), meta_epoch)
                    writer.add_scalar("test/return", test_return, meta_epoch)
                    writer.add_scalar("test/length", test_ep_len, meta_epoch)

            # 保存模型
            if not config["debug"]:
                agent.save_weights(config["path"], meta_epoch)


def evaluate_policy(agent, env, config, episodes=10):
    """
    Evaluate the performance of an agent's policy on a given environment.

    Args:
        agent (object): An instance of the agent to be evaluated.
        env (object): An instance of the OpenAI gym environment to evaluate the agent on.
        device (str): The device to run the evaluation on (e.g. 'cpu', 'cuda').
        episodes (int): The number of episodes to run the evaluation for.

    Returns:
        A tuple of two floats, representing the average return and average episode length
        over the given number of episodes.
    """

    rnn_state = agent.actor_critic.initialize_state(batch_size=1)
    prev_action = (
        torch.tensor(env.action_space.sample()).to(config["device"]).view(-1)
    )
    prev_rew = torch.tensor(0).to(config["device"]).view(1, 1)

    avg_return, avg_ep_len = [], []
    for _ in range(1, episodes):
        obs, _ = env.reset()
        termination, truncated = False, False

        while not (termination or truncated):
            obs = torch.tensor(obs).to(config["device"]).float().unsqueeze(0)
            act, _, _, rnn_state = agent.act(
                obs, prev_action, prev_rew, rnn_state
            )
            next_obs, rew, termination, truncated, info = env.step(
                act.cpu().numpy()
            )

            # Update the observation
            obs = next_obs

            # Set previous action and reward tensors
            prev_action = act.detach()
            prev_rew = torch.tensor(rew).to(config["device"]).view(1, 1)

            if termination or truncated:
                avg_return.append(info["episode"]["r"])
                avg_ep_len.append(info["episode"]["l"])
                break

    return np.array(avg_return).mean(), np.array(avg_ep_len).mean()


def meta_evaluate_policy(agent, env, config, episodes=10):
    """
    Evaluate the performance of an agent's policy on a given environment.

    Args:
        agent (object): An instance of the agent to be evaluated.
        env (object): An instance of the OpenAI gym environment to evaluate the agent on.
        device (str): The device to run the evaluation on (e.g. 'cpu', 'cuda').
        episodes (int): The number of episodes to run the evaluation for.

    Returns:
        A tuple of two floats, representing the average return and average episode length
        over the given number of episodes.
    """

    rnn_state = agent.actor_critic.initialize_state(batch_size=1)
    prev_action = (
        torch.tensor(env.action_space.sample()).to(config["device"])
    )
    prev_rew = torch.tensor(0).to(config["device"]).view(1, 1)

    avg_return, avg_ep_len = [], []
    for _ in range(1, episodes):
        obs, _ = env.reset()
        termination, truncated = False, False

        while not (termination or truncated):
            obs = torch.tensor(obs).to(config["device"]).float()
            act, _, _, rnn_state = agent.act(
                obs, prev_action, prev_rew, rnn_state
            )
            next_obs, rew, termination, truncated, info = env.step(
                act
            )

            # Update the observation
            obs = next_obs

            # Set previous action and reward tensors
            prev_action = act.detach()
            prev_rew = torch.tensor(rew).to(config["device"]).view(1, 1)

            if termination or truncated:
                avg_return.append(info["episode"]["r"].cpu().numpy())
                avg_ep_len.append(info["episode"]["l"])
                break

    return np.array(avg_return).mean(), np.array(avg_ep_len).mean()


# 支持n_envs>1的并行环境
def meta_evaluate_policy_vecenv(agent, env, config, episodes=10):
    """
    Evaluate the performance of an agent's policy on a vectorized environment (n_envs > 1).

    Returns:
        平均 return 和平均 episode length（所有环境、所有 episode 的均值）
    """
    n_envs = env.unwrapped.num_envs
    avg_return, avg_ep_len = [], []

    for _ in range(episodes):
        rnn_state = agent.actor_critic.initialize_state(batch_size=n_envs)
        prev_action = torch.tensor(env.action_space.sample()).to(config["device"])
        prev_rew = torch.zeros((n_envs, 1), device=config["device"])
        obs, _ = env.reset()
        termination = np.array([False] * n_envs)
        truncated = np.array([False] * n_envs)
        ep_return = np.zeros(n_envs)
        ep_length = np.zeros(n_envs, dtype=int)

        while not ((termination | truncated).all()):
            obs_tensor = obs.detach().clone().to(config["device"]).float() \
                if torch.is_tensor(obs) \
                else torch.tensor(obs).to(config["device"]).float()
            act, _, _, rnn_state = agent.act(
                obs_tensor, prev_action, prev_rew, rnn_state
            )
            next_obs, rew, term, trunc, info = env.step(act)
            ep_return += rew.cpu().numpy()
            ep_length += 1

            obs = next_obs
            prev_action = act.detach()
            prev_rew = torch.tensor(rew).to(config["device"]).view(n_envs, 1)
            termination = term
            truncated = trunc

        # 统计每个环境的 return 和 length
        avg_return.extend(ep_return.tolist())
        avg_ep_len.extend(ep_length.tolist())

    return np.mean(avg_return), np.mean(avg_ep_len)