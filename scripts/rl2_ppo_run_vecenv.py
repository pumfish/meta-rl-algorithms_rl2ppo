import os
import sys
sys.path.append(os.path.abspath(r"D:\WorkSpace\meta-rl-algorithms"))
import yaml
import logging
import argparse

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from isaaclab.app import AppLauncher
import gymnasium as gym
from gymnasium.wrappers import OrderEnforcing, RecordEpisodeStatistics

from utils.logger import configure_logger
from algos.rl2_ppo.learning import train_rl2_ppo_meta_reset_vecenv, meta_evaluate_policy
from algos.rl2_ppo.agent import PPO

RL2_PPO_CFG_PATH = r"D:\WorkSpace\meta-rl-algorithms\configs\rl2_ppo_test.yml"


# 把obs从dict转为tensor
class DictToTensorWrapper(gym.Wrapper):
    """
    将 dict 类型的 observation 转成单一 tensor
    支持 step、reset、meta_reset
    """
    def __init__(self, env, key="policy"):
        super().__init__(env)
        self.key = key
        # 更新 observation_space
        self.observation_space = env.observation_space[self.key]

    def _convert_obs(self, obs):
        if isinstance(obs, dict):
            obs = obs[self.key]
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._convert_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if "episode" in info:
            del info["episode"]
        return self._convert_obs(obs), reward, terminated, truncated, info

    def meta_reset(self, **kwargs):
        """
        如果环境有 meta_reset 接口，也需要做同样处理
        """
        obs, info = self.env.meta_reset(**kwargs)
        return self._convert_obs(obs), info


# -------------------
# 主训练函数
# -------------------
def main(config):
    # 初始化环境
    TASK_NAME = "Isaac-Grasp-Cube-Franka-Meta"
    env_cfg = load_cfg_from_registry(TASK_NAME, "env_cfg_entry_point")
    logging.info(f"Start meta-training, experiment name: {config['name']}")
    logging.info(f"config: {config}")

    train_envs = [
        RecordEpisodeStatistics(
            DictToTensorWrapper(gym.make(TASK_NAME, cfg=env_cfg))
        )
    ]

    test_envs = train_envs

    # 不可重复定义环境
    # test_envs = [
    #     OrderEnforcing(
    #         RecordEpisodeStatistics(
    #             gym.make(TASK_NAME, cfg=env_cfg)
    #         )
    #     )
    # ]

    logging.info(
        f"Env spaces: {train_envs[0].observation_space, train_envs[0].action_space}, "
        f"max steps: {config['max_episode_steps']}"
    )

    # TensorBoard
    writer = SummaryWriter(os.path.join(config["path"], "tb"))

    # 启动训练
    train_rl2_ppo_meta_reset_vecenv(config, train_envs, test_envs, writer)

    # # 简单测试
    # # 加载模型：
    # agent = PPO(
    #     obs_space=test_envs[0].observation_space,
    #     action_space=test_envs[0].action_space,
    #     writer=writer,
    #     device=config["device"],
    #     ac_kwargs=config["actor_critic"],
    #     **config["ppo"],
    # )
    # path = r"D:\WorkSpace\runs\debug2\e1_state"
    # agent.load_weights(path)
    # test_return, test_ep_len = meta_evaluate_policy(agent, test_envs[0], config, episodes=2)
    # print(f"[Test] avg return: {test_return:.2f}, avg ep length: {test_ep_len:.1f}")

    # 清理
    for env in train_envs + test_envs:
        env.close()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, required=True, help="Experiment name")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    parser.add_argument("-c", "--config", type=str, default=RL2_PPO_CFG_PATH)
    args = parser.parse_args()

    # 读取配置文件
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 加入额外配置项
    config["name"] = args.name
    config["debug"] = args.debug
    config["path"] = f"runs/{args.name}"

    # 初始化日志
    configure_logger(args.name, config["path"])

    # 设备与随机数种子
    config["device"] = torch.device(config["device_id"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # IsaacLab 启动（必须放 main 外部，否则训练时环境不正常）
    app_launcher = AppLauncher(headless=True)
    sim_app = app_launcher.app
    from isaaclab_tasks.utils import load_cfg_from_registry

    main(config)
    sim_app.close()
