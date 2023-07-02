import argparse
import os

import torch
import yaml

from mobrob.rl_control.ppo import PPOCtrl
from mobrob.utils import DATA_DIR

torch.set_num_threads(1)


def train_with_ppo(env_name):
    config = yaml.load(
        open(f"{DATA_DIR}/configs/{env_name}-ppo.yaml", "r"), Loader=yaml.FullLoader
    )
    ppo_ctrl = PPOCtrl.from_config(config=config)
    ppo_ctrl.learn(total_timesteps=config["total_timesteps"])

    os.makedirs(f"{DATA_DIR}/policies", exist_ok=True)
    ppo_ctrl.save_model(f"{DATA_DIR}/policies/{env_name}-ppo.zip")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--env_name", type=str, default="car")

    args = args_parser.parse_args()
    train_with_ppo(env_name=args.env_name)
