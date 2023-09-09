import argparse
import os

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from mobrob.rl_control.ppo import PPOCtrl
from mobrob.utils import DATA_DIR

# for fast inference
torch.set_num_threads(1)


def train_with_ppo(env_name, finetune=False, save_freq=1_000_000):
    """
    Train a PPO agent with the given environment name.
    The training logs and intermediate models are saved in DATA_DIR/policies/tmp/{env_name}-ppo.

    @param env_name: The name of the environment to train on.
    @param finetune: Whether to finetune a pretrained policy.
    @param save_freq: The frequency (each save_freq timesteps) which to save the policy.
    """
    config = yaml.load(
        open(f"{DATA_DIR}/configs/{env_name}-ppo.yaml", "r"), Loader=yaml.FullLoader
    )
    ppo_ctrl = PPOCtrl.from_config(config=config)

    if finetune:
        ppo_ctrl.ppo.policy.load_state_dict(
            PPO.load(f"{DATA_DIR}/policies/{env_name}-ppo.zip").policy.state_dict(),
        )

    temp_dir = f"{DATA_DIR}/policies/tmp/{env_name}-ppo"
    save_callback = CheckpointCallback(
        save_freq=save_freq // config["n_envs"],
        save_path=f"{temp_dir}/models",
        name_prefix=f"timestep",
        verbose=1,
    )
    ppo_ctrl.learn(
        total_timesteps=config["total_timesteps"],
        callback=save_callback,
        progress_bar=True,
    )

    os.makedirs(f"{DATA_DIR}/policies", exist_ok=True)
    ppo_ctrl.save_model(f"{DATA_DIR}/policies/{env_name}-ppo.zip")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--env-name", type=str, default="turtlebot3_partial_obs")
    args_parser.add_argument("--finetune", action="store_true", default=False)
    args_parser.add_argument("--save-freq", type=int, default=1_000_000)

    args = args_parser.parse_args()
    train_with_ppo(
        env_name=args.env_name, finetune=args.finetune, save_freq=args.save_freq
    )
