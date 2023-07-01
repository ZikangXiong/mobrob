import os
from os.path import abspath, dirname

from stable_baselines3 import PPO

import mobrob

DATA_DIR = os.path.join(dirname(dirname(dirname(abspath(mobrob.__file__)))), "data")
PROJ_DIR = dirname(abspath(mobrob.__file__))


def load_policy(env_name: str, policy_name: str):
    return PPO.load(f"{DATA_DIR}/policies/{env_name}-{policy_name}.zip")
