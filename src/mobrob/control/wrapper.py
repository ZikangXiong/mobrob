import numpy as np
from stable_baselines3 import PPO

from mobrob.utils import DATA_DIR


class Controller:
    def __init__(self, robot_name: str):
        if robot_name == "point":
            self.policy = PPO.load(f"{DATA_DIR}/controllers/point_ppo.zip")
        elif robot_name == "car":
            self.policy = PPO.load(f"{DATA_DIR}/controllers/car_ppo.zip")
        elif robot_name == "doggo":
            self.policy = PPO.load(f"{DATA_DIR}/controllers/doggo_ppo.zip")
        elif robot_name == "drone":
            self.policy = PPO.load(f"{DATA_DIR}/controllers/drone_ppo.zip")
        else:
            raise ValueError(f"Unknown robot name: {robot_name}")

    def __call__(self, gc_obs: np.ndarray) -> np.ndarray:
        action = self.policy.predict(gc_obs)[0]
        return action


def get_controller(robot_name: str) -> Controller:
    return Controller(robot_name)
