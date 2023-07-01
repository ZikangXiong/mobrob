import time

import numpy as np

from mobrob.envs.wrapper import get_env


def mujoco_render():
    env = get_env("doggo", enable_gui=True)
    env.reset()

    for i in range(1000):
        env.render()
        env.step(env.action_space.sample())


def bullet_render():
    env = get_env("drone", enable_gui=True)
    env.reset()

    for i in range(1000):
        env.render()
        env.step(np.array([4500, 4500, 4500, 4500]))
        time.sleep(1 / 240)


if __name__ == "__main__":
    # mujoco_render()
    bullet_render()
