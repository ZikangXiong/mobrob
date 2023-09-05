import matplotlib.pyplot as plt

from mobrob.envs.maps.builder import MapBuilder
from mobrob.envs.wrapper import get_env


def turtlebot3_map_builder():
    map_builder = MapBuilder()
    map = map_builder.sample_a_map(2.42, 5)
    map.generate_map_img()

    env = get_env("turtlebot3", enable_gui=True, map_config=map.map_config)
    env.reset()

    while True:
        env.render()
        action = env.action_space.sample()
        env.step(action)


if __name__ == "__main__":
    # test_map_builder()
    # test_mujoco_map()
    turtlebot3_map_builder()
