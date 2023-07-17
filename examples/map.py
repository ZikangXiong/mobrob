from mobrob.envs.maps.builder import MapBuilder
from mobrob.envs.wrapper import get_env
from mobrob.utils import DATA_DIR


def test_map_builder():
    map_builder = MapBuilder.from_predefined_map(5)
    map_builder.plot_map()


def test_mujoco_map():
    env = get_env("point", enable_gui=True, map_id=7)

    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())


if __name__ == "__main__":
    test_map_builder()
    test_mujoco_map()
