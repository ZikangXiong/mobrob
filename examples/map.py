from mobrob.envs.maps.builder import MapBuilder


def test_map_builder():
    map_builder = MapBuilder.from_predefined_map(5)
    map_builder.plot_map()


if __name__ == "__main__":
    test_map_builder()
