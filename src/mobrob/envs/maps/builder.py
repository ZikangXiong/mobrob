import matplotlib.pyplot as plt
import numpy as np
import yaml
from mobrob.utils import DATA_DIR


class MapBuilder:
    def __init__(self, map_size: np.ndarray, pixel_size: float):
        self.map_size = map_size
        self.pixel_size = pixel_size
        self.map_pixel_dim = self.value_to_pixel(self.map_size)
        self._map = self.create_empty_map()
        self.config = {
            "map_size": self.map_size.tolist(),
            "pixel_size": self.pixel_size,
            "obstacles": [],
        }

    @classmethod
    def from_predefined_map(cls, map_id: int):
        """Creates a map builder from a yaml config file."""
        config_path: str = f"{DATA_DIR}/maps/{map_id}.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        map_builder = cls(
            map_size=np.array(config["map_size"]),
            pixel_size=config["pixel_size"],
        )

        for obstacle in config["obstacles"]:
            if obstacle["type"] == "circle":
                map_builder.add_circle_obstacle(
                    center=np.array(obstacle["center"]),
                    radius=obstacle["radius"],
                )
            elif obstacle["type"] == "rectangle":
                map_builder.add_rectangle_obstacle(
                    center=np.array(obstacle["center"]),
                    size=np.array(obstacle["size"]),
                )
            else:
                raise ValueError(f"Unknown obstacle type {obstacle['type']}")

        return map_builder

    def create_empty_map(self):
        """Creates an empty map of given size."""
        return np.zeros(self.map_pixel_dim.astype(int))

    def value_to_pixel(self, value: np.ndarray) -> np.ndarray:
        """Converts a value in meters to pixels."""
        pixel_value = (value / self.pixel_size).round()
        if pixel_value.size == 1:
            return pixel_value.item()
        return pixel_value

    def position_to_pixel(self, position: np.ndarray) -> np.ndarray:
        """Converts a position in meters to pixels."""
        position = position[..., ::-1]
        pixel_position = self.value_to_pixel(position)

        # rotate 90 degrees
        rotate_matrix = np.array([[0, -1], [1, 0]])
        pixel_position = np.matmul(pixel_position, rotate_matrix)
        # move origin to upper left corner
        pixel_position = pixel_position + self.map_pixel_dim / 2

        return pixel_position

    def add_circle_obstacle(self, center: np.ndarray, radius: float):
        """Adds a circular obstacle to the map."""
        self.config["obstacles"].append(
            {
                "type": "circle",
                "center": center.tolist(),
                "radius": radius,
            }
        )
        center_pixel = self.position_to_pixel(center)
        radius_pixel = self.value_to_pixel(np.array(radius))
        y, x = np.ogrid[
            -center_pixel[1] : self.map_pixel_dim[0] - center_pixel[1],
            -center_pixel[0] : self.map_pixel_dim[1] - center_pixel[0],
        ]

        mask = x * x + y * y <= radius_pixel**2
        self._map[mask] = 1

    def add_rectangle_obstacle(self, center: np.ndarray, size: np.ndarray):
        """Adds a rectangular obstacle to the map."""
        self.config["obstacles"].append(
            {
                "type": "rectangle",
                "center": center.tolist(),
                "size": size.tolist(),
            }
        )

        half_size = size / 2
        lower_left = center - half_size
        upper_right = center + half_size

        # convert these corners to pixels
        lower_left_pixel = self.position_to_pixel(lower_left)
        upper_right_pixel = self.position_to_pixel(upper_right)

        # fill in the rectangle on the map
        self._map[
            int(upper_right_pixel[1]) : int(lower_left_pixel[1]),
            int(lower_left_pixel[0]) : int(upper_right_pixel[0]),
        ] = 1

    def scale_map(self, scale_factor: float):
        """Scales the map."""
        self.pixel_size *= scale_factor
        self.map_size *= scale_factor
        self.map_pixel_dim = self.value_to_pixel(self.map_size)

        self.config["map_size"] = (
            np.array(self.config["map_size"]) * scale_factor
        ).tolist()
        self.config["pixel_size"] *= scale_factor

        self._map = self.create_empty_map()
        obstacles = self.config["obstacles"]
        self.config["obstacles"] = []
        for obstacle in obstacles:
            if obstacle["type"] == "circle":
                self.add_circle_obstacle(
                    center=np.array(obstacle["center"]) * scale_factor,
                    radius=obstacle["radius"] * scale_factor,
                )
            elif obstacle["type"] == "rectangle":
                self.add_rectangle_obstacle(
                    center=np.array(obstacle["center"]) * scale_factor,
                    size=np.array(obstacle["size"]) * scale_factor,
                )
            else:
                raise ValueError(f"Unknown obstacle type {obstacle['type']}")

    def plot_map(self):
        plt.imshow(self._map)
        plt.show()

    def dump_config(self, path):
        with open(path, "w") as f:
            yaml.dump(self.config, f)

    def get_map_array(self) -> np.ndarray:
        return self._map
