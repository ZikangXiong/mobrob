import os

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p


class TwoDMap:
    def __init__(self, map_config: dict):
        self.map_config = map_config
        self.lim = (0, map_config["map_size"])
        self.pybullet_objects = []

    def generate_map_img(self, resolution: int = 224) -> np.ndarray:
        # plot obstacles in an array
        map_img = np.zeros((resolution, resolution))

        for obstacle in self.map_config["obstacles"]:
            if obstacle["type"] == "circle":
                center = np.array(obstacle["center"])
                radius = obstacle["radius"]

                # convert these corners to pixels
                center = center / (self.lim[1] - self.lim[0]) * resolution
                center = center.astype(int)
                radius = radius / (self.lim[1] - self.lim[0]) * resolution
                radius = int(radius)

                x, y = np.ogrid[
                    -center[1] : resolution - center[1],
                    -center[0] : resolution - center[0],
                ]
                mask = x * x + y * y <= radius**2
                map_img[mask] = 1
            elif obstacle["type"] == "rectangle":
                center = np.array(obstacle["center"])
                size = np.array(obstacle["size"])

                # convert these corners to pixels
                center = center * resolution / (self.lim[1] - self.lim[0])
                center = center.astype(int)
                size = size * resolution / (self.lim[1] - self.lim[0])
                size = size.astype(int)

                upper_left = center - size
                lower_right = center + size

                # fill in the rectangle on the map
                upper_left = np.clip(upper_left, 0, resolution)
                lower_right = np.clip(lower_right, 0, resolution)

                map_img[
                    int(upper_left[0]) : int(lower_right[0]),
                    int(upper_left[1]) : int(lower_right[1]),
                ] = 1

        return map_img

    def to_mojoco(self) -> dict:
        """
        Converts the map to a mojoco config.
        1. switch x and y axis.
        2. move origin to the center.
        3. invert y axis.
        """
        mujoco_config = {
            "obstacles": [],
            "map_size": self.map_config["map_size"],
        }
        for obj in self.map_config:
            swich_xy = [obj["center"][1], obj["center"][0]]
            move_origin = [
                swich_xy[0] - (self.lim[0] + self.lim[1]) / 2,
                swich_xy[1] - (self.lim[0] + self.lim[1]) / 2,
            ]
            invert_y = [move_origin[0], -move_origin[1]]
            center = invert_y

            if obj["type"] == "rectangle":
                size = [obj["size"][1], obj["size"][0]]
                mujoco_config["obstacles"].append(
                    {
                        "type": "rectangle",
                        "center": center,
                        "rotation": [0, 0, 0, 1],
                        "size": size,
                    }
                )
            elif obj["type"] == "circle":
                radius = obj["radius"]
                mujoco_config["obstacles"].append(
                    {
                        "type": "circle",
                        "center": center,
                        "radius": radius,
                    }
                )

        return mujoco_config

    def to_pybullet(self, client_id: int = None):
        """
        Builds the map in pybullet.
        """

        for obj in self.map_config["obstacles"]:
            if obj["type"] == "circle":
                shape_id = p.createCollisionShape(
                    p.GEOM_CYLINDER,
                    radius=obj["radius"],
                    height=0.1,
                    physicsClientId=client_id,
                )
                p.createMultiBody(
                    baseMass=0,
                    basePosition=np.r_[obj["center"][::-1], 0.1],
                    baseCollisionShapeIndex=shape_id,
                    physicsClientId=client_id,
                )
            elif obj["type"] == "rectangle":
                shape_id = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=np.r_[obj["size"], 0.1],
                    physicsClientId=client_id,
                )
                p.createMultiBody(
                    baseMass=0,
                    basePosition=np.r_[obj["center"], 0.1],
                    baseCollisionShapeIndex=shape_id,
                    physicsClientId=client_id,
                )


class MapBuilder:
    def __init__(self):
        self.config = {}

    def add_wall(self, map_size: float):
        # north wall
        self.config["obstacles"].append(
            {
                "type": "rectangle",
                "center": [0, map_size / 2],
                "rotation": [0, 0, 0, 1],
                "size": [0.1, map_size / 2],
            }
        )

        # south wall
        self.config["obstacles"].append(
            {
                "type": "rectangle",
                "center": [map_size, map_size / 2],
                "rotation": [0, 0, 0, 1],
                "size": [0.1, map_size / 2],
            }
        )

        # east wall
        self.config["obstacles"].append(
            {
                "type": "rectangle",
                "center": [map_size / 2, 0],
                "rotation": [0, 0, 0, 1],
                "size": [map_size / 2, 0.1],
            }
        )

        # west wall
        self.config["obstacles"].append(
            {
                "type": "rectangle",
                "center": [map_size / 2, map_size],
                "rotation": [0, 0, 0, 1],
                "size": [map_size / 2, 0.1],
            }
        )

    def add_rectangle(self, center: list, size: list):
        self.config["obstacles"].append(
            {
                "type": "rectangle",
                "center": center,
                "rotation": [0, 0, 0, 1],
                "size": size,
            }
        )

    def add_circle(self, center: list, radius: float):
        self.config["obstacles"].append(
            {
                "type": "circle",
                "center": center,
                "radius": radius,
            }
        )

    def sample_a_map(
        self,
        map_size: float,
        n_obs: int,
        size_range: tuple = (0.05, 0.1),
        keepout: float = 0.3,
    ) -> TwoDMap:
        self.config = {
            "map_size": map_size,
            "obstacles": [],
        }

        for _ in range(n_obs):
            shape = np.random.choice(["circle", "rectangle"])

            if shape == "circle":
                for _ in range(100):
                    radius = np.random.uniform(*size_range)
                    center = np.random.uniform(0.2, map_size - 0.2, 2)
                    new_obj = {
                        "type": "circle",
                        "center": center.tolist(),
                        "radius": radius,
                    }
                    if not self.check_collision(new_obj, keepout=keepout):
                        break
                self.add_circle(center.tolist(), radius)
            elif shape == "rectangle":
                for _ in range(100):
                    size = np.random.uniform(*size_range, 2)
                    center = np.random.uniform(0.2, map_size - 0.2, 2)
                    new_obj = {
                        "type": "rectangle",
                        "center": center.tolist(),
                        "size": size.tolist(),
                    }
                    if not self.check_collision(new_obj, keepout=keepout):
                        break
                self.add_rectangle(center.tolist(), size.tolist())

        self.add_wall(map_size)

        return TwoDMap(self.config)

    def check_collision(self, new_obj: dict, keepout=0.3):
        if new_obj["type"] == "rectangle":
            radius = np.linalg.norm(new_obj["size"])
        elif new_obj["type"] == "circle":
            radius = new_obj["radius"]

        for obj in self.config["obstacles"]:
            if obj["type"] == "rectangle":
                size = np.linalg.norm(obj["size"])
            elif obj["type"] == "circle":
                size = obj["radius"]

            if (
                np.linalg.norm(np.array(new_obj["center"]) - np.array(obj["center"]))
                < radius + size - keepout
            ):
                return True

        return False
