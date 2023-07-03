import pybullet as p
import pybullet_data

from mobrob.envs.pybullet_robots.base import WorldBase
from mobrob.envs.pybullet_robots.utils import ROBOT_ASSETS_PATH

URDF_FOLDER = f"{ROBOT_ASSETS_PATH}/turtlebot3"


def wall_info(wall_length, wall_thick, wall_height):
    half_extents = [
        [(wall_length - wall_thick) / 2, wall_thick / 2, wall_height / 2],
        [wall_thick / 2, (wall_length - wall_thick) / 2, wall_height / 2],
        [(wall_length - wall_thick) / 2, wall_thick / 2, wall_height / 2],
        [wall_thick / 2, (wall_length - wall_thick) / 2, wall_height / 2],
    ]

    frame_positions = [
        [-wall_thick / 2, (wall_length - wall_thick) / 2, 0],
        [(wall_length - wall_thick) / 2, wall_thick / 2, 0],
        [wall_thick / 2, -(wall_length - wall_thick) / 2, 0],
        [-(wall_length - wall_thick) / 2, -wall_thick / 2, 0],
    ]

    return half_extents, frame_positions


def create_wall(wall_length, wall_thick, wall_height, client_id):
    half_extents, frame_positions = wall_info(wall_length, wall_thick, wall_height)

    wall_vid = p.createVisualShapeArray(
        shapeTypes=[p.GEOM_BOX] * 4,
        halfExtents=half_extents,
        rgbaColors=[[0.9, 0.6, 0.6, 1]] * 4,
        visualFramePositions=frame_positions,
        physicsClientId=client_id,
    )
    wall_cid = p.createCollisionShapeArray(
        shapeTypes=[p.GEOM_BOX] * 4,
        halfExtents=half_extents,
        collisionFramePositions=frame_positions,
        physicsClientId=client_id,
    )
    wall_id = p.createMultiBody(
        baseVisualShapeIndex=wall_vid,
        baseCollisionShapeIndex=wall_cid,
        basePosition=[0.0, 0.0, wall_height / 2],
        physicsClientId=client_id,
    )

    return wall_id


class World(WorldBase):
    def _init_param(self):
        self.g = 9.8
        self.timestep = 1 / 50

    def _build_world(self):
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.client_id
        )
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        p.setGravity(0, 0, -self.g, physicsClientId=self.client_id)
        p.setTimeStep(self.timestep, physicsClientId=self.client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)

        create_wall(
            wall_length=2.42 + 0.28 * 2,
            wall_thick=0.265,
            wall_height=0.265,
            client_id=self.client_id,
        )

    def reset(self):
        pass
