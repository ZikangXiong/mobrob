from xml.etree import ElementTree

import numpy as np
import pybullet as p
import pybullet_data

from mobrob.envs.pybullet_robots.base import RobotBase, WorldBase
from mobrob.envs.pybullet_robots.robots import ROBOT_ASSETS_PATH


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

    def reset(self):
        pass


class Drone(RobotBase):
    def __init__(self, drone_name: str = "hb", enable_gui=True):
        self.urdf_path = f"{ROBOT_ASSETS_PATH}/drone"
        self.drone_name = drone_name
        self.init_low = [-3, -3, 1]
        self.init_high = [3, 3, 3]

        world = World(enable_gui=enable_gui)
        super(Drone, self).__init__(world)
        self._init_param()

    def reset(self):
        init_pos = np.random.uniform(self.init_low, self.init_high)
        p.resetBasePositionAndOrientation(
            self.robot_id, init_pos, p.getQuaternionFromEuler([0, 0, 0]), self.client_id
        )

    def _load_robot(self) -> int:
        robot_id = p.loadURDF(
            self.urdf_path + f"/{self.drone_name}.urdf",
            basePosition=[2, 2, 2],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self.client_id,
        )

        return robot_id

    def _init_param(self):
        urdf_tree = ElementTree.parse(
            f"{self.urdf_path}/{self.drone_name}.urdf"
        ).getroot()

        self.m = float(urdf_tree[1][0][1].attrib["value"])
        self.l = float(urdf_tree[0].attrib["arm"])
        self.thrust2weight = float(urdf_tree[0].attrib["thrust2weight"])
        self.kf = float(urdf_tree[0].attrib["kf"])
        self.km = float(urdf_tree[0].attrib["km"])
        self.max_speed_kmh = float(urdf_tree[0].attrib["max_speed_kmh"])
        self.prop_radius = float(urdf_tree[0].attrib["prop_radius"])

        ixx = float(urdf_tree[1][0][2].attrib["ixx"])
        iyy = float(urdf_tree[1][0][2].attrib["iyy"])
        izz = float(urdf_tree[1][0][2].attrib["izz"])
        self.J = np.diag([ixx, iyy, izz])
        self.J_inv = np.linalg.inv(self.J)

        self.collision_h = float(urdf_tree[1][2][1][0].attrib["length"])
        self.collision_r = float(urdf_tree[1][2][1][0].attrib["radius"])
        collision_shape_offsets = [
            float(s) for s in urdf_tree[1][2][0].attrib["xyz"].split(" ")
        ]
        self.collision_z_offset = collision_shape_offsets[2]

        self.gnd_eff_coef = float(urdf_tree[0].attrib["gnd_eff_coeff"])

        drag_coeff_xy = float(urdf_tree[0].attrib["drag_coeff_xy"])
        drag_coeff_z = float(urdf_tree[0].attrib["drag_coeff_z"])
        self.drag_coef = np.array([drag_coeff_xy, drag_coeff_xy, drag_coeff_z])

        self.dw_coef_1 = float(urdf_tree[0].attrib["dw_coeff_1"])
        self.dw_coef_2 = float(urdf_tree[0].attrib["dw_coeff_2"])
        self.dw_coef_3 = float(urdf_tree[0].attrib["dw_coeff_3"])

        gravity = self.world.g * self.m  # noqa
        self.hover_rpm = np.sqrt(gravity / (4 * self.kf))
        self.max_rpm = np.sqrt((self.thrust2weight * gravity) / (4 * self.kf))

        self.max_thrust = 4 * self.kf * self.max_rpm**2
        if self.drone_name == "cf2x":
            self.max_xy_torque = (2 * self.l * self.kf * self.max_rpm**2) / np.sqrt(2)
        elif self.drone_name in ["cf2p", "hb"]:
            self.max_xy_torque = self.l * self.kf * self.max_rpm**2

        self.max_z_torque = 2 * self.km * self.max_rpm**2

        # A and B are used for building the equation for computing control input
        self.A = np.array([[1, 1, 1, 1], [0, 1, 0, -1], [-1, 0, 1, 0], [-1, 1, -1, 1]])
        self.A_inv = np.linalg.inv(self.A)
        self.B = np.array(
            [1 / self.kf, 1 / (self.kf * self.l), 1 / (self.kf * self.l), 1 / self.km]
        )

    def ctrl(self, cmd: np.ndarray):
        """
        Apply control cmd and step simulation
        :param cmd: rotation per minute of 4 motors
        """
        forces = np.array(cmd**2) * self.kf
        torques = np.array(cmd**2) * self.km
        z_torque = -torques[0] + torques[1] - torques[2] + torques[3]
        for i in range(4):
            p.applyExternalForce(
                self.robot_id,
                i,
                forceObj=[0, 0, forces[i]],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.client_id,
            )
        p.applyExternalTorque(
            self.robot_id,
            -1,
            torqueObj=[0, 0, z_torque],
            flags=p.LINK_FRAME,
            physicsClientId=self.client_id,
        )

    def get_base_pos_and_ori(self) -> np.ndarray:
        return p.getBasePositionAndOrientation(self.robot_id, self.client_id)

    def get_obs(self) -> np.ndarray:
        pos, ori = p.getBasePositionAndOrientation(self.robot_id, self.client_id)
        ori = p.getEulerFromQuaternion(ori)
        vel, rot = p.getBaseVelocity(self.robot_id, self.client_id)

        return np.concatenate([pos, ori, vel, rot])
