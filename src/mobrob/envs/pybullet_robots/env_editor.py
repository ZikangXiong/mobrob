from typing import Dict, List, Tuple, Union

import numpy as np
import pybullet as p
import pybullet_data

from mobrob.envs.pybullet_robots.base import ObjID


class EnvEditor:
    def __init__(self, client_id: int):
        self.client_id = client_id

        # cache objects to avoid recreation
        self.objects: Dict[str, Dict[Tuple, ObjID]] = {
            "ball": {},
            "cube": {},
            "duck": {},
        }

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.client_id
        )

    def get_shape(self, name: str, props: tuple) -> ObjID:
        obj_id = self.objects[name].get(props, None)
        if obj_id is not None:
            return obj_id

        if name == "ball":
            size, color = props
            radius = 0.1 * size
            vid = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                rgbaColor=color,
                radius=radius,
                physicsClientId=self.client_id,
            )
            cid = None
            obj_id = ObjID(vid, cid)

        elif name == "cube":
            size, color, specular = props
            radius = 0.1 * size
            vid = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                rgbaColor=color,
                halfExtents=[radius] * 3,
                physicsClientId=self.client_id,
            )
            cid = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[radius] * 3,
                physicsClientId=self.client_id,
            )
            obj_id = ObjID(vid, cid)

        elif name == "duck":
            size, color, specular = props
            shift = np.array([0, -0.02, 0]) * size
            mesh_scale = np.array([0.1, 0.1, 0.1]) * size
            vid = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName="duck.obj",
                rgbaColor=color,
                specularColor=specular,
                visualFramePosition=shift,
                meshScale=mesh_scale,
                physicsClientId=self.client_id,
            )
            cid = p.createCollisionShape(
                shapeType=p.GEOM_MESH,
                fileName="duck_vhacd.obj",
                collisionFramePosition=shift,
                meshScale=mesh_scale,
                physicsClientId=self.client_id,
            )
            obj_id = ObjID(vid, cid)
        else:
            raise NotImplementedError()

        self.objects[name][props] = obj_id
        return obj_id

    def add_ball(
        self,
        pos: Union[List, np.ndarray],
        size: float = 2.0,
        color: Tuple = (0, 1, 0, 0.5),
    ) -> int:
        props = (size, color)
        obj_id = self.get_shape("ball", props)
        body_id = p.createMultiBody(
            baseMass=0.0,
            baseInertialFramePosition=[0, 0, 0],
            baseVisualShapeIndex=obj_id.visual_id,
            basePosition=pos,
            useMaximalCoordinates=True,
            physicsClientId=self.client_id,
        )
        return body_id

    def remove_body(self, body_id):
        p.removeBody(body_id, physicsClientId=self.client_id)

    def add_cube(
        self,
        pos: Union[List, np.ndarray],
        ori: Union[List, np.ndarray] = (np.pi / 2, 0, np.pi),
        size: float = 1.0,
        color: Tuple = (0.7, 0.7, 0.7, 1.0),
        specular: Tuple = (0.7, 0.7, 0.7, 0.4),
    ) -> int:
        props = (size, color, specular)
        obj_id = self.get_shape("cube", props)
        body_id = p.createMultiBody(
            baseMass=1.0 * size,
            baseVisualShapeIndex=obj_id.visual_id,
            baseCollisionShapeIndex=obj_id.collision_id,
            basePosition=pos,
            baseOrientation=p.getQuaternionFromEuler(ori),
            useMaximalCoordinates=True,
            physicsClientId=self.client_id,
        )
        return body_id

    def add_duck(
        self,
        pos: Union[List, np.ndarray],
        ori: Union[List, np.ndarray] = (np.pi / 2, 0, np.pi),
        size: float = 1.0,
        color: Tuple = (1, 1, 1, 0.75),
        specular: Tuple = (0, 1, 1, 0.4),
    ) -> int:
        props = (size, color, specular)
        obj_id = self.get_shape("duck", props)
        body_id = p.createMultiBody(
            baseMass=1.0 * size,
            baseInertialFramePosition=[0, 0, 0],
            baseVisualShapeIndex=obj_id.visual_id,
            baseCollisionShapeIndex=obj_id.collision_id,
            basePosition=pos,
            baseOrientation=p.getQuaternionFromEuler(ori),
            useMaximalCoordinates=True,
            physicsClientId=self.client_id,
        )
        return body_id

    def change_color(self, obj_id: int, rgb_color: Tuple, link_id: int = -1):
        p.changeVisualShape(
            objectUniqueId=obj_id,
            linkIndex=link_id,
            rgbaColor=rgb_color,
            physicsClientId=self.client_id,
        )

    def plot_trajectory(self, traj):
        for i, waypoint in enumerate(traj):
            self.add_ball(waypoint, size=0.5, color=(0, 0.5, 1.0, 0.5))

    def attach(
        self,
        par_body_id: int,
        par_link_id: int,
        chi_body_id: int,
        chi_link_id: int,
        joint_type: int = p.JOINT_FIXED,
        joint_axis: Union[Tuple, List] = (0, 0, 0),
        par_frame_pos: Union[Tuple, List] = (0, 0, 0),
        chi_frame_pos: Union[Tuple, List] = (0, 0, 0),
        par_frame_ori: Union[Tuple, List] = None,
        chi_frame_ori: Union[Tuple, List] = None,
    ) -> int:
        return p.createConstraint(
            parentBodyUniqueId=par_body_id,
            parentLinkIndex=par_link_id,
            childBodyUniqueId=chi_body_id,
            childLinkIndex=chi_link_id,
            jointType=joint_type,
            jointAxis=joint_axis,
            parentFramePosition=par_frame_pos,
            childFramePosition=chi_frame_pos,
            parentFrameOrientation=par_frame_ori,
            childFrameOrientation=chi_frame_ori,
            physicsClientId=self.client_id,
        )

    def detach(self, attach_id: int):
        p.removeConstraint(attach_id, self.client_id)
