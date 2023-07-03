from collections import namedtuple
from os.path import abspath, dirname

import numpy as np
import pybullet as p

ROBOT_ASSETS_PATH = f"{dirname(abspath(__file__))}/assets"

JointInfo = namedtuple('JointInfo',
                       ['joint_index', 'joint_name', 'joint_type', 'q_index', 'u_index', 'flags',
                        'joint_damping', 'joint_friction', 'joint_lower_limit', 'joint_upper_limit',
                        'joint_max_force', 'joint_max_velocity', 'link_name', 'joint_axis',
                        'parent_frame_pos', 'parent_frame_orn', 'parent_index'])


def get_joint_info(body, joint, client_id) -> JointInfo:
    return JointInfo(*p.getJointInfo(body, joint, physicsClientId=client_id))


def is_fixed_joint(robot_id, joint, client_id) -> bool:
    return get_joint_info(robot_id, joint, client_id).joint_type == p.JOINT_FIXED


def is_movable_joint(robot_id, joint_id, client_id) -> bool:
    return not is_fixed_joint(robot_id, joint_id, client_id)


def get_joint_ranges(robot_id, controllable_joints, client_id):
    lower_limits, upper_limits, joint_ranges = [], [], []

    for i in controllable_joints:
        joint_info = p.getJointInfo(robot_id, i, physicsClientId=client_id)
        lb, ub = joint_info[8:10]
        r = lb - ub

        lower_limits.append(lb)
        upper_limits.append(ub)
        joint_ranges.append(r)

    return lower_limits, upper_limits, joint_ranges


def reset_joint_pos(pos, robot_id: int, client_id: int):
    indx = 0
    for i in range(p.getNumJoints(robot_id, client_id)):
        if not is_fixed_joint(robot_id, i, client_id):
            p.resetJointState(robot_id,
                              jointIndex=i,
                              targetValue=pos[indx],
                              physicsClientId=client_id)
            indx += 1


def half_angle(degree):
    n = np.round(degree / (2 * np.pi))
    degree = degree - n * 2 * np.pi

    return degree


class no_render:
    def __enter__(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

    def __exit__(self, exc_type, exc_val, exc_tb):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
