import time

import numpy as np

from mobrob.envs.wrapper import get_env


def random_fallback():
    env = get_env("turtlebot3premitive", enable_gui=True, terminate_on_goal=True)

    env.reset()

    actions = [
        "forward",
        "backward",
        "spin_clockwise",
        "spin_counter_clockwise",
    ]

    for _ in range(1000):
        cmd = env.env.robot.prop_ctrl(env.get_goal(), np.array([1, 1]))
        env.env.robot.ctrl(np.array(cmd))
        ray_obs = env.env.robot.get_ray_obs()

        if env.reached():
            env.set_goal(env.goal_space.sample())

        if np.min(ray_obs) < 0.2:
            action = np.random.choice(actions)

            ray_obs = env.env.robot.get_ray_obs()
            cur_min_dist = np.min(ray_obs)

            for _ in range(3):
                if action == "forward":
                    env.move_forward()
                elif action == "backward":
                    env.move_backward()
                elif action == "spin_clockwise":
                    env.spin_clockwise()
                elif action == "spin_counter_clockwise":
                    env.spin_counter_clockwise()

            ray_obs = env.env.robot.get_ray_obs()
            new_min_dist = np.min(ray_obs)

            if new_min_dist < cur_min_dist:
                for _ in range(8):
                    if action == "forward":
                        env.move_backward()
                    elif action == "backward":
                        env.move_forward()
                    elif action == "spin_clockwise":
                        env.spin_counter_clockwise()
                    elif action == "spin_counter_clockwise":
                        env.spin_clockwise()
            else:
                for _ in range(2):
                    if action == "forward":
                        env.move_forward()
                    elif action == "backward":
                        env.move_backward()
                    elif action == "spin_clockwise":
                        env.spin_clockwise()
                    elif action == "spin_counter_clockwise":
                        env.spin_counter_clockwise()


def to_sparse_fallback():
    env = get_env("turtlebot3premitive", enable_gui=True, terminate_on_goal=True)

    env.reset()

    for _ in range(1000):
        ray_obs = env.env.robot.get_ray_obs()

        if env.reached():
            env.set_goal(env.goal_space.sample())

        cmd = env.env.robot.prop_ctrl(env.get_goal(), np.array([1, 1]))
        if np.min(ray_obs) < 0.3:
            # find most sparse direction in ray_obs
            window_size = 9
            direction_mean = []
            for i in range(4):
                direction_mean.append(
                    np.mean(ray_obs[i * window_size : (i + 1) * window_size])
                )
            direction_mean = np.array(direction_mean)
            worst_dir = np.argmin(direction_mean)

            # import ipdb; ipdb.set_trace()  # fmt: skip
            if worst_dir == 0:
                cmd_delta = np.array([-cmd[0], 0.5])
            elif worst_dir == 1:
                cmd_delta = np.array([0.0, 0.0])
            elif worst_dir == 2:
                cmd_delta = np.array([0.0, 0.0])
            elif worst_dir == 3:
                cmd_delta = np.array([-cmd[0], -0.5])
        else:
            cmd_delta = np.array([0.0, 0.0])

        env.env.robot.ctrl(np.array(cmd) + cmd_delta)
        time.sleep(1 / 50)


if __name__ == "__main__":
    # random_fallback()
    to_sparse_fallback()
