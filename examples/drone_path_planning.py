from mobrob.envs.pybullet_robots.env_editor import EnvEditor
from mobrob.envs.wrapper import get_env
from mobrob.planning.a_star import AStar, OcupancyGrid, EuclideanHeuristics
import numpy as np
from loguru import logger


class DronePlanning:
    def __init__(self, enable_gui: bool=False, n_dim: int=3, grid_each_dim: int=10, grid_size: float=0.3):
        self.env = get_env("drone", enable_gui=enable_gui)
        self.occupancy_grid_config = {
            "n_dim": n_dim,
            "grid_each_dim": grid_each_dim,
            "grid_size": grid_size
        }
        self.occupancy_grid = OcupancyGrid(**self.occupancy_grid_config)
        self.astar = AStar(self.occupancy_grid, EuclideanHeuristics())
        self.env_editor = EnvEditor(self.env.env.client_id)
        self.obs_ids = {}

        self.env.reset()
        self.env.set_pos(np.array([0, 0, 2]))
    
    def add_obstacle(self, pos: np.ndarray):
        self.occupancy_grid.add_obstacle(pos)
        obj_id = self.env_editor.add_cube(
            size=self.occupancy_grid_config["grid_size"] * 5,
            pos=pos,
            color=(1, 0, 0, 0.5),
        )
        self.obs_ids[tuple(pos)] = obj_id
    
    def remove_obstacle(self, pos: np.ndarray):
        self.occupancy_grid.remove_obstacle(pos)
        self.env_editor.remove_body(self.obs_ids[tuple(pos)])
        del self.obs_ids[tuple(pos)]
    
    def goto_pos(self, pos: np.ndarray, max_iter: int=100):
        logger.info(f"Planning a path from {self.env.get_pos()} to {pos}")
        path = self.astar.search(self.env.get_pos(), pos)

        # plot path
        path_id = []
        for p in path:
            path_id.append(self.env_editor.add_ball(
                size=1,
                pos=p,
                color=(0, 1, 0, 0.5),
            ))
        
        for pos in path:
            self.env.set_goal(pos)
            for _ in range(max_iter):
                self.env.step([0] * self.env.action_space.shape[0])
                if self.env.reached():
                    break
        
        # remove the plot path
        for pid in path_id:
            self.env_editor.remove_body(pid)

if __name__ == "__main__":
    drone_planning = DronePlanning(enable_gui=True, n_dim=3, grid_each_dim=4, grid_size=1)

    # add obstacles
    drone_planning.add_obstacle(np.array([0, 0, 0]))

    drone_planning.add_obstacle(np.array([0, 0, 3]))

    # plan a path
    for _ in range(10):
        drone_planning.goto_pos(np.array([0, 1, 1]))
        drone_planning.goto_pos(np.array([2, 1, 2]))
        drone_planning.goto_pos(np.array([3, 3, 1]))