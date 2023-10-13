from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
from loguru import logger
import time

class OcupancyGrid:
    """
    Grid map for path planning.
    The map only supports positive coordinates.
    """
    def __init__(self, n_dim: int, grid_each_dim: int, grid_size: float):
        self.n_dim = n_dim
        self.grid_each_dim = grid_each_dim
        self.grid_size = grid_size
        self.grid_map = np.zeros((grid_each_dim,) * n_dim)
        self._nx_graph = None
    
    def add_obstacle(self, obstacle: np.ndarray):
        """Add an obstacle to the grid map."""
        grid_idx = self.get_grid_idx(obstacle)
        self.grid_map[grid_idx] = 1
        self._nx_graph = None
    
    def remove_obstacle(self, obstacle: np.ndarray):
        """Remove an obstacle from the grid map."""
        grid_idx = self.get_grid_idx(obstacle)
        self.grid_map[grid_idx] = 0
        self._nx_graph = None
    
    def get_grid_idx(self, pos: np.ndarray) -> tuple:
        """Get the index of the grid that the position is in."""
        grid_idx = tuple((pos / self.grid_size).astype(int))
        # safety check with numpy
        if np.any(np.array(grid_idx) >= self.grid_each_dim) or np.any(np.array(grid_idx) < 0):
            raise ValueError("The position is out of the grid map.")

        return grid_idx

    def to_networkx(self):
        """convert the grid map to a networkx graph."""
        if self._nx_graph is None:
            start_time = time.time()
            graph = nx.Graph()
            for idx in np.ndindex(self.grid_map.shape):
                if self.grid_map[idx] == 0:
                    graph.add_node(idx)
                    for neighbor in self._get_neighbors(idx):
                        if self.grid_map[neighbor] == 0:
                            graph.add_edge(idx, neighbor)
            self._nx_graph = graph
            end_time = time.time()
            logger.info(f"Time to build the graph: {end_time - start_time}")
            logger.info(f"Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")

        return self._nx_graph
    
    def idx_to_pos(self, idx: np.ndarray) -> np.ndarray:
        """Convert the grid index to the position."""
        return idx * self.grid_size
    
    def _get_neighbors(self, idx: tuple) -> list:
        """Get all the neghbors of the given grid."""
        neighbors = []
        for n in np.ndindex((3,) * self.n_dim):
            neighbor = tuple(np.array(idx) + np.array(n) - 1)
            if neighbor != idx and np.all(np.array(neighbor) >= 0) and np.all(np.array(neighbor) < self.grid_each_dim):
                neighbors.append(neighbor)
        return neighbors

class HeuristicsBase(ABC):
    """Base class for heuristic functions."""

    @abstractmethod
    def __call__(self, neighbor, target):
        """Return the heuristic value for the given state."""
        raise NotImplementedError("This method must be implemented by a subclass.")


class EuclideanHeuristics(HeuristicsBase):
    """Euclidean distance heuristic."""

    def __call__(self, neighbor, target):
        return np.linalg.norm(np.array(neighbor) - np.array(target))


class AStar:
    def __init__(self, occupancy_grid: OcupancyGrid, heuristic: HeuristicsBase):
        self.occupancy_grid = occupancy_grid
        self.heuristic = heuristic
    
    def search(self, start: np.ndarray, goal: np.ndarray):
        """Search for a path from start to goal."""
        start_indx = self.occupancy_grid.get_grid_idx(start)
        goal_indx = self.occupancy_grid.get_grid_idx(goal)
        graph = self.occupancy_grid.to_networkx()
        path = nx.astar_path(graph, start_indx, goal_indx, heuristic=self.heuristic)

        return self.occupancy_grid.idx_to_pos(np.array(path))
        
