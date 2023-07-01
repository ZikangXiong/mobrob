from abc import ABC, abstractmethod


class ControllerBase(ABC):
    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def control(self, goal):
        raise NotImplementedError()
