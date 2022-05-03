from abc import ABC, abstractmethod


class RewardFunction(ABC):
    """
    User-defined reward function, call method should return reward value
    """

    @abstractmethod
    def __call__(self, gridworld) -> float:
        return NotImplemented


class RewardBinary(RewardFunction):
    """
    Binary reward if goal found
    """

    def __call__(self, gridworld) -> float:
        return int(gridworld.current_pos in gridworld.reward_coords)