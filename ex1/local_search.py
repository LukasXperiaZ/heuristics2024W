from abc import abstractmethod, ABC
from enum import Enum

class StepFunction(Enum):
    first_improvement = 1
    best_improvement = 2
    random = 3


class LocalSearchSolution(ABC):
    """
    Abstract class for a solution that can do local search.
    """

    @abstractmethod
    def get_neighbor(self, current_solution, current_obj: int, neighborhood, step_function: StepFunction):
        """
        Get the neighbor together with its objective value based on the specified neighborhood and the step function
        """
        raise NotImplementedError

    @abstractmethod
    def local_search(self, initial_solution: [int], neighborhood, step_function: StepFunction, max_iterations: int):
        """
        Run local search on the specified neighborhood and the step function.
        """
        raise NotImplementedError

