from pymhlib.demos.graph_coloring import GCSolution
from pymhlib.permutation_solution import PermutationSolution

from ex1.MWCCP_instance import MWCCPSolution
from ex1.read_instance import read_instance

if __name__ == '__main__':
    mwccp_instance = read_instance("../data/test_instances/MCInstances/test")
    print(mwccp_instance)

    mwccp_solution = MWCCPSolution(mwccp_instance)
    obj_value = mwccp_solution.calc_objective()
    print(obj_value)
