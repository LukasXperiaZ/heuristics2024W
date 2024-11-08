import unittest

from pymhlib.demos.graph_coloring import GCSolution
from pymhlib.permutation_solution import PermutationSolution

from ex1.MWCCP_instance import MWCCPSolution, MWCCPInstance
from ex1.read_instance import read_instance

class TestCases(unittest.TestCase):

    mwccp_instance: MWCCPInstance

    def setUp(self):
        self.mwccp_instance = read_instance("../data/test_instances/MCInstances/test")
        print(self.mwccp_instance)

    def test_loading_and_objective_function(self):
        mwccp_solution = MWCCPSolution(self.mwccp_instance)
        obj_value = mwccp_solution.calc_objective()
        print("Obj value: " + str(obj_value))
        assert(obj_value == 51)

    def test_check(self):
        mwccp_solution = MWCCPSolution(self.mwccp_instance)
        mwccp_solution.x[::-1].sort()
        print(mwccp_solution.x)

        self.assertRaises(ValueError, mwccp_solution.check)

    def test_deterministic_construction_heuristic_simple(self):
        mwccp_solution = MWCCPSolution(self.mwccp_instance)
        mwccp_solution.deterministic_construction_heuristic()
        mwccp_solution.check()
        obj_value = mwccp_solution.calc_objective()
        print("Obj value: " + str(obj_value))

    def test_deterministic_construction_heuristic_complicated(self):
        mwccp_instance = read_instance("../data/test_instances/MCInstances/test_1")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solution.deterministic_construction_heuristic()
        mwccp_solution.check()
        obj_value = mwccp_solution.calc_objective()
        print("Obj value: " + str(obj_value))

    def test_deterministic_construction_heuristic_small(self):
        mwccp_instance = read_instance("../data/test_instances/MCInstances/small/inst_50_4_00001")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solution.deterministic_construction_heuristic()
        mwccp_solution.check()
        obj_value = mwccp_solution.calc_objective()
        print("Obj value: " + str(obj_value))

    def test_deterministic_construction_heuristic_medium(self):
        mwccp_instance = read_instance("../data/test_instances/MCInstances/medium/inst_200_20_00001")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solution.deterministic_construction_heuristic()
        mwccp_solution.check()
        obj_value = mwccp_solution.calc_objective()
        print("Obj value: " + str(obj_value))

    def test_deterministic_construction_heuristic_large(self):
        mwccp_instance = read_instance("../data/test_instances/MCInstances/large/inst_1000_60_00001")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solution.deterministic_construction_heuristic()
        mwccp_solution.check()
        # TODO This takes looong
        print("Starting calc_objective()")
        obj_value = mwccp_solution.calc_objective()
        print("Obj value: " + str(obj_value))

