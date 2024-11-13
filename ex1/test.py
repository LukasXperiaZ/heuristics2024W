import time
import unittest

import numpy as np

from ex1.MWCCP import MWCCPSolution, MWCCPInstance, MWCCPNeighborhoods
from ex1.local_search import StepFunction
from ex1.read_instance import read_instance


class Basics(unittest.TestCase):
    mwccp_instance: MWCCPInstance

    def setUp(self):
        self.mwccp_instance = read_instance("../data/test_instances/test")
        print(self.mwccp_instance)

    def test_loading_and_objective_function(self):
        mwccp_solution = MWCCPSolution(self.mwccp_instance)
        obj_value = mwccp_solution.calc_objective()
        print("Obj value: " + str(obj_value))
        assert (obj_value == 51)

    def test_loading_and_objective_function_medium_large(self):
        print("Starting reading")
        r_start = time.time()
        mwccp_instance = read_instance("../data/test_instances/medium_large/inst_500_40_00001")
        r_end= time.time()
        print("Finished reading: " + str(r_end-r_start))

        mwccp_solution = MWCCPSolution(mwccp_instance)

        print("Starting calculating the objective value")
        r_start = time.time()
        obj_value = mwccp_solution.calc_objective()
        r_end = time.time()
        print("Finished calculating the obj function: " + str(r_end - r_start))
        print("Obj value: " + str(obj_value))

    def test_loading_and_objective_function_large(self):
        print("Starting reading")
        mwccp_instance = read_instance("../data/test_instances/large/inst_1000_60_00001")
        print("Finished reading")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        print("Starting calculating the objective value")
        obj_value = mwccp_solution.calc_objective()
        print("Obj value: " + str(obj_value))

    def test_check(self):
        mwccp_solution = MWCCPSolution(self.mwccp_instance)
        mwccp_solution.x[::-1].sort()
        print(mwccp_solution.x)

        self.assertRaises(ValueError, mwccp_solution.check)


class DCH(unittest.TestCase):

    def test_deterministic_construction_heuristic_simple(self):
        mwccp_instance = read_instance("../data/test_instances/test")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solution.deterministic_construction_heuristic()
        mwccp_solution.check()
        print("Solution: " + str(mwccp_solution.x))
        obj_value = mwccp_solution.calc_objective()
        print("Obj value: " + str(obj_value))

    def test_deterministic_construction_heuristic_complicated(self):
        mwccp_instance = read_instance("../data/test_instances/test_1")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solution.deterministic_construction_heuristic()
        mwccp_solution.check()
        print("Solution: " + str(mwccp_solution.x))
        obj_value = mwccp_solution.calc_objective()
        assert (obj_value == 28)
        print("Obj value: " + str(obj_value))

    def test_deterministic_construction_heuristic_small(self):
        mwccp_instance = read_instance("../data/test_instances/small/inst_50_4_00001")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solution.deterministic_construction_heuristic()
        mwccp_solution.check()
        print("Solution: " + str(mwccp_solution.x))
        obj_value = mwccp_solution.calc_objective()
        print("Obj value: " + str(obj_value))

    def test_deterministic_construction_heuristic_medium(self):
        mwccp_instance = read_instance("../data/test_instances/medium/inst_200_20_00001")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solution.deterministic_construction_heuristic()
        mwccp_solution.check()
        print("Solution: " + str(mwccp_solution.x))
        obj_value = mwccp_solution.calc_objective()
        print("Obj value: " + str(obj_value))

    def test_deterministic_construction_heuristic_large(self):
        print("Starting reading the instance")
        mwccp_instance = read_instance("../data/test_instances/large/inst_1000_60_00001")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        print("Starting deterministic_construction_heuristic()")
        mwccp_solution.deterministic_construction_heuristic()
        print("Starting check()")
        mwccp_solution.check()
        print("Solution: " + str(mwccp_solution.x))
        # TODO This takes looong
        print("Starting calc_objective()")
        obj_value = mwccp_solution.calc_objective()
        print("Obj value: " + str(obj_value))


class RCH(unittest.TestCase):

    def test_randomized_construction_heuristic_simple(self):
        mwccp_instance = read_instance("../data/test_instances/test")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solution.randomized_construction_heuristic()
        mwccp_solution.check()
        print("Solution: " + str(mwccp_solution.x))
        obj_value = mwccp_solution.calc_objective()
        print("Obj value: " + str(obj_value))

    def test_randomized_construction_heuristic_complicated(self):
        mwccp_instance = read_instance("../data/test_instances/test_1")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solution.randomized_construction_heuristic()
        mwccp_solution.check()
        print("Solution: " + str(mwccp_solution.x))
        obj_value = mwccp_solution.calc_objective()
        print("Obj value: " + str(obj_value))

    def test_randomized_construction_heuristic_small(self):
        mwccp_instance = read_instance("../data/test_instances/small/inst_50_4_00001")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solution.randomized_construction_heuristic()
        mwccp_solution.check()
        print("Solution: " + str(mwccp_solution.x))
        obj_value = mwccp_solution.calc_objective()
        print("Obj value: " + str(obj_value))

    def test_randomized_construction_heuristic_medium(self):
        mwccp_instance = read_instance("../data/test_instances/medium/inst_200_20_00001")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solution.randomized_construction_heuristic()
        mwccp_solution.check()
        print("Solution: " + str(mwccp_solution.x))
        obj_value = mwccp_solution.calc_objective()
        print("Obj value: " + str(obj_value))


class Neighborhoods(unittest.TestCase):

    def test_get_neighbor_flip_two_adjacent_vertices_first_improvement(self):
        mwccp_instance = read_instance("../data/test_instances/test")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solution.x = np.array([6, 7, 8, 9, 10])
        mwccp_solution.check()
        print("Solution: " + str(mwccp_solution.x))
        obj_value = mwccp_solution.calc_objective()
        print("Obj value: " + str(obj_value))
        print("----------------------")

        next_neighbor, next_obj = mwccp_solution.get_neighbor_flip_two_adjacent_vertices(mwccp_solution.x.tolist(),
                                                                                         obj_value,
                                                                                         StepFunction.first_improvement)
        print("Next neighbor: " + str(next_neighbor))
        print("Next objective: " + str(next_obj))
        assert (next_neighbor[0] == 7)
        assert (next_neighbor[1] == 6)
        assert (next_obj == 39)

    def test_get_neighbor_flip_two_adjacent_vertices_first_improvement_no_improvement_possible(self):
        mwccp_instance = read_instance("../data/test_instances/test")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solution.deterministic_construction_heuristic()
        mwccp_solution.check()
        print("Solution: " + str(mwccp_solution.x))
        obj_value = mwccp_solution.calc_objective()
        print("Obj value: " + str(obj_value))

        next_neighbor, next_obj = mwccp_solution.get_neighbor_flip_two_adjacent_vertices(mwccp_solution.x.tolist(),
                                                                                         obj_value,
                                                                                         StepFunction.first_improvement)
        assert (next_neighbor[0] == 7)
        assert (next_neighbor[1] == 8)
        assert (next_obj == 0)

    def test_get_neighbor_flip_two_adjacent_vertices_first_improvement_medium_instance(self):
        mwccp_instance = read_instance("../data/test_instances/medium/inst_200_20_00001")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solution.deterministic_construction_heuristic()
        mwccp_solution.check()
        print("Solution: " + str(mwccp_solution.x))
        obj_value = mwccp_solution.calc_objective()
        print("Obj value: " + str(obj_value))
        print("----------------------")

        next_neighbor, next_obj = mwccp_solution.get_neighbor_flip_two_adjacent_vertices(mwccp_solution.x.tolist(),
                                                                                         obj_value,
                                                                                         StepFunction.first_improvement)
        print("Next neighbor: " + str(next_neighbor))
        print("Next objective: " + str(next_obj))


class LocalSearch(unittest.TestCase):

    def test_local_search_simple_no_better_solution(self):
        mwccp_instance = read_instance("../data/test_instances/test")
        mwccp_solution = MWCCPSolution(mwccp_instance)

        solution, obj = mwccp_solution.run_local_search(MWCCPNeighborhoods.flip_two_adjacent_vertices,
                                                        StepFunction.first_improvement, 1000)

        print("----------------------")
        print("Solution after local search: " + str(solution))
        print("Objective value after local search: " + str(obj))
        assert obj == 0

    def test_local_search_medium_instance(self):
        mwccp_instance = read_instance("../data/test_instances/medium/inst_200_20_00001")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solution.deterministic_construction_heuristic()
        mwccp_solution.check()
        initial_obj_value = mwccp_solution.calc_objective()

        solution, obj = mwccp_solution.run_local_search(MWCCPNeighborhoods.flip_two_adjacent_vertices,
                                                        StepFunction.first_improvement, 1000)

        print("----------------------")
        print("Solution after local search: " + str(solution))
        print("Objective value after local search: " + str(obj))
        assert initial_obj_value > obj


class VND(unittest.TestCase):

    def test_VND(self):
        mwccp_instance = read_instance("../data/test_instances/test")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        # vnd = GVNS(mwccp_solution)
        # TODO
