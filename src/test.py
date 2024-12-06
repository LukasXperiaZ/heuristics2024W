import random
import time
import unittest

import numpy as np

from src.MWCCP import MWCCPSolution, MWCCPInstance, MWCCPNeighborhoods
from src.evaluation import MultiStats
from src.local_search import StepFunction
from src.read_instance import read_instance


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
        r_end = time.time()
        print("Finished reading: " + str(r_end - r_start))

        mwccp_solution = MWCCPSolution(mwccp_instance)

        print("Starting calculating the objective value")
        r_start = time.time()
        obj_value = mwccp_solution.calc_objective()
        r_end = time.time()
        print("Finished calculating the obj function: " + str(r_end - r_start))
        print("Obj value: " + str(obj_value))

    def test_loading_and_objective_function_large(self):
        # TAKES LOOOONG
        print("Starting reading")
        r_start = time.time()
        mwccp_instance = read_instance("../data/test_instances/large/inst_1000_60_00001")
        r_end = time.time()
        print("Finished reading: " + str(r_end - r_start))

        mwccp_solution = MWCCPSolution(mwccp_instance)

        print("Starting calculating the objective value")
        r_start = time.time()
        obj_value = mwccp_solution.calc_objective()
        r_end = time.time()
        print("Finished calculating the obj function: " + str(r_end - r_start))
        print("Obj value: " + str(obj_value))

    def test_check(self):
        mwccp_solution = MWCCPSolution(self.mwccp_instance)
        mwccp_solution.x[::-1].sort()
        print(mwccp_solution.x)

        self.assertRaises(ValueError, mwccp_solution.check)

    def test_is_valid_solution(self):
        mwccp_solution = MWCCPSolution(self.mwccp_instance)
        mwccp_solution.check()
        assert mwccp_solution.is_valid_solution(mwccp_solution.x.tolist())

    def test_is_valid_solution_neg(self):
        mwccp_solution = MWCCPSolution(self.mwccp_instance)
        mwccp_solution.x[::-1].sort()
        assert not mwccp_solution.is_valid_solution(mwccp_solution.x.tolist())


class DCH(unittest.TestCase):

    def test_deterministic_construction_heuristic_simple(self):
        mwccp_instance = read_instance("../data/test_instances/test")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        sol, obj_value, stats = mwccp_solution.deterministic_construction_heuristic()
        print("Solution: " + str(sol))
        print("Obj value: " + str(obj_value))

    def test_deterministic_construction_heuristic_complicated(self):
        mwccp_instance = read_instance("../data/test_instances/test_1")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        sol, obj_value, stats = mwccp_solution.deterministic_construction_heuristic()
        print("Solution: " + str(sol))
        assert (obj_value == 28)
        print("Obj value: " + str(obj_value))

    def test_deterministic_construction_heuristic_small(self):
        mwccp_instance = read_instance("../data/test_instances/small/inst_50_4_00001")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        sol, obj_value, stats = mwccp_solution.deterministic_construction_heuristic()
        print("Solution: " + str(sol))
        print("Obj value: " + str(obj_value))

    def test_deterministic_construction_heuristic_medium(self):
        mwccp_instance = read_instance("../data/test_instances/medium/inst_200_20_00001")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        sol, obj_value, stats = mwccp_solution.deterministic_construction_heuristic()
        print("Solution: " + str(sol))
        print("Obj value: " + str(obj_value))

    def test_deterministic_construction_heuristic_medium_large(self):
        mwccp_instance = read_instance("../data/test_instances/medium_large/inst_500_40_00001")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        sol, obj_value, stats = mwccp_solution.deterministic_construction_heuristic()
        print("Solution: " + str(sol))
        print("Obj value: " + str(obj_value))

    """
    def test_deterministic_construction_heuristic_large(self):
        # TAKES LOOONG
        print("Starting reading the instance")
        mwccp_instance = read_instance("../data/test_instances/large/inst_1000_60_00001")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        print("Starting deterministic_construction_heuristic()")
        mwccp_solution.deterministic_construction_heuristic()
        print("Starting check()")
        mwccp_solution.check()
        print("Solution: " + str(mwccp_solution.x))
        print("Starting calc_objective()")
        obj_value = mwccp_solution.calc_objective()
        print("Obj value: " + str(obj_value))
    """


class RCH(unittest.TestCase):

    def test_randomized_construction_heuristic_simple(self):
        mwccp_instance = read_instance("../data/test_instances/test")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        sol, obj_value, stats = mwccp_solution.randomized_construction_heuristic()
        print("Solution: " + str(sol))
        print("Obj value: " + str(obj_value))

    def test_randomized_construction_heuristic_complicated(self):
        mwccp_instance = read_instance("../data/test_instances/test_1")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        sol, obj_value, stats = mwccp_solution.randomized_construction_heuristic()
        print("Solution: " + str(sol))
        print("Obj value: " + str(obj_value))

    def test_randomized_construction_heuristic_small(self):
        mwccp_instance = read_instance("../data/test_instances/small/inst_50_4_00001")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        sol, obj_value, stats = mwccp_solution.randomized_construction_heuristic()
        print("Solution: " + str(sol))
        print("Obj value: " + str(obj_value))

    def test_randomized_construction_heuristic_medium(self):
        mwccp_instance = read_instance("../data/test_instances/medium/inst_200_20_00001")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        sol, obj_value, stats = mwccp_solution.randomized_construction_heuristic()
        print("Solution: " + str(sol))
        print("Obj value: " + str(obj_value))

    def test_randomized_construction_heuristic_medium_large(self):
        mwccp_instance = read_instance("../data/test_instances/medium_large/inst_500_40_00001")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        sol, obj_value, stats = mwccp_solution.randomized_construction_heuristic()
        print("Solution: " + str(sol))
        print("Obj value: " + str(obj_value))

    def test_rand_const_heu_variance(self):
        mwccp_instance = read_instance("../data/test_instances/small/inst_50_4_00001")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        obj_values = []
        for i in range(0, 100):
            sol, obj_value, stats = mwccp_solution.randomized_construction_heuristic()
            obj_values.append(obj_value)

        obj_avg = np.average(obj_values)
        obj_std = np.std(obj_values)
        print("Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")

    def test_rand_const_heu_diff_structures(self):
        mwccp_instance = read_instance("../data/test_instances/small/inst_50_4_00001")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        for i in range(0, 20):
            sol, obj_value, stats = mwccp_solution.randomized_construction_heuristic()
            print(sol)


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

        next_neighbor, next_obj = mwccp_solution.get_neighbor(mwccp_solution.x.tolist(),
                                                              obj_value, MWCCPNeighborhoods.flip_two_adjacent_vertices,
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
        print("Solution: " + str(mwccp_solution.x))
        obj_value = mwccp_solution.calc_objective()
        print("Obj value: " + str(obj_value))

        next_neighbor, next_obj = mwccp_solution.get_neighbor(mwccp_solution.x.tolist(),
                                                              obj_value, MWCCPNeighborhoods.flip_two_adjacent_vertices,
                                                              StepFunction.first_improvement)
        assert (next_neighbor[0] == 7)
        assert (next_neighbor[1] == 8)
        # Make sure that a higher value is returned since there does not exist a better solution
        assert (next_obj > obj_value)

    def test_get_neighbor_flip_two_adjacent_vertices_first_improvement_medium_instance(self):
        mwccp_instance = read_instance("../data/test_instances/medium/inst_200_20_00001")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solution.deterministic_construction_heuristic()
        print("Solution: " + str(mwccp_solution.x))
        obj_value = mwccp_solution.calc_objective()
        print("Obj value: " + str(obj_value))
        print("----------------------")

        next_neighbor, next_obj = mwccp_solution.get_neighbor(mwccp_solution.x.tolist(),
                                                              obj_value, MWCCPNeighborhoods.flip_two_adjacent_vertices,
                                                              StepFunction.first_improvement)
        print("Next neighbor: " + str(next_neighbor))
        print("Next objective: " + str(next_obj))


class LocalSearch(unittest.TestCase):

    def test_local_search_simple_no_better_solution(self):
        mwccp_instance = read_instance("../data/test_instances/test")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solution.deterministic_construction_heuristic()
        initial_solution = mwccp_solution.x.tolist()

        iterations = 100

        print("\n----------- First improvement -----------")
        solution_first, _, stats_first = mwccp_solution.local_search(initial_solution,
                                                                     MWCCPNeighborhoods.flip_two_adjacent_vertices,
                                                                     StepFunction.first_improvement,
                                                                     max_iterations=iterations)
        print("Solution after local search: " + str(solution_first))
        stats_first.show_plot("test")
        assert stats_first.final_objective == 0

        print("\n----------- Best Improvement -----------")
        solution_best, _, stats_best = mwccp_solution.local_search(initial_solution,
                                                                   MWCCPNeighborhoods.flip_two_adjacent_vertices,
                                                                   StepFunction.best_improvement,
                                                                   max_iterations=iterations)
        print("Solution after local search: " + str(solution_best))
        stats_best.show_plot("test")
        assert stats_best.final_objective == 0

        print("\n----------- Random Improvement -----------")
        solution_rand, _, stats_rand = mwccp_solution.local_search(initial_solution,
                                                                   MWCCPNeighborhoods.flip_two_adjacent_vertices,
                                                                   StepFunction.random, max_iterations=iterations)
        print("Solution after local search: " + str(solution_rand))
        stats_rand.show_plot("test")
        assert stats_rand.final_objective == 0

    def test_local_search_simple_local_maximum_already_reached(self):
        mwccp_instance = read_instance("../data/test_instances/test_1")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solution.deterministic_construction_heuristic()
        initial_solution = mwccp_solution.x.tolist()

        iterations = 100

        print("\n----------- First improvement -----------")
        solution_first, _, stats_first = mwccp_solution.local_search(initial_solution,
                                                                     MWCCPNeighborhoods.flip_two_adjacent_vertices,
                                                                     StepFunction.first_improvement,
                                                                     max_iterations=iterations)
        print("Solution after local search: " + str(solution_first))
        stats_first.show_plot("test_1")
        assert stats_first.final_objective == 28

        print("\n----------- Best Improvement -----------")
        solution_best, _, stats_best = mwccp_solution.local_search(initial_solution,
                                                                   MWCCPNeighborhoods.flip_two_adjacent_vertices,
                                                                   StepFunction.best_improvement,
                                                                   max_iterations=iterations)
        print("Solution after local search: " + str(solution_best))
        stats_best.show_plot("test_1")
        assert stats_best.final_objective == 28

        print("\n----------- Random Improvement -----------")
        solution_rand, _, stats_rand = mwccp_solution.local_search(initial_solution,
                                                                   MWCCPNeighborhoods.flip_two_adjacent_vertices,
                                                                   StepFunction.random, max_iterations=iterations)
        print("Solution after local search: " + str(solution_rand))
        stats_rand.show_plot("test_1")
        assert stats_rand.final_objective == 28

    def test_local_search_small_1(self):
        mwccp_instance = read_instance("../data/test_instances/small/inst_50_4_00001")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solution.deterministic_construction_heuristic()
        initial_solution = mwccp_solution.x.tolist()

        iterations = 100

        print("\n----------- First improvement -----------")
        solution_first, _, stats_first = mwccp_solution.local_search(initial_solution,
                                                                     MWCCPNeighborhoods.flip_two_adjacent_vertices,
                                                                     StepFunction.first_improvement,
                                                                     max_iterations=iterations)
        print("Solution after local search: " + str(solution_first))
        stats_first.show_plot("inst_50_4_00001")

        print("\n----------- Best Improvement -----------")
        solution_best, _, stats_best = mwccp_solution.local_search(initial_solution,
                                                                   MWCCPNeighborhoods.flip_two_adjacent_vertices,
                                                                   StepFunction.best_improvement,
                                                                   max_iterations=iterations)
        print("Solution after local search: " + str(solution_best))
        stats_best.show_plot("inst_50_4_00001")

        print("\n----------- Random Improvement -----------")
        solution_rand, _, stats_rand = mwccp_solution.local_search(initial_solution,
                                                                   MWCCPNeighborhoods.flip_two_adjacent_vertices,
                                                                   StepFunction.random, max_iterations=iterations)
        print("Solution after local search: " + str(solution_rand))
        stats_rand.show_plot("inst_50_4_00001")

        multi_stats = MultiStats([stats_first, stats_best, stats_rand])
        multi_stats.plot_stats("inst_50_4_00001, local search")

    def test_local_search_medium_1(self):
        mwccp_instance = read_instance("../data/test_instances/medium/inst_200_20_00001")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solution.deterministic_construction_heuristic()
        initial_solution = mwccp_solution.x.tolist()

        max_time = 5

        print("\n----------- First improvement -----------")
        solution_first, _, stats_first = mwccp_solution.local_search(initial_solution,
                                                                     MWCCPNeighborhoods.flip_two_adjacent_vertices,
                                                                     StepFunction.first_improvement,
                                                                     max_time_in_s=max_time)
        print("Solution after local search: " + str(solution_first))

        print("\n----------- Best Improvement -----------")
        solution_best, _, stats_best = mwccp_solution.local_search(initial_solution,
                                                                   MWCCPNeighborhoods.flip_two_adjacent_vertices,
                                                                   StepFunction.best_improvement,
                                                                   max_time_in_s=max_time)
        print("Solution after local search: " + str(solution_best))

        print("\n----------- Random Improvement -----------")
        solution_rand, _, stats_rand = mwccp_solution.local_search(initial_solution,
                                                                   MWCCPNeighborhoods.flip_two_adjacent_vertices,
                                                                   StepFunction.random, max_time_in_s=max_time)
        print("Solution after local search: " + str(solution_rand))

        multi_stats = MultiStats([stats_first, stats_best, stats_rand])
        multi_stats.plot_stats("inst_200_20_00001, local search")

    def test_local_search_different_neighborhoods_small(self):
        mwccp_instance = read_instance("../data/test_instances/small/inst_50_4_00001")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solution.deterministic_construction_heuristic()
        initial_solution = mwccp_solution.x.tolist()

        iterations = 100

        print("\n----------- Flip Two -----------")
        _, _, stats_two = mwccp_solution.local_search(initial_solution,
                                                      MWCCPNeighborhoods.flip_two_adjacent_vertices,
                                                      StepFunction.best_improvement,
                                                      max_iterations=iterations)

        print("\n----------- Flip Three -----------")
        _, _, stats_three = mwccp_solution.local_search(initial_solution,
                                                        MWCCPNeighborhoods.flip_three_adjacent_vertices,
                                                        StepFunction.best_improvement,
                                                        max_iterations=iterations)

        print("\n----------- Flip Four -----------")
        _, _, stats_four = mwccp_solution.local_search(initial_solution,
                                                       MWCCPNeighborhoods.flip_four_adjacent_vertices,
                                                       StepFunction.best_improvement,
                                                       max_iterations=iterations)

        multi_stats = MultiStats([stats_two, stats_three, stats_four])
        multi_stats.plot_stats("inst_50_4_00001, local search")

    def test_local_search_different_neighborhoods_medium_large(self):
        mwccp_instance = read_instance("../data/test_instances/medium_large/inst_500_40_00004")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solution.deterministic_construction_heuristic()
        initial_solution = mwccp_solution.x.tolist()

        max_time = 5

        print("\n----------- Flip Two -----------")
        _, _, stats_two = mwccp_solution.local_search(initial_solution,
                                                      MWCCPNeighborhoods.flip_two_adjacent_vertices,
                                                      StepFunction.best_improvement,
                                                      max_time_in_s=max_time)

        print("\n----------- Flip Three -----------")
        _, _, stats_three = mwccp_solution.local_search(initial_solution,
                                                        MWCCPNeighborhoods.flip_three_adjacent_vertices,
                                                        StepFunction.best_improvement,
                                                        max_time_in_s=max_time)

        print("\n----------- Flip Four -----------")
        _, _, stats_four = mwccp_solution.local_search(initial_solution,
                                                       MWCCPNeighborhoods.flip_four_adjacent_vertices,
                                                       StepFunction.best_improvement,
                                                       max_time_in_s=max_time)

        multi_stats = MultiStats([stats_two, stats_three, stats_four])
        multi_stats.plot_stats("inst_50_4_00001, local search")

    def test_local_search_medium_1_time_constraint(self):
        mwccp_instance = read_instance("../data/test_instances/medium/inst_200_20_00001")
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solution.deterministic_construction_heuristic()
        initial_solution = mwccp_solution.x.tolist()

        max_time = 1

        print("\n----------- First improvement -----------")
        solution_first, _, stats_first = mwccp_solution.local_search(initial_solution,
                                                                     MWCCPNeighborhoods.flip_two_adjacent_vertices,
                                                                     StepFunction.first_improvement,
                                                                     max_time_in_s=max_time)
        print("Solution after local search: " + str(solution_first))

        print("\n----------- Best Improvement -----------")
        solution_best, _, stats_best = mwccp_solution.local_search(initial_solution,
                                                                   MWCCPNeighborhoods.flip_two_adjacent_vertices,
                                                                   StepFunction.best_improvement,
                                                                   max_time_in_s=max_time)
        print("Solution after local search: " + str(solution_best))

        print("\n----------- Random Improvement -----------")
        solution_rand, _, stats_rand = mwccp_solution.local_search(initial_solution,
                                                                   MWCCPNeighborhoods.flip_two_adjacent_vertices,
                                                                   StepFunction.random, max_time_in_s=max_time)
        print("Solution after local search: " + str(solution_rand))

        multi_stats = MultiStats([stats_first, stats_best, stats_rand])
        multi_stats.plot_stats("inst_200_20_00001, local search")


class VND(unittest.TestCase):

    def test_VND_only_one_neighborhood_small(self):
        mwccp_instance = read_instance("../data/test_instances/small/inst_50_4_00002")
        mwccp_solution = MWCCPSolution(mwccp_instance)

        max_iterations = 500
        solution_best, stats_best = mwccp_solution.vnd([MWCCPNeighborhoods.flip_two_adjacent_vertices],
                                                       StepFunction.best_improvement, max_iterations=max_iterations)
        stats_best.show_plot("inst_50_4_00002")

    def test_VND_only_one_neighborhood_medium(self):
        mwccp_instance = read_instance("../data/test_instances/medium/inst_200_20_00002")
        mwccp_solution = MWCCPSolution(mwccp_instance)

        max_iterations = 10000
        solution_best, stats_best = mwccp_solution.vnd([MWCCPNeighborhoods.flip_two_adjacent_vertices],
                                                       StepFunction.best_improvement, max_iterations=max_iterations)
        stats_best.show_plot("inst_200_20_00002")

    def test_VND_only_one_neighborhood_time_constraint(self):
        mwccp_instance = read_instance("../data/test_instances/medium_large/inst_500_40_00001")
        mwccp_solution = MWCCPSolution(mwccp_instance)

        max_time = 10
        solution_best, stats_best = mwccp_solution.vnd([MWCCPNeighborhoods.flip_two_adjacent_vertices],
                                                       StepFunction.best_improvement, max_time_in_s=max_time)
        stats_best.show_plot("inst_500_40_00001")

    def test_VND_small(self):
        mwccp_instance = read_instance("../data/test_instances/small/inst_50_4_00002")
        mwccp_solution = MWCCPSolution(mwccp_instance)

        # max_time = 10
        max_iterations = 500
        solution_best, stats_best = mwccp_solution.vnd(
            [MWCCPNeighborhoods.flip_two_adjacent_vertices, MWCCPNeighborhoods.flip_three_adjacent_vertices,
             MWCCPNeighborhoods.flip_four_adjacent_vertices],
            StepFunction.best_improvement, max_iterations=max_iterations)  # max_time_in_s=max_time)
        stats_best.show_plot("inst_50_4_00002")

    def test_VND_medium(self):
        mwccp_instance = read_instance("../data/test_instances/medium/inst_200_20_00004")
        mwccp_solution = MWCCPSolution(mwccp_instance)

        max_time = 5
        solution_best, stats_best = mwccp_solution.vnd(
            [MWCCPNeighborhoods.flip_two_adjacent_vertices, MWCCPNeighborhoods.flip_three_adjacent_vertices,
             MWCCPNeighborhoods.flip_four_adjacent_vertices],
            StepFunction.best_improvement, max_time_in_s=max_time)
        stats_best.show_plot("inst_200_20_00004")

    def test_VND_medium_large(self):
        mwccp_instance = read_instance("../data/test_instances/medium_large/inst_500_40_00010")
        mwccp_solution = MWCCPSolution(mwccp_instance)

        max_time = 10
        solution_best, stats_best = mwccp_solution.vnd(
            [MWCCPNeighborhoods.flip_two_adjacent_vertices, MWCCPNeighborhoods.flip_three_adjacent_vertices,
             MWCCPNeighborhoods.flip_four_adjacent_vertices],
            StepFunction.best_improvement, max_time_in_s=max_time)
        stats_best.show_plot("inst_500_40_00010")


class GRASP(unittest.TestCase):
    def test_GRASP_small(self):
        mwccp_instance = read_instance("../data/test_instances/small/inst_50_4_00002")
        mwccp_solution = MWCCPSolution(mwccp_instance)

        max_time = 10
        solution_best, stats_best = mwccp_solution.grasp(MWCCPNeighborhoods.flip_two_adjacent_vertices,
                                                         StepFunction.best_improvement, max_time_in_s=max_time)
        stats_best.show_plot("inst_50_4_00002")

    def test_GRASP_medium(self):
        mwccp_instance = read_instance("../data/test_instances/medium/inst_200_20_00002")
        mwccp_solution = MWCCPSolution(mwccp_instance)

        # NOTE: The larger the instances get, the longer one iteration is going to last.
        max_time = 10
        solution_best, stats_best = mwccp_solution.grasp(MWCCPNeighborhoods.flip_two_adjacent_vertices,
                                                         StepFunction.best_improvement, max_time_in_s=max_time)
        stats_best.show_plot("inst_200_20_00002")


class GeneticAlgorithm(unittest.TestCase):
    def test_tournament_selection(self):
        mwccp_instance = read_instance("../data/test_instances/small/inst_50_4_00002")
        mwccp_solution = MWCCPSolution(mwccp_instance)

        test_population = []
        for i in range(100):
            test_population.append(([], random.randint(1, 100)))

        selected_individuals = mwccp_solution.tournament_selection(test_population, 10)
        print(selected_individuals)
        integers = [value for _, value in selected_individuals]
        average = sum(integers) / len(integers) if integers else 0
        print(average)
        # Average is lower if k is closer to the number of individuals. Thus, works as intended

    def test_partially_matched_crossover(self):
        mwccp_instance = read_instance("../data/test_instances/small/inst_50_4_00002")
        mwccp_solution = MWCCPSolution(mwccp_instance)

        top = [([9, 8, 4, 5, 6, 7, 1, 3, 2, 0], 10)]
        rest = [([8, 7, 1, 2, 3, 0, 9, 5, 4, 6], 9)]
        mid = mwccp_solution.partially_matched_crossover(top, rest, 2, 3, 1.5)
        print(mid)

    def test_partially_matched_crossover_small(self):
        mwccp_instance = read_instance("../data/test_instances/small/inst_50_4_00002")
        mwccp_solution = MWCCPSolution(mwccp_instance)

        sol_1, obj_1, _ = mwccp_solution.randomized_construction_heuristic()
        sol_2, obj_2, _ = mwccp_solution.randomized_construction_heuristic()

        top = [(sol_1, obj_1)]
        rest = [(sol_2, obj_2)]
        children = mwccp_solution.partially_matched_crossover(top, rest, 2, 3, 1.5)

        print("Valid solution: " + str(mwccp_solution.is_valid_solution(children[0][0])))
        print("Valid solution: " + str(mwccp_solution.is_valid_solution(children[1][0])))
        assert not mwccp_solution.has_duplicates(children[0][0])
        assert not mwccp_solution.has_duplicates(children[1][0])
        print(children)

    def test_partially_matched_crossover_problem(self):
        mwccp_instance = read_instance("../data/test_instances/small/inst_50_4_00002")
        mwccp_solution = MWCCPSolution(mwccp_instance)

        for i in range(10000):
            sol_1, obj_1, _ = mwccp_solution.randomized_construction_heuristic()
            sol_2, obj_2, _ = mwccp_solution.randomized_construction_heuristic()

            top = [(sol_1, obj_1)]
            rest = [(sol_2, obj_2)]
            children = mwccp_solution.partially_matched_crossover(top, rest, 2, 3, 1.5)

            assert not mwccp_solution.has_duplicates(children[0][0])
            assert not mwccp_solution.has_duplicates(children[1][0])

    def test_insertion_mutation(self):
        mwccp_instance = read_instance("../data/test_instances/small/inst_50_4_00002")
        mwccp_solution = MWCCPSolution(mwccp_instance)

        test_population = [([9, 8, 4, 5, 6, 7, 1, 3, 2, 0], 10), ([8, 7, 1, 2, 3, 0, 9, 5, 4, 6], 9)]
        mutated_population = mwccp_solution.insertion_mutation(test_population, 0.1, 1.5)
        print(mutated_population)

    def test_insertion_mutation_small(self):
        mwccp_instance = read_instance("../data/test_instances/small/inst_50_4_00002")
        mwccp_solution = MWCCPSolution(mwccp_instance)

        sol_1, obj_1, _ = mwccp_solution.randomized_construction_heuristic()
        sol_2, obj_2, _ = mwccp_solution.randomized_construction_heuristic()

        test_population = [(sol_1, obj_1), (sol_2, obj_2)]
        mutated_population = mwccp_solution.insertion_mutation(test_population, 0.5, 1.2)

        print("Valid solution: " + str(mwccp_solution.is_valid_solution(mutated_population[0][0])))
        print("Valid solution: " + str(mwccp_solution.is_valid_solution(mutated_population[1][0])))
        assert not mwccp_solution.has_duplicates(mutated_population[0][0])
        assert not mwccp_solution.has_duplicates(mutated_population[1][0])
        print(mutated_population)

    def test_repair_small(self):
        mwccp_instance = read_instance("../data/test_instances/small/inst_50_4_00002")
        mwccp_solution = MWCCPSolution(mwccp_instance)

        sol_1, obj_1, _ = mwccp_solution.randomized_construction_heuristic()
        sol_2, obj_2, _ = mwccp_solution.randomized_construction_heuristic()

        top = [(sol_1, obj_1)]
        rest = [(sol_2, obj_2)]
        children = mwccp_solution.partially_matched_crossover(top, rest, 2, 3, 1.5)

        print("=== After partially_matched_crossover ===")
        print("Valid solution: " + str(mwccp_solution.is_valid_solution(children[0][0])))
        print("Valid solution: " + str(mwccp_solution.is_valid_solution(children[1][0])))
        print(children)
        print("=== === === === === === === === === === ===")

        repaired_population = mwccp_solution.repair(children, 1)
        print("=== After repair ===")
        print("Valid solution: " + str(mwccp_solution.is_valid_solution(repaired_population[0][0])))
        print("Valid solution: " + str(mwccp_solution.is_valid_solution(repaired_population[1][0])))
        print(repaired_population)
        print("=== === === === === ===")

    def test_replacement_brkga(self):
        mwccp_instance = read_instance("../data/test_instances/small/inst_50_4_00002")
        mwccp_solution = MWCCPSolution(mwccp_instance)

        top = [([1], 3), ([2], 4), ([3], 8), ([4], 7), ([], 20), ([], 20), ([], 20), ([], 20), ([], 20), ([], 20)]
        mid = [([11], 6), ([12], 11), ([13], 5), ([14], 9), ([], 20), ([], 20), ([], 20), ([], 20), ([], 20), ([], 20)]

        replaced_population = mwccp_solution.replacement_brkga(top, mid, 25)
        print(replaced_population)

    def test_genetic_algorithm(self):
        mwccp_instance = read_instance("../data/test_instances/small/inst_50_4_00002")
        mwccp_solution = MWCCPSolution(mwccp_instance)

        best_sol, stats = mwccp_solution.genetic_algorithm(population_size=1000)
        stats.show_plot("Default")

        best_sol, stats = mwccp_solution.genetic_algorithm(population_size=1000, randomized_const_heuristic_initialization="standard")
        stats.show_plot("Standard RCH")

    def test_genetic_algorithm_medium(self):
        mwccp_instance = read_instance("../data/test_instances/medium/inst_200_20_00004")
        mwccp_solution = MWCCPSolution(mwccp_instance)

        best_sol, stats = mwccp_solution.genetic_algorithm(population_size=200, k=50, max_time_in_s=20)
        stats.show_plot("Default")

        best_sol, stats = mwccp_solution.genetic_algorithm(population_size=200, k=50, max_time_in_s=20, randomized_const_heuristic_initialization="standard")
        stats.show_plot("Standard RCH")

class GeneticAlgorithmWithVND(unittest.TestCase):
    def test_genetic_algorithm_with_VND(self):
        mwccp_instance = read_instance("../data/test_instances/small/inst_50_4_00002")
        mwccp_solution = MWCCPSolution(mwccp_instance)

        best_sol, stats = mwccp_solution.genetic_algorithm_with_vnd(population_size=1000)
        stats.show_plot("+VND")
