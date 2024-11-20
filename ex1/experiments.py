import os
import unittest

from ex1.MWCCP import MWCCPSolution, MWCCPNeighborhoods
from ex1.evaluation import MultiStats
from ex1.local_search import StepFunction
from ex1.read_instance import read_instance


class Experiments(unittest.TestCase):

    def test_deterministic_and_randomized_ch_and_GRASP(self):
        small = "../data/test_instances/small/"

        # === Deterministic CH ===
        print("============== DCH ==============")
        repetitions = 100

        obj_sum_avg = 0
        time_sum_avg = 0.0

        n_instances = 0
        for small_test in os.listdir(small):
            n_instances += 1
            path = small + small_test
            obj_inst_sum = 0
            time_inst_sum = 0.0
            for rep in range(repetitions):
                mwccp_instance = read_instance(path)
                mwccp_solution = MWCCPSolution(mwccp_instance)
                _, obj, stats = mwccp_solution.deterministic_construction_heuristic()
                obj_inst_sum += obj
                time_inst_sum += stats.get_run_time()

            obj_inst_avg = obj_inst_sum / repetitions
            time_inst_avg = time_inst_sum / repetitions

            obj_sum_avg += obj_inst_avg
            time_sum_avg += time_inst_avg

        obj_avg = obj_sum_avg / n_instances
        time_sum_avg = time_sum_avg / n_instances
        print("Average objective value: " + f"{obj_avg:.1f}")
        print("Average time : " + f"{time_sum_avg:.6f}s")
        print("============== === ==============")
        # === === === === === ===

        # === Randomized CH ===
        print("============== RCH ==============")
        repetitions = 100

        obj_sum_avg = 0
        time_sum_avg = 0.0

        n_instances = 0

        for small_test in os.listdir(small):
            n_instances += 1
            path = small + small_test
            obj_inst_sum = 0
            time_inst_sum = 0.0
            for rep in range(repetitions):
                mwccp_instance = read_instance(path)
                mwccp_solution = MWCCPSolution(mwccp_instance)
                _, obj, stats = mwccp_solution.randomized_construction_heuristic()
                obj_inst_sum += obj
                time_inst_sum += stats.get_run_time()

            obj_inst_avg = obj_inst_sum / repetitions
            time_inst_avg = time_inst_sum / repetitions

            obj_sum_avg += obj_inst_avg
            time_sum_avg += time_inst_avg

        obj_avg = obj_sum_avg / n_instances
        time_sum_avg = time_sum_avg / n_instances
        print("Average objective value: " + f"{obj_avg:.1f}")
        print("Average time : " + f"{time_sum_avg:.6f}s")
        print("============== === ==============")
        # === === === === === ===

        # === GRASP ===
        print("============== GRASP ==============")
        repetitions = 30

        obj_sum_avg = 0
        time_sum_avg = 0.0

        n_instances = 0

        neighborhood = MWCCPNeighborhoods.flip_two_adjacent_vertices
        step_function = StepFunction.best_improvement
        max_iterations = 100

        print("Repeating each instance " + str(repetitions) + " times and having a maximum number of " + str(max_iterations) + " iterations.")

        stats_list = []

        for small_test in os.listdir(small):
            n_instances += 1
            path = small + small_test
            obj_inst_sum = 0
            time_inst_sum = 0.0
            for rep in range(repetitions):
                mwccp_instance = read_instance(path)
                mwccp_solution = MWCCPSolution(mwccp_instance)
                _, stats = mwccp_solution.grasp(neighborhood, step_function, max_iterations=max_iterations)
                obj_inst_sum += stats.get_final_objective()
                time_inst_sum += stats.get_run_time()
                stats_list.append(stats)

            obj_inst_avg = obj_inst_sum / repetitions
            time_inst_avg = time_inst_sum / repetitions

            obj_sum_avg += obj_inst_avg
            time_sum_avg += time_inst_avg

        obj_avg = obj_sum_avg / n_instances
        time_sum_avg = time_sum_avg / n_instances
        print("Average objective value: " + f"{obj_avg:.1f}")
        print("Average time : " + f"{time_sum_avg:.6f}s")
        multi_stats = MultiStats(stats_list)
        multi_stats.plot_avg_obj_over_time("GRASP")
        print("============== === ==============")
        # === === === === === ===
