import os
import unittest

import numpy as np

from ex1.MWCCP import MWCCPSolution, MWCCPNeighborhoods
from ex1.evaluation import MultiStats
from ex1.local_search import StepFunction
from ex1.read_instance import read_instance


class Experiments(unittest.TestCase):

    def test_deterministic_and_randomized_ch_and_GRASP_small(self):
        directory = "../data/test_instances/small/"
        # TODO Report Mean and STD also for every instance!!!

        # === Deterministic CH ===
        print("============== DCH ==============")
        repetitions = 1
        runtimes = []
        obj_values = []
        for test in os.listdir(directory):
            if test.endswith(".pkl"):
                continue
            path = directory + test
            for rep in range(repetitions):
                mwccp_instance = read_instance(path)
                mwccp_solution = MWCCPSolution(mwccp_instance)
                _, obj, stats = mwccp_solution.deterministic_construction_heuristic()
                obj_values.append(obj)
                runtimes.append(stats.get_run_time())

        obj_avg = np.average(obj_values)
        obj_std = np.std(obj_values)
        runtime_avg = np.average(runtimes)
        runtime_std = np.std(runtimes)
        print("Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
        print("Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
        print("============== === ==============")
        # === === === === === ===

        # === Randomized CH ===
        print("============== RCH ==============")
        repetitions = 100
        runtimes = []
        obj_values = []
        for test in os.listdir(directory):
            if test.endswith(".pkl"):
                continue
            path = directory + test
            for rep in range(repetitions):
                mwccp_instance = read_instance(path)
                mwccp_solution = MWCCPSolution(mwccp_instance)
                _, obj, stats = mwccp_solution.randomized_construction_heuristic()
                obj_values.append(obj)
                runtimes.append(stats.get_run_time())

        obj_avg = np.average(obj_values)
        obj_std = np.std(obj_values)
        runtime_avg = np.average(runtimes)
        runtime_std = np.std(runtimes)
        print("Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
        print("Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
        print("============== === ==============")
        # === === === === === ===

        # === GRASP ===
        print("============== GRASP ==============")
        neighborhood = MWCCPNeighborhoods.flip_two_adjacent_vertices
        step_function = StepFunction.best_improvement
        max_iterations = 100
        repetitions = 30

        print("Repeating each instance " + str(repetitions) + " times and having a maximum number of " + str(max_iterations) + " iterations.")

        stats_list = []
        runtimes = []
        obj_values = []
        for test in os.listdir(directory):
            if test.endswith(".pkl"):
                continue
            path = directory + test
            for rep in range(repetitions):
                mwccp_instance = read_instance(path)
                mwccp_solution = MWCCPSolution(mwccp_instance)
                _, stats = mwccp_solution.grasp(neighborhood, step_function, max_iterations=max_iterations)
                obj_values.append(stats.get_final_objective())
                runtimes.append(stats.get_run_time())
                stats_list.append(stats)

        obj_avg = np.average(obj_values)
        obj_std = np.std(obj_values)
        runtime_avg = np.average(runtimes)
        runtime_std = np.std(runtimes)
        print("Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
        print("Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
        multi_stats = MultiStats(stats_list)
        multi_stats.plot_avg_obj_over_time("GRASP")
        print("============== === ==============")
        # === === === === === ===

    def test_deterministic_and_randomized_ch_and_GRASP_medium(self):
        directory = "../data/test_instances/medium/"

        # === Deterministic CH ===
        print("============== DCH ==============")
        repetitions = 1
        runtimes = []
        obj_values = []
        for test in os.listdir(directory):
            if test.endswith(".pkl"):
                continue
            path = directory + test
            for rep in range(repetitions):
                mwccp_instance = read_instance(path)
                mwccp_solution = MWCCPSolution(mwccp_instance)
                _, obj, stats = mwccp_solution.deterministic_construction_heuristic()
                obj_values.append(obj)
                runtimes.append(stats.get_run_time())

        obj_avg = np.average(obj_values)
        obj_std = np.std(obj_values)
        runtime_avg = np.average(runtimes)
        runtime_std = np.std(runtimes)
        print("Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
        print("Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
        print("============== === ==============")
        # === === === === === ===

        # === Randomized CH ===
        print("============== RCH ==============")
        repetitions = 100
        runtimes = []
        obj_values = []
        for test in os.listdir(directory):
            if test.endswith(".pkl"):
                continue
            path = directory + test
            for rep in range(repetitions):
                mwccp_instance = read_instance(path)
                mwccp_solution = MWCCPSolution(mwccp_instance)
                _, obj, stats = mwccp_solution.randomized_construction_heuristic()
                obj_values.append(obj)
                runtimes.append(stats.get_run_time())

        obj_avg = np.average(obj_values)
        obj_std = np.std(obj_values)
        runtime_avg = np.average(runtimes)
        runtime_std = np.std(runtimes)
        print("Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
        print("Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
        print("============== === ==============")
        # === === === === === ===

        # === GRASP ===
        print("============== GRASP ==============")
        neighborhood = MWCCPNeighborhoods.flip_two_adjacent_vertices
        step_function = StepFunction.best_improvement
        max_iterations = 30
        repetitions = 10

        print("Repeating each instance " + str(repetitions) + " times and having a maximum number of " + str(
            max_iterations) + " iterations.")

        stats_list = []
        runtimes = []
        obj_values = []
        for test in os.listdir(directory):
            if test.endswith(".pkl"):
                continue
            print("Running GRASP on: " + str(test))
            path = directory + test
            for rep in range(repetitions):
                mwccp_instance = read_instance(path)
                mwccp_solution = MWCCPSolution(mwccp_instance)
                _, stats = mwccp_solution.grasp(neighborhood, step_function, max_iterations=max_iterations)
                obj_values.append(stats.get_final_objective())
                runtimes.append(stats.get_run_time())
                stats_list.append(stats)

        obj_avg = np.average(obj_values)
        obj_std = np.std(obj_values)
        runtime_avg = np.average(runtimes)
        runtime_std = np.std(runtimes)
        print("Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
        print("Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
        multi_stats = MultiStats(stats_list)
        multi_stats.plot_avg_obj_over_time("GRASP")
        print("============== === ==============")
        # === === === === === ===
