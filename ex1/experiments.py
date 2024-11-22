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

        # === Deterministic CH ===
        print("============== DCH ==============")
        repetitions = 1
        runtimes = []
        obj_values = []
        for test in os.listdir(directory):
            if test.endswith(".pkl"):
                continue
            path = directory + test
            obj_values_inst = []
            runtimes_inst = []
            for rep in range(repetitions):
                mwccp_instance = read_instance(path)
                mwccp_solution = MWCCPSolution(mwccp_instance)
                _, obj, stats = mwccp_solution.deterministic_construction_heuristic()
                obj_values_inst.append(obj)
                runtimes_inst.append(stats.get_run_time())

            obj_avg = np.mean(obj_values_inst)
            obj_std = np.std(obj_values_inst)
            runtime_avg = np.mean(runtimes_inst)
            runtime_std = np.std(runtimes_inst)
            print("#################### " + str(test) + " ####################")
            print("Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
            print("Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
            print("#################### #################### ####################")
            obj_values.append(obj_values_inst)
            runtimes.append(runtimes_inst)

        obj_avg = np.average(obj_values)
        obj_std = np.std(obj_values)
        runtime_avg = np.average(runtimes)
        runtime_std = np.std(runtimes)
        print("---------------------- Values over all instances ----------------------")
        print("Mean Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
        print("Mean Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
        print("---------------------- ---------------------- ----------------------")
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
            obj_values_inst = []
            runtimes_inst = []
            for rep in range(repetitions):
                mwccp_instance = read_instance(path)
                mwccp_solution = MWCCPSolution(mwccp_instance)
                _, obj, stats = mwccp_solution.randomized_construction_heuristic()
                obj_values_inst.append(obj)
                runtimes_inst.append(stats.get_run_time())

            obj_avg = np.mean(obj_values_inst)
            obj_std = np.std(obj_values_inst)
            runtime_avg = np.mean(runtimes_inst)
            runtime_std = np.std(runtimes_inst)
            print("#################### " + str(test) + " ####################")
            print("Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
            print("Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
            print("#################### #################### ####################")
            obj_values.append(obj_values_inst)
            runtimes.append(runtimes_inst)

        obj_avg = np.average(obj_values)
        obj_std = np.std(obj_values)
        runtime_avg = np.average(runtimes)
        runtime_std = np.std(runtimes)
        print("---------------------- Values over all instances ----------------------")
        print("Mean Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
        print("Mean Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
        print("---------------------- ---------------------- ----------------------")
        print("============== === ==============")
        # === === === === === ===

        # === GRASP ===
        print("============== GRASP ==============")
        neighborhood = MWCCPNeighborhoods.flip_two_adjacent_vertices
        step_function = StepFunction.best_improvement
        max_iterations = 100
        repetitions = 30

        print("Repeating each instance " + str(repetitions) + " times and having a maximum number of " + str(
            max_iterations) + " iterations.")

        stats_list = []
        runtimes = []
        obj_values = []
        for test in os.listdir(directory):
            if test.endswith(".pkl"):
                continue
            path = directory + test
            obj_values_inst = []
            runtimes_inst = []
            for rep in range(repetitions):
                mwccp_instance = read_instance(path)
                mwccp_solution = MWCCPSolution(mwccp_instance)
                _, stats = mwccp_solution.grasp(neighborhood, step_function, max_iterations=max_iterations)
                obj_values_inst.append(stats.get_final_objective())
                runtimes_inst.append(stats.get_run_time())
                stats_list.append(stats)

            obj_avg = np.mean(obj_values_inst)
            obj_std = np.std(obj_values_inst)
            runtime_avg = np.mean(runtimes_inst)
            runtime_std = np.std(runtimes_inst)
            print("#################### " + str(test) + " ####################")
            print("Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
            print("Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
            print("#################### #################### ####################")
            obj_values.append(obj_values_inst)
            runtimes.append(runtimes_inst)

        obj_avg = np.average(obj_values)
        obj_std = np.std(obj_values)
        runtime_avg = np.average(runtimes)
        runtime_std = np.std(runtimes)
        print("---------------------- Values over all instances ----------------------")
        print("Mean Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
        print("Mean Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
        print("---------------------- ---------------------- ----------------------")
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
            obj_values_inst = []
            runtimes_inst = []
            for rep in range(repetitions):
                mwccp_instance = read_instance(path)
                mwccp_solution = MWCCPSolution(mwccp_instance)
                _, obj, stats = mwccp_solution.deterministic_construction_heuristic()
                obj_values_inst.append(obj)
                runtimes_inst.append(stats.get_run_time())

            obj_avg = np.mean(obj_values_inst)
            obj_std = np.std(obj_values_inst)
            runtime_avg = np.mean(runtimes_inst)
            runtime_std = np.std(runtimes_inst)
            print("#################### " + str(test) + " ####################")
            print("Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
            print("Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
            print("#################### #################### ####################")
            obj_values.append(obj_values_inst)
            runtimes.append(runtimes_inst)

        obj_avg = np.average(obj_values)
        obj_std = np.std(obj_values)
        runtime_avg = np.average(runtimes)
        runtime_std = np.std(runtimes)
        print("---------------------- Values over all instances ----------------------")
        print("Mean Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
        print("Mean Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
        print("---------------------- ---------------------- ----------------------")
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
            obj_values_inst = []
            runtimes_inst = []
            for rep in range(repetitions):
                mwccp_instance = read_instance(path)
                mwccp_solution = MWCCPSolution(mwccp_instance)
                _, obj, stats = mwccp_solution.randomized_construction_heuristic()
                obj_values_inst.append(obj)
                runtimes_inst.append(stats.get_run_time())

            obj_avg = np.mean(obj_values_inst)
            obj_std = np.std(obj_values_inst)
            runtime_avg = np.mean(runtimes_inst)
            runtime_std = np.std(runtimes_inst)
            print("#################### " + str(test) + " ####################")
            print("Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
            print("Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
            print("#################### #################### ####################")
            obj_values.append(obj_values_inst)
            runtimes.append(runtimes_inst)

        obj_avg = np.average(obj_values)
        obj_std = np.std(obj_values)
        runtime_avg = np.average(runtimes)
        runtime_std = np.std(runtimes)
        print("---------------------- Values over all instances ----------------------")
        print("Mean Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
        print("Mean Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
        print("---------------------- ---------------------- ----------------------")
        print("============== === ==============")
        # === === === === === ===

        # === GRASP ===
        print("============== GRASP ==============")
        neighborhood = MWCCPNeighborhoods.flip_two_adjacent_vertices
        step_function = StepFunction.best_improvement
        max_iterations = 30
        max_iter_local_search = 50
        repetitions = 10

        print("Repeating each instance " + str(repetitions) + " times and having a maximum number of " + str(
            max_iterations) + " iterations.")

        stats_list = []
        runtimes = []
        obj_values = []
        for test in os.listdir(directory):
            if test.endswith(".pkl"):
                continue
            print(test)
            path = directory + test
            obj_values_inst = []
            runtimes_inst = []
            for rep in range(repetitions):
                mwccp_instance = read_instance(path)
                mwccp_solution = MWCCPSolution(mwccp_instance)
                _, stats = mwccp_solution.grasp(neighborhood, step_function, max_iterations=max_iterations,
                                                max_iter_local_search=max_iter_local_search)
                obj_values_inst.append(stats.get_final_objective())
                runtimes_inst.append(stats.get_run_time())
                stats_list.append(stats)

            obj_avg = np.mean(obj_values_inst)
            obj_std = np.std(obj_values_inst)
            runtime_avg = np.mean(runtimes_inst)
            runtime_std = np.std(runtimes_inst)
            print("#################### " + str(test) + " ####################")
            print("Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
            print("Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
            print("#################### #################### ####################")
            obj_values.append(obj_values_inst)
            runtimes.append(runtimes_inst)

        obj_avg = np.average(obj_values)
        obj_std = np.std(obj_values)
        runtime_avg = np.average(runtimes)
        runtime_std = np.std(runtimes)
        print("---------------------- Values over all instances ----------------------")
        print("Mean Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
        print("Mean Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
        print("---------------------- ---------------------- ----------------------")
        multi_stats = MultiStats(stats_list)
        multi_stats.plot_avg_obj_over_time("GRASP")
        print("============== === ==============")
        # === === === === === ===

    def test_deterministic_and_randomized_ch_and_GRASP_medium_large(self):
        directory = "../data/test_instances/medium_large/"

        # === Deterministic CH ===
        print("============== DCH ==============")
        repetitions = 1
        runtimes = []
        obj_values = []
        for test in os.listdir(directory):
            if test.endswith(".pkl"):
                continue
            path = directory + test
            obj_values_inst = []
            runtimes_inst = []
            for rep in range(repetitions):
                mwccp_instance = read_instance(path)
                mwccp_solution = MWCCPSolution(mwccp_instance)
                _, obj, stats = mwccp_solution.deterministic_construction_heuristic()
                obj_values_inst.append(obj)
                runtimes_inst.append(stats.get_run_time())

            obj_avg = np.mean(obj_values_inst)
            obj_std = np.std(obj_values_inst)
            runtime_avg = np.mean(runtimes_inst)
            runtime_std = np.std(runtimes_inst)
            print("#################### " + str(test) + " ####################")
            print("Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
            print("Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
            print("#################### #################### ####################")
            obj_values.append(obj_values_inst)
            runtimes.append(runtimes_inst)

        obj_avg = np.average(obj_values)
        obj_std = np.std(obj_values)
        runtime_avg = np.average(runtimes)
        runtime_std = np.std(runtimes)
        print("---------------------- Values over all instances ----------------------")
        print("Mean Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
        print("Mean Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
        print("---------------------- ---------------------- ----------------------")
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
            obj_values_inst = []
            runtimes_inst = []
            for rep in range(repetitions):
                mwccp_instance = read_instance(path)
                mwccp_solution = MWCCPSolution(mwccp_instance)
                _, obj, stats = mwccp_solution.randomized_construction_heuristic()
                obj_values_inst.append(obj)
                runtimes_inst.append(stats.get_run_time())

            obj_avg = np.mean(obj_values_inst)
            obj_std = np.std(obj_values_inst)
            runtime_avg = np.mean(runtimes_inst)
            runtime_std = np.std(runtimes_inst)
            print("#################### " + str(test) + " ####################")
            print("Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
            print("Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
            print("#################### #################### ####################")
            obj_values.append(obj_values_inst)
            runtimes.append(runtimes_inst)

        obj_avg = np.average(obj_values)
        obj_std = np.std(obj_values)
        runtime_avg = np.average(runtimes)
        runtime_std = np.std(runtimes)
        print("---------------------- Values over all instances ----------------------")
        print("Mean Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
        print("Mean Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
        print("---------------------- ---------------------- ----------------------")
        print("============== === ==============")
        # === === === === === ===

        # === GRASP ===
        print("============== GRASP ==============")
        neighborhood = MWCCPNeighborhoods.flip_two_adjacent_vertices
        step_function = StepFunction.best_improvement
        max_iterations = 20
        max_iter_local_search = 30
        repetitions = 5

        print("Repeating each instance " + str(repetitions) + " times and having a maximum number of " + str(
            max_iterations) + " iterations.")

        stats_list = []
        runtimes = []
        obj_values = []
        for test in os.listdir(directory):
            if test.endswith(".pkl"):
                continue
            print(test)
            path = directory + test
            obj_values_inst = []
            runtimes_inst = []
            for rep in range(repetitions):
                mwccp_instance = read_instance(path)
                mwccp_solution = MWCCPSolution(mwccp_instance)
                _, stats = mwccp_solution.grasp(neighborhood, step_function, max_iterations=max_iterations,
                                                max_iter_local_search=max_iter_local_search)
                obj_values_inst.append(stats.get_final_objective())
                runtimes_inst.append(stats.get_run_time())
                stats_list.append(stats)

            obj_avg = np.mean(obj_values_inst)
            obj_std = np.std(obj_values_inst)
            runtime_avg = np.mean(runtimes_inst)
            runtime_std = np.std(runtimes_inst)
            print("#################### " + str(test) + " ####################")
            print("Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
            print("Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
            print("#################### #################### ####################")
            obj_values.append(obj_values_inst)
            runtimes.append(runtimes_inst)

        obj_avg = np.average(obj_values)
        obj_std = np.std(obj_values)
        runtime_avg = np.average(runtimes)
        runtime_std = np.std(runtimes)
        print("---------------------- Values over all instances ----------------------")
        print("Mean Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
        print("Mean Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
        print("---------------------- ---------------------- ----------------------")
        multi_stats = MultiStats(stats_list)
        multi_stats.plot_avg_obj_over_time("GRASP")
        print("============== === ==============")
        # === === === === === ===

    def test_deterministic_and_randomized_ch_and_GRASP_large(self):
        directory = "../data/test_instances/large/"

        # === Deterministic CH ===
        print("============== DCH ==============")
        repetitions = 1
        runtimes = []
        obj_values = []
        for test in os.listdir(directory):
            if test.endswith(".pkl"):
                continue
            path = directory + test
            obj_values_inst = []
            runtimes_inst = []
            for rep in range(repetitions):
                mwccp_instance = read_instance(path)
                mwccp_solution = MWCCPSolution(mwccp_instance)
                _, obj, stats = mwccp_solution.deterministic_construction_heuristic()
                obj_values_inst.append(obj)
                runtimes_inst.append(stats.get_run_time())

            obj_avg = np.mean(obj_values_inst)
            obj_std = np.std(obj_values_inst)
            runtime_avg = np.mean(runtimes_inst)
            runtime_std = np.std(runtimes_inst)
            print("#################### " + str(test) + " ####################")
            print("Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
            print("Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
            print("#################### #################### ####################")
            obj_values.append(obj_values_inst)
            runtimes.append(runtimes_inst)

        obj_avg = np.average(obj_values)
        obj_std = np.std(obj_values)
        runtime_avg = np.average(runtimes)
        runtime_std = np.std(runtimes)
        print("---------------------- Values over all instances ----------------------")
        print("Mean Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
        print("Mean Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
        print("---------------------- ---------------------- ----------------------")
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
            obj_values_inst = []
            runtimes_inst = []
            for rep in range(repetitions):
                mwccp_instance = read_instance(path)
                mwccp_solution = MWCCPSolution(mwccp_instance)
                _, obj, stats = mwccp_solution.randomized_construction_heuristic()
                obj_values_inst.append(obj)
                runtimes_inst.append(stats.get_run_time())

            obj_avg = np.mean(obj_values_inst)
            obj_std = np.std(obj_values_inst)
            runtime_avg = np.mean(runtimes_inst)
            runtime_std = np.std(runtimes_inst)
            print("#################### " + str(test) + " ####################")
            print("Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
            print("Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
            print("#################### #################### ####################")
            obj_values.append(obj_values_inst)
            runtimes.append(runtimes_inst)

        obj_avg = np.average(obj_values)
        obj_std = np.std(obj_values)
        runtime_avg = np.average(runtimes)
        runtime_std = np.std(runtimes)
        print("---------------------- Values over all instances ----------------------")
        print("Mean Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
        print("Mean Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
        print("---------------------- ---------------------- ----------------------")
        print("============== === ==============")
        # === === === === === ===

        # === GRASP ===
        print("============== GRASP ==============")
        neighborhood = MWCCPNeighborhoods.flip_two_adjacent_vertices
        step_function = StepFunction.best_improvement
        max_iterations = 10
        max_iter_local_search = 10
        repetitions = 5

        print("Repeating each instance " + str(repetitions) + " times and having a maximum number of " + str(
            max_iterations) + " iterations.")

        stats_list = []
        runtimes = []
        obj_values = []
        for test in os.listdir(directory):
            if test.endswith(".pkl"):
                continue
            print(test)
            path = directory + test
            obj_values_inst = []
            runtimes_inst = []
            for rep in range(repetitions):
                mwccp_instance = read_instance(path)
                mwccp_solution = MWCCPSolution(mwccp_instance)
                _, stats = mwccp_solution.grasp(neighborhood, step_function, max_iterations=max_iterations,
                                                max_iter_local_search=max_iter_local_search)
                obj_values_inst.append(stats.get_final_objective())
                runtimes_inst.append(stats.get_run_time())
                stats_list.append(stats)

            obj_avg = np.mean(obj_values_inst)
            obj_std = np.std(obj_values_inst)
            runtime_avg = np.mean(runtimes_inst)
            runtime_std = np.std(runtimes_inst)
            print("#################### " + str(test) + " ####################")
            print("Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
            print("Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
            print("#################### #################### ####################")
            obj_values.append(obj_values_inst)
            runtimes.append(runtimes_inst)

        obj_avg = np.average(obj_values)
        obj_std = np.std(obj_values)
        runtime_avg = np.average(runtimes)
        runtime_std = np.std(runtimes)
        print("---------------------- Values over all instances ----------------------")
        print("Mean Average objective value: " + f"{obj_avg:.1f}, with std: " + f"{obj_std:.4f}")
        print("Mean Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
        print("---------------------- ---------------------- ----------------------")
        multi_stats = MultiStats(stats_list)
        multi_stats.plot_avg_obj_over_time("GRASP")
        print("============== === ==============")
        # === === === === === ===
