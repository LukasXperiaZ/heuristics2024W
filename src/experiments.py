import os
import unittest

import numpy as np

from src.MWCCP import MWCCPSolution, MWCCPNeighborhoods
from src.evaluation import MultiStats
from src.local_search import StepFunction
from src.read_instance import read_instance


class DCH_RCH_GRASP(unittest.TestCase):

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


class LocalSearch(unittest.TestCase):
    def test_local_search_small(self):
        directory = "../data/test_instances/small/"
        self.local_search_experiment(directory, False)

    def test_local_search_medium(self):
        directory = "../data/test_instances/medium/"
        self.local_search_experiment(directory, False)

    def test_local_search_medium_large(self):
        directory = "../data/test_instances/medium_large/"
        self.local_search_experiment(directory, False)

    def test_local_search_large(self):
        directory = "../data/test_instances/large/"
        self.local_search_experiment(directory, False)

    def local_search_experiment(self, directory: str, only_plot: bool):
        stats_list = [None, None, None]

        print("============== Best Improvement ==============")
        step_function = StepFunction.best_improvement
        max_time = 5

        instances: [(MWCCPSolution, [], int)] = []
        for test in os.listdir(directory):
            if test.endswith(".pkl"):
                continue
            path = directory + test
            mwccp_instance = read_instance(path)
            mwccp_solution = MWCCPSolution(mwccp_instance)
            initial_solution, obj, _ = mwccp_solution.deterministic_construction_heuristic()
            instances.append((mwccp_solution, initial_solution, obj))

        for neighborhood in MWCCPNeighborhoods:
            runtimes_config = []
            obj_values_config = []
            iterations_config = []
            for (mwccp_solution, initial_solution, obj) in instances:
                if only_plot:
                    if stats_list[0] is not None:
                        break
                sol, obj, stats = mwccp_solution.local_search(initial_solution, neighborhood, step_function,
                                                              initial_obj=obj, max_time_in_s=max_time)
                runtimes_config.append(stats.get_run_time())
                obj_values_config.append(obj)
                iterations_config.append(stats.get_iterations())
                if stats_list[0] is None:
                    stats_list[0] = stats

            obj_avg = np.mean(obj_values_config)
            runtime_avg = np.mean(runtimes_config)
            runtime_std = np.std(runtimes_config)
            iterations_avg = np.mean(iterations_config)
            iterations_std = np.std(iterations_config)
            print("#################### " + str(neighborhood) + " ####################")
            print("Average objective value: " + f"{obj_avg:.1f}")
            print("Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
            print("Average iterations: " + f"{iterations_avg:.1f}, with std: " + f"{iterations_std:.4f}s")
            print("#################### #################### ####################")
        print("============== === ==============")

        print("============== First Improvement ==============")
        step_function = StepFunction.first_improvement
        max_time = 5

        instances: [(MWCCPSolution, [], int)] = []
        for test in os.listdir(directory):
            if test.endswith(".pkl"):
                continue
            path = directory + test
            mwccp_instance = read_instance(path)
            mwccp_solution = MWCCPSolution(mwccp_instance)
            initial_solution, obj, _ = mwccp_solution.deterministic_construction_heuristic()
            instances.append((mwccp_solution, initial_solution, obj))

        for neighborhood in MWCCPNeighborhoods:
            runtimes_config = []
            obj_values_config = []
            iterations_config = []
            for (mwccp_solution, initial_solution, obj) in instances:
                if only_plot:
                    if stats_list[1] is not None:
                        break
                sol, obj, stats = mwccp_solution.local_search(initial_solution, neighborhood, step_function,
                                                              initial_obj=obj, max_time_in_s=max_time)
                runtimes_config.append(stats.get_run_time())
                obj_values_config.append(obj)
                iterations_config.append(stats.get_iterations())
                if stats_list[1] is None:
                    stats_list[1] = stats

            obj_avg = np.mean(obj_values_config)
            runtime_avg = np.mean(runtimes_config)
            runtime_std = np.std(runtimes_config)
            iterations_avg = np.mean(iterations_config)
            iterations_std = np.std(iterations_config)
            print("#################### " + str(neighborhood) + " ####################")
            print("Average objective value: " + f"{obj_avg:.1f}")
            print("Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
            print("Average iterations: " + f"{iterations_avg:.1f}, with std: " + f"{iterations_std:.4f}s")
            print("#################### #################### ####################")
        print("============== === ==============")

        print("============== Random ==============")
        step_function = StepFunction.random
        max_time = 1
        repetitions = 10

        instances: [(MWCCPSolution, [], int)] = []
        for test in os.listdir(directory):
            if test.endswith(".pkl"):
                continue
            path = directory + test
            mwccp_instance = read_instance(path)
            mwccp_solution = MWCCPSolution(mwccp_instance)
            initial_solution, obj, _ = mwccp_solution.deterministic_construction_heuristic()
            instances.append((mwccp_solution, initial_solution, obj))

        for neighborhood in MWCCPNeighborhoods:
            runtimes_config = []
            obj_values_config = []
            iterations_config = []
            for i in range(repetitions):
                for (mwccp_solution, initial_solution, obj) in instances:
                    if only_plot:
                        if stats_list[2] is not None:
                            break
                    sol, obj, stats = mwccp_solution.local_search(initial_solution, neighborhood, step_function,
                                                                  initial_obj=obj, max_time_in_s=max_time)
                    runtimes_config.append(stats.get_run_time())
                    obj_values_config.append(obj)
                    iterations_config.append(stats.get_iterations())
                    if stats_list[2] is None:
                        stats_list[2] = stats
                i += 1

            obj_avg = np.mean(obj_values_config)
            runtime_avg = np.mean(runtimes_config)
            runtime_std = np.std(runtimes_config)
            iterations_avg = np.mean(iterations_config)
            iterations_std = np.std(iterations_config)
            print("#################### " + str(neighborhood) + " ####################")
            print("Average objective value: " + f"{obj_avg:.1f}")
            print("Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
            print("Average iterations: " + f"{iterations_avg:.1f}, with std: " + f"{iterations_std:.4f}s")
            print("#################### #################### ####################")

        multi_stats = MultiStats(stats_list)
        multi_stats.plot_stats("first instance")
        print("============== === ==============")


class VND(unittest.TestCase):
    def test_VND_small(self):
        directory = "../data/test_instances/small/"
        max_runtime = 60  # 1 minute max
        self.VND_experiment(directory, max_runtime, False)

    def test_VND_medium(self):
        directory = "../data/test_instances/medium/"
        max_runtime = 60
        self.VND_experiment(directory, max_runtime, False)

    def test_VND_medium_large(self):
        directory = "../data/test_instances/medium_large/"
        max_runtime = 60
        self.VND_experiment(directory, max_runtime, False)

    def test_VND_large(self):
        directory = "../data/test_instances/large/"
        max_runtime = 60
        self.VND_experiment(directory, max_runtime, False)


    def VND_experiment(self, directory: str, max_runtime: int, only_plot: bool
                       ):
        neighborhoods = [MWCCPNeighborhoods.flip_two_adjacent_vertices, MWCCPNeighborhoods.flip_three_adjacent_vertices,
                         MWCCPNeighborhoods.flip_four_adjacent_vertices]
        step_function = StepFunction.first_improvement

        stats_to_plot = None
        test_name = ""

        obj_values = []
        runtimes = []
        iterations = []

        for test in os.listdir(directory):
            if test.endswith(".pkl"):
                continue
            if only_plot:
                if stats_to_plot is not None:
                    break
            path = directory + test
            mwccp_instance = read_instance(path)
            mwccp_solution = MWCCPSolution(mwccp_instance)
            solution, stats = mwccp_solution.vnd(neighborhoods, step_function, max_time_in_s=max_runtime)
            stats.print_stats(test)
            obj_values.append(stats.get_final_objective())
            runtimes.append(stats.get_run_time())
            iterations.append(stats.get_iterations())

            if stats_to_plot is None:
                stats_to_plot = stats
                test_name = test

        obj_avg = np.mean(obj_values)
        runtime_avg = np.mean(runtimes)
        runtime_std = np.std(runtimes)
        iterations_avg = np.mean(iterations)
        iterations_std = np.std(iterations)
        print("#################### Averages ####################")
        print("Average objective value: " + f"{obj_avg:.1f}")
        print("Average time : " + f"{runtime_avg:.6f}s, with std: " + f"{runtime_std:.4f}s")
        print("Average iterations: " + f"{iterations_avg:.1f}, with std: " + f"{iterations_std:.4f}s")
        print("#################### #################### ####################")


        stats_to_plot.show_plot(test_name)