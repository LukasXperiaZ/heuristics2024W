import os
import time
import unittest

from src.MWCCP import MWCCPSolution
from src.read_instance import read_instance


def run_test(paths: [], max_time_one_run: int):
    print("Start reading the instances ...")
    start = time.time()
    mwccp_solutions: [MWCCPSolution] = []
    for path in paths:
        if path.endswith(".pkl"):
            continue
        mwccp_instance = read_instance(path)
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solutions.append(mwccp_solution)
    end = time.time()
    print("Finish reading the instances in {:.4f}s!".format(end - start))

    print("Starting to run the algorithms ...")
    print("Progress: 0%")
    n_solutions = len(mwccp_solutions)
    current = 1
    for mwccp_solution in mwccp_solutions:
        _, stats = mwccp_solution.genetic_algorithm(
            population_size=46,
            randomized_const_heuristic_initialization="random_and_repair",
            elitist_population=0.15918,
            bot_population=0.11336,
            k=46,
            crossover_range=7,
            mutation_prob=0.04676,
            repair_percentage=0.87525,
            penalize_factor=1.99102,
            max_time_in_s=max_time_one_run,
        )
        stats.show_plot(os.path.basename(mwccp_solution.inst.path))
        print("Progress: {}%".format(int(100 * current / n_solutions)))
        current = current + 1


class AdditionalTests(unittest.TestCase):

    def test_additional_test_small(self):
        paths = ["../data/test_instances/small/inst_50_4_00001", "../data/test_instances/small/inst_50_4_00005",
                 "../data/test_instances/small/inst_50_4_00010"]
        max_time_one_run = 1
        run_test(paths, max_time_one_run)

    def test_additional_test_medium(self):
        paths = ["../data/test_instances/medium/inst_200_20_00001",
                 "../data/test_instances/medium/inst_200_20_00005",
                 "../data/test_instances/medium/inst_200_20_00010"]
        max_time_one_run = 60*3
        run_test(paths, max_time_one_run)

    def test_additional_test_medium_large(self):
        paths = ["../data/test_instances/medium_large/inst_500_40_00001",
                 "../data/test_instances/medium_large/inst_500_40_00010",
                 "../data/test_instances/medium_large/inst_500_40_00019"]
        max_time_one_run = 60*3
        run_test(paths, max_time_one_run)

    def test_additional_test_large(self):
        paths = ["../data/test_instances/large/inst_1000_60_00001",
                 "../data/test_instances/large/inst_1000_60_00005",
                 "../data/test_instances/large/inst_1000_60_00010"]
        max_time_one_run = 60*10
        run_test(paths, max_time_one_run)
