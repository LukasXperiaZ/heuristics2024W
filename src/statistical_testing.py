import csv
import os
import time
import unittest

import numpy as np

from src.MWCCP import MWCCPSolution, MWCCPNeighborhoods
from src.local_search import StepFunction
from src.read_instance import read_instance


class StatisticalTesting(unittest.TestCase):
    def test_generate_statistical_data_small(self):
        directory = "../data/test_instances/small/"
        repetitions = 10
        max_time_one_run = 5

        print("Start reading the instances ...")
        start = time.time()
        mwccp_solutions: [MWCCPSolution] = []
        for test in os.listdir(directory):
            if test.endswith(".pkl"):
                continue
            path = directory + test
            mwccp_instance = read_instance(path)
            mwccp_solution = MWCCPSolution(mwccp_instance)
            mwccp_solutions.append(mwccp_solution)
        end = time.time()
        print("Finish reading the instances in {:.4f}s!".format(end - start))

        print("Start running the algorithms ...")
        print("Progress: 0%")
        start = time.time()
        obj_values: [[]] = []
        n_solutions = len(mwccp_solutions)
        current = 1
        for mwccp_solution in mwccp_solutions:
            obj_values_inst_genetic = []
            obj_values_inst_hybrid = []
            for rep in range(repetitions):
                _, stats = mwccp_solution.genetic_algorithm(
                    population_size=117,
                    randomized_const_heuristic_initialization="random_and_repair",
                    elitist_population=0.21950,
                    bot_population=0.16065,
                    k=39,
                    crossover_range=12,
                    mutation_prob=0.05040,
                    repair_percentage=0.28665,
                    penalize_factor=1.45558,
                    max_time_in_s=max_time_one_run,
                )
                obj_values_inst_genetic.append(stats.get_final_objective())

                _, stats = mwccp_solution.genetic_algorithm_with_vnd(
                    population_size=147,
                    randomized_const_heuristic_initialization="random_and_repair",
                    elitist_population=0.13477,
                    bot_population=0.20927,
                    k=14,
                    crossover_range=7,
                    mutation_prob=0.01941,
                    repair_percentage=0.73638,
                    penalize_factor=1.69300,
                    vnd_percentage=0.21028,
                    vnd_max_runtime_in_s=0.561,
                    vnd_neighborhoods=[MWCCPNeighborhoods.flip_two_adjacent_vertices,
                                       MWCCPNeighborhoods.flip_three_adjacent_vertices,
                                       MWCCPNeighborhoods.flip_four_adjacent_vertices],
                    step_function=StepFunction.first_improvement,
                    vnd_randomized_const_heuristic="random_and_repair",
                    max_time_in_s=max_time_one_run,
                )
                obj_values_inst_hybrid.append(stats.get_final_objective())

            obj_values_inst_genetic_mean = np.mean(obj_values_inst_genetic)
            obj_values_inst_hybrid_mean = np.mean(obj_values_inst_hybrid)
            obj_values.append([obj_values_inst_genetic_mean, obj_values_inst_hybrid_mean])

            print("Progress: {}%".format(int(100 * current / n_solutions)))
            current = current + 1

        end = time.time()
        print("Finish running the algorithms in {:.4f}s!".format(end - start))

        print("Start saving the statistical data ...")
        # Save objective values
        filename = "../data/statistical_testing/data_small.csv"
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerow(["obj_A", "obj_B"])
            writer.writerows(obj_values)
        print("Finish saving the statistical data!")

    def test_generate_statistical_data_medium(self):
        directory = "../data/test_instances/medium/"
        repetitions = 5
        max_time_one_run = 10

        print("Start reading the instances ...")
        start = time.time()
        mwccp_solutions: [MWCCPSolution] = []
        for test in os.listdir(directory):
            if test.endswith(".pkl"):
                continue
            path = directory + test
            mwccp_instance = read_instance(path)
            mwccp_solution = MWCCPSolution(mwccp_instance)
            mwccp_solutions.append(mwccp_solution)
        end = time.time()
        print("Finish reading the instances in {:.4f}s!".format(end - start))

        print("Start running the algorithms ...")
        print("Progress: 0%")
        start = time.time()
        obj_values: [[]] = []
        n_solutions = len(mwccp_solutions)
        current = 1
        for mwccp_solution in mwccp_solutions:
            obj_values_inst_genetic = []
            obj_values_inst_hybrid = []
            for rep in range(repetitions):
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
                obj_values_inst_genetic.append(stats.get_final_objective())

                _, stats = mwccp_solution.genetic_algorithm_with_vnd(
                    population_size=51,
                    randomized_const_heuristic_initialization="random_and_repair",
                    elitist_population=0.12610,
                    bot_population=0.14248,
                    k=5,
                    crossover_range=92,
                    mutation_prob=0.10787,
                    repair_percentage=0.44487,
                    penalize_factor=1.73904,
                    vnd_percentage=0.77032,
                    vnd_max_runtime_in_s=0.480,
                    vnd_neighborhoods=[MWCCPNeighborhoods.flip_three_adjacent_vertices],
                    step_function=StepFunction.random,
                    vnd_randomized_const_heuristic="random_and_repair",
                    max_time_in_s=max_time_one_run,
                )
                obj_values_inst_hybrid.append(stats.get_final_objective())

            obj_values_inst_genetic_mean = np.mean(obj_values_inst_genetic)
            obj_values_inst_hybrid_mean = np.mean(obj_values_inst_hybrid)
            obj_values.append([obj_values_inst_genetic_mean, obj_values_inst_hybrid_mean])

            print("Progress: {}%".format(int(100 * current / n_solutions)))
            current = current + 1

        end = time.time()
        print("Finish running the algorithms in {:.4f}s!".format(end - start))

        print("Start saving the statistical data ...")
        # Save objective values
        filename = "../data/statistical_testing/data_medium.csv"
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerow(["obj_A", "obj_B"])
            writer.writerows(obj_values)
        print("Finish saving the statistical data!")
