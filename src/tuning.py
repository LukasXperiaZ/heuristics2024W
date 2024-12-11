import os
import random
import unittest
import warnings

import numpy as np
from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UniformFloatHyperparameter
from smac import Scenario, HyperparameterOptimizationFacade

from src.MWCCP import MWCCPSolution, MWCCPNeighborhoods
from src.local_search import StepFunction
from src.read_instance import read_instance


def evaluate_genetic_algorithm_small(params, seed):
    random.seed(seed)
    directory = "../data/tuning/tuning/small/"
    values_to_normalize = [80230.0, 21215.6, 11434.4, 6112.0, 4458.8, 2934.8, 2416.2, 1904.0, 1113.0, 719.2]
    return evaluate_genetic_algorithm(params, directory, values_to_normalize)


def evaluate_genetic_algorithm_medium(params, seed):
    random.seed(seed)
    directory = "../data/tuning/tuning/medium/"
    values_to_normalize = [22498113.2, 7972404.6, 3996357.8, 2255912.6, 1537340.8, 1095402.2, 818517.4, 596553.4,
                           466588.6, 371635.8]
    return evaluate_genetic_algorithm(params, directory, values_to_normalize)


def evaluate_genetic_algorithm_large(params, seed):
    random.seed(seed)
    directory = "../data/tuning/tuning/large/"
    values_to_normalize = [14727208525.6, 5317437510.2, 2689732309.0, 1616581844.4, 1061879453.6, 750038288.2,
                           568421573.2, 442069617.0, 354454608.0, 288258956.0]
    return evaluate_genetic_algorithm(params, directory, values_to_normalize)


def evaluate_genetic_algorithm(params, directory, values_to_normalize: [float]):
    repetitions = 5

    mwccp_solutions: [MWCCPSolution] = []
    for test in os.listdir(directory):
        if test.endswith(".pkl"):
            continue
        path = directory + test
        mwccp_instance = read_instance(path)
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solutions.append(mwccp_solution)

    obj_values = []
    for mwccp_solution in mwccp_solutions:
        obj_values_inst = []
        for rep in range(repetitions):
            _, stats = mwccp_solution.genetic_algorithm(
                population_size=params["population_size"],
                randomized_const_heuristic_initialization=params["randomized_const_heuristic_initialization"],
                elitist_population=params["elitist_population"],
                bot_population=params["bot_population"],
                k=params["k"],
                crossover_range=params["crossover_range"],
                mutation_prob=params["mutation_prob"],
                repair_percentage=params["repair_percentage"],
                penalize_factor=params["penalize_factor"],
                max_time_in_s=1,
            )
            obj_values_inst.append(stats.get_final_objective())

        mean_val_inst = np.mean(obj_values_inst)
        obj_values.append(mean_val_inst)

    # print("Objective values:")
    # print(obj_values)
    obj_values = np.array(obj_values)
    normalized_obj_values = (obj_values / np.array(values_to_normalize))
    # print("Normalized Objective values:")
    # print(normalized_obj_values)
    mean_normalized_value = np.mean(normalized_obj_values)

    return mean_normalized_value


def evaluate_hybrid_algorithm_small(params, seed):
    random.seed(seed)
    directory = "../data/tuning/tuning/small/"
    values_to_normalize = [80230.0, 21213.0, 11432.0, 6112.0, 4445.4, 2933.0, 2417.8, 1904.2, 1113.0, 719.0]
    return evaluate_hybrid_algorithm(params, directory, values_to_normalize)


def evaluate_hybrid_algorithm_medium(params, seed):
    random.seed(seed)
    directory = "../data/tuning/tuning/medium/"
    values_to_normalize = [22838464.2, 8048093.0, 4126629.6, 2345240.2, 1605810.6, 1134458.6, 862697.2, 631232.4,
                           496779.8, 405854.8]
    return evaluate_hybrid_algorithm(params, directory, values_to_normalize)


def evaluate_hybrid_algorithm_large(params, seed):
    random.seed(seed)
    directory = "../data/tuning/tuning/large/"
    values_to_normalize = [14727208525.6, 5317437510.2, 2689732309.0, 1616581844.4, 1061879453.6, 750038288.2,
                           568421573.2, 442069617.0, 354454608.0, 288258956.0]
    return evaluate_hybrid_algorithm(params, directory, values_to_normalize)


def evaluate_hybrid_algorithm(params, directory, values_to_normalize: [float]):
    repetitions = 5

    if params["step_function"] == "first":
        step_function = StepFunction.first_improvement
    elif params["step_function"] == "best":
        step_function = StepFunction.best_improvement
    else:
        step_function = StepFunction.random

    neighborhoods_numbers = params["vnd_neighborhoods"]
    neighborhood = []
    for neighbor in neighborhoods_numbers:
        if neighbor == 2:
            neighborhood.append(MWCCPNeighborhoods.flip_two_adjacent_vertices)
        elif neighbor == 3:
            neighborhood.append(MWCCPNeighborhoods.flip_three_adjacent_vertices)
        else:
            neighborhood.append(MWCCPNeighborhoods.flip_four_adjacent_vertices)

    mwccp_solutions: [MWCCPSolution] = []
    for test in os.listdir(directory):
        if test.endswith(".pkl"):
            continue
        path = directory + test
        mwccp_instance = read_instance(path)
        mwccp_solution = MWCCPSolution(mwccp_instance)
        mwccp_solutions.append(mwccp_solution)

    obj_values = []
    for mwccp_solution in mwccp_solutions:
        obj_values_inst = []
        for rep in range(repetitions):
            _, stats = mwccp_solution.genetic_algorithm_with_vnd(
                population_size=params["population_size"],
                randomized_const_heuristic_initialization="random_and_repair",
                elitist_population=params["elitist_population"],
                bot_population=params["bot_population"],
                k=params["k"],
                crossover_range=params["crossover_range"],
                mutation_prob=params["mutation_prob"],
                repair_percentage=params["repair_percentage"],
                penalize_factor=params["penalize_factor"],
                vnd_percentage=params["vnd_percentage"],
                vnd_max_runtime_in_s=params["vnd_max_runtime_in_s"],
                vnd_neighborhoods=neighborhood,
                step_function=step_function,
                vnd_randomized_const_heuristic="random_and_repair",
                max_time_in_s=1,
            )
            obj_values_inst.append(stats.get_final_objective())

        mean_val_inst = np.mean(obj_values_inst)
        obj_values.append(mean_val_inst)

    #print("Objective values:")
    #print(obj_values)
    obj_values = np.array(obj_values)
    normalized_obj_values = (obj_values / np.array(values_to_normalize))
    #print("Normalized Objective values:")
    #print(normalized_obj_values)
    mean_normalized_value = np.mean(normalized_obj_values)

    return mean_normalized_value


class Tuning(unittest.TestCase):
    def test_evaluate_genetic_algorithm_small(self):
        print("Starting ...")
        evaluate_genetic_algorithm_small({
            "population_size": 100,
            "randomized_const_heuristic_initialization": "random_and_repair",
            "elitist_population": 0.2,
            "bot_population": 0.2,
            "k": 10,
            "crossover_range": 5,
            "mutation_prob": 0.05,
            "repair_percentage": 0.5,
            "penalize_factor": 1.5,
        }, 0)

    def test_evaluate_genetic_algorithm_medium(self):
        print("Starting ...")
        evaluate_genetic_algorithm_medium({
            "population_size": 100,
            "randomized_const_heuristic_initialization": "random_and_repair",
            "elitist_population": 0.2,
            "bot_population": 0.2,
            "k": 10,
            "crossover_range": 5,
            "mutation_prob": 0.05,
            "repair_percentage": 0.5,
            "penalize_factor": 1.5,
        }, 0)

    def test_evaluate_genetic_algorithm_large(self):
        print("Starting ...")
        evaluate_genetic_algorithm_large({
            "population_size": 100,
            "randomized_const_heuristic_initialization": "random_and_repair",
            "elitist_population": 0.2,
            "bot_population": 0.2,
            "k": 10,
            "crossover_range": 5,
            "mutation_prob": 0.05,
            "repair_percentage": 0.5,
            "penalize_factor": 1.5,
        }, 0)

    def test_evaluate_hybrid_algorithm_small(self):
        print("Starting ...")
        evaluate_hybrid_algorithm_small({
            "population_size": 100,
            "randomized_const_heuristic_initialization": "random_and_repair",
            "elitist_population": 0.2,
            "bot_population": 0.2,
            "k": 10,
            "crossover_range": 5,
            "mutation_prob": 0.05,
            "repair_percentage": 0.5,
            "penalize_factor": 1.5,
            "vnd_percentage": 0.5,
            "vnd_max_runtime_in_s": 0.05,
            "vnd_neighborhoods": [
                MWCCPNeighborhoods.flip_two_adjacent_vertices,
                MWCCPNeighborhoods.flip_three_adjacent_vertices,
                MWCCPNeighborhoods.flip_four_adjacent_vertices],
            "step_function": StepFunction.first_improvement,
            "vnd_randomized_const_heuristic": "random_and_repair",
        }, 0)

    def test_evaluate_hybrid_algorithm_medium(self):
        print("Starting ...")
        evaluate_hybrid_algorithm_medium({
            "population_size": 100,
            "randomized_const_heuristic_initialization": "random_and_repair",
            "elitist_population": 0.2,
            "bot_population": 0.2,
            "k": 10,
            "crossover_range": 5,
            "mutation_prob": 0.05,
            "repair_percentage": 0.5,
            "penalize_factor": 1.5,
            "vnd_percentage": 0.5,
            "vnd_max_runtime_in_s": 0.05,
            "vnd_neighborhoods": [
                MWCCPNeighborhoods.flip_two_adjacent_vertices,
                MWCCPNeighborhoods.flip_three_adjacent_vertices,
                MWCCPNeighborhoods.flip_four_adjacent_vertices],
            "step_function": StepFunction.first_improvement,
            "vnd_randomized_const_heuristic": "random_and_repair",
        }, 0)

    def test_evaluate_hybrid_algorithm_large(self):
        print("Starting ...")
        evaluate_hybrid_algorithm_large({
            "population_size": 100,
            "randomized_const_heuristic_initialization": "random_and_repair",
            "elitist_population": 0.2,
            "bot_population": 0.2,
            "k": 10,
            "crossover_range": 5,
            "mutation_prob": 0.05,
            "repair_percentage": 0.5,
            "penalize_factor": 1.5,
            "vnd_percentage": 0.5,
            "vnd_max_runtime_in_s": 0.05,
            "vnd_neighborhoods": [
                MWCCPNeighborhoods.flip_two_adjacent_vertices,
                MWCCPNeighborhoods.flip_three_adjacent_vertices,
                MWCCPNeighborhoods.flip_four_adjacent_vertices],
            "step_function": StepFunction.first_improvement,
            "vnd_randomized_const_heuristic": "random_and_repair",
        }, 0)

    # -------------------- Functions (test cases) used for tuning --------------------
    def test_tune_genetic_algorithm_small(self):
        self.run_genetic_algorithm_test("Genetic Algorithm Small", 25, 200)

    def test_tune_genetic_algorithm_medium(self):
        self.run_genetic_algorithm_test("Genetic Algorithm Medium", 100, 200)

    def test_tune_genetic_algorithm_large(self):
        self.run_genetic_algorithm_test("Genetic Algorithm Large", 500, 50, only_fast_RCH=True)

    def run_genetic_algorithm_test(self, scenario_name, solution_size, n_trials, only_fast_RCH: bool = False):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        cs = ConfigurationSpace()

        # Hyperparameters
        population_size = UniformIntegerHyperparameter("population_size", lower=50, upper=500, default_value=100)
        if not only_fast_RCH:
            randomized_const_heuristic_initialization = CategoricalHyperparameter(
                "randomized_const_heuristic_initialization", ["standard", "random_and_repair"],
                default_value="random_and_repair")
        else:
            randomized_const_heuristic_initialization = CategoricalHyperparameter(
                "randomized_const_heuristic_initialization", ["random_and_repair"],
                default_value="random_and_repair")
        elitist_population = UniformFloatHyperparameter("elitist_population", lower=0.1, upper=0.25, default_value=0.2)
        bot_population = UniformFloatHyperparameter("bot_population", lower=0.1, upper=0.3, default_value=0.2)
        k = UniformIntegerHyperparameter("k", lower=1, upper=50, default_value=10)
        crossover_range = UniformIntegerHyperparameter("crossover_range", lower=1, upper=solution_size - 1,
                                                       default_value=5)
        mutation_prob = UniformFloatHyperparameter("mutation_prob", lower=0.01, upper=0.3, default_value=0.05)
        repair_percentage = UniformFloatHyperparameter("repair_percentage", lower=0.0, upper=1.0, default_value=0.5)
        penalize_factor = UniformFloatHyperparameter("penalize_factor", lower=1.0, upper=2.0, default_value=1.5)

        cs.add(population_size, randomized_const_heuristic_initialization, elitist_population, bot_population, k,
               crossover_range, mutation_prob, repair_percentage, penalize_factor)

        scenario = Scenario(
            name=scenario_name,
            configspace=cs,
            deterministic=False,
            walltime_limit=60 * 15,  # 15 minutes
            n_trials=n_trials,
            n_workers=24,
            seed=0  # for reproducibility
        )

        print("Starting optimization...")
        if scenario_name == "Genetic Algorithm Small":
            smac = HyperparameterOptimizationFacade(scenario, evaluate_genetic_algorithm_small)
        elif scenario_name == "Genetic Algorithm Medium":
            smac = HyperparameterOptimizationFacade(scenario, evaluate_genetic_algorithm_medium)
        elif scenario_name == "Genetic Algorithm Large":
            smac = HyperparameterOptimizationFacade(scenario, evaluate_genetic_algorithm_large)
        else:
            raise ValueError("Unknown scenario name")

        incumbent = smac.optimize()
        print("Best found configuration:")
        print(incumbent)

    def test_tune_hybrid_algorithm_small(self):
        self.run_hybrid_algorithm_test("Hybrid Algorithm Small", 25, 200)

    def test_tune_hybrid_algorithm_medium(self):
        self.run_hybrid_algorithm_test("Hybrid Algorithm Medium", 100, 200)

    def test_tune_hybrid_algorithm_large(self):
        self.run_hybrid_algorithm_test("Hybrid Algorithm Large", 500, 50)

    def run_hybrid_algorithm_test(self, scenario_name, solution_size, n_trials):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        cs = ConfigurationSpace()

        # Hyperparameters
        population_size = UniformIntegerHyperparameter("population_size", lower=50, upper=500, default_value=100)
        elitist_population = UniformFloatHyperparameter("elitist_population", lower=0.1, upper=0.25, default_value=0.2)
        bot_population = UniformFloatHyperparameter("bot_population", lower=0.1, upper=0.3, default_value=0.2)
        k = UniformIntegerHyperparameter("k", lower=1, upper=50, default_value=10)
        crossover_range = UniformIntegerHyperparameter("crossover_range", lower=1, upper=solution_size - 1,
                                                       default_value=5)
        mutation_prob = UniformFloatHyperparameter("mutation_prob", lower=0.01, upper=0.3, default_value=0.05)
        repair_percentage = UniformFloatHyperparameter("repair_percentage", lower=0.0, upper=1.0, default_value=0.5)
        penalize_factor = UniformFloatHyperparameter("penalize_factor", lower=1.0, upper=2.0, default_value=1.5)
        vnd_percentage = UniformFloatHyperparameter("vnd_percentage", lower=0.01, upper=1.0, default_value=0.5)
        vnd_max_runtime_in_s = UniformFloatHyperparameter("vnd_max_runtime_in_s", lower=0.01, upper=1.0,
                                                          default_value=0.05)
        vnd_neighborhoods = CategoricalHyperparameter("vnd_neighborhoods", [
            [2],
            [3],
            [4],
            [2, 3],
            [3, 4],
            [2, 3, 4]
        ], default_value=[2, 3, 4])
        step_function = CategoricalHyperparameter("step_function",
                                                  ["first", "best",
                                                   "random"], default_value="first")

        cs.add(population_size, elitist_population, bot_population, k,
               crossover_range, mutation_prob, repair_percentage, penalize_factor, vnd_percentage, vnd_max_runtime_in_s,
               vnd_neighborhoods, step_function)

        scenario = Scenario(
            name=scenario_name,
            configspace=cs,
            deterministic=False,
            walltime_limit=60 * 15,  # 15 minutes
            n_trials=n_trials,
            n_workers=24,
            seed=0  # for reproducibility
        )

        print("Starting optimization...")
        if scenario_name == "Hybrid Algorithm Small":
            smac = HyperparameterOptimizationFacade(scenario, evaluate_hybrid_algorithm_small)
        elif scenario_name == "Hybrid Algorithm Medium":
            smac = HyperparameterOptimizationFacade(scenario, evaluate_hybrid_algorithm_medium)
        elif scenario_name == "Hybrid Algorithm Large":
            smac = HyperparameterOptimizationFacade(scenario, evaluate_hybrid_algorithm_large)
        else:
            raise ValueError("Unknown scenario name")

        incumbent = smac.optimize()
        print("Best found configuration:")
        print(incumbent)
    # -------------------- -------------------- -------------------- --------------------
