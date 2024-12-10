import os
import random
import unittest
import warnings

import numpy as np
from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UniformFloatHyperparameter
from smac import Scenario, HyperparameterOptimizationFacade

from src.MWCCP import MWCCPSolution
from src.read_instance import read_instance


def evaluate_genetic_algorithm_small(params, seed):
    random.seed(seed)

    directory = "../data/tuning/tuning/small/"

    return evaluate_genetic_algorithm(params, directory)


def evaluate_genetic_algorithm_medium(params, seed):
    random.seed(seed)

    directory = "../data/tuning/tuning/medium/"

    return evaluate_genetic_algorithm(params, directory)


def evaluate_genetic_algorithm_large(params, seed):
    random.seed(seed)

    directory = "../data/tuning/tuning/large/"
    return evaluate_genetic_algorithm(params, directory)


def evaluate_genetic_algorithm(params, directory):
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

    obj_values = np.array(obj_values)
    normalized_obj_value = (obj_values - np.min(obj_values)) / (
            np.max(obj_values) - np.min(obj_values) + 1)  # Problem with division by 0 -> +1
    mean_normalized_value = np.mean(normalized_obj_value)

    return mean_normalized_value


class Tuning(unittest.TestCase):
    def test_tune_genetic_algorithm_small(self):
        self.run_genetic_algorithm_test("Genetic Algorithm Small", 25)

    def test_tune_genetic_algorithm_medium(self):
        self.run_genetic_algorithm_test("Genetic Algorithm Medium", 99)

    def test_tune_genetic_algorithm_large(self):
        self.run_genetic_algorithm_test("Genetic Algorithm Large", 499)

    def run_genetic_algorithm_test(self, scenario_name, solution_size):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        cs = ConfigurationSpace()

        # Hyperparameters
        population_size = UniformIntegerHyperparameter("population_size", lower=50, upper=500, default_value=100)
        randomized_const_heuristic_initialization = CategoricalHyperparameter(
            "randomized_const_heuristic_initialization", ["standard", "random_and_repair"],
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
            n_trials=200,
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
