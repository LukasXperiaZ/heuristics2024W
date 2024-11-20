import os
import unittest

from ex1.MWCCP import MWCCPSolution
from ex1.read_instance import read_instance


class Experiments(unittest.TestCase):

    def test_deterministic_and_randomized_ch_and_GRASP(self):
        small = "../data/test_instances/small/"
        repetitions = 30

        for small_test in os.listdir(small):
            path = small + small_test
            for rep in range(repetitions):
                mwccp_instance = read_instance(path)
                mwccp_solution = MWCCPSolution(mwccp_instance)
                sol, obj, stats = mwccp_solution.deterministic_construction_heuristic()