import os
import unittest

from ex1.MWCCP import MWCCPSolution
from ex1.read_instance import read_instance


class Experiments(unittest.TestCase):

    def test_deterministic_and_randomized_ch_and_GRASP(self):
        small = "../data/test_instances/small/"
        repetitions = 30

        obj_sum_avg = 0
        time_sum_avg = 0.0

        n_instances = 0

        # === Deterministic CH ===
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
        print("Average objective value: " + str(obj_avg))
        print("Average time : " + f"{time_sum_avg:.6f}s")
        # === === === === === ===