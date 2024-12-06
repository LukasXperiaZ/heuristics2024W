import pickle
import random
import time
from enum import Enum
from itertools import chain

import numpy as np
from pymhlib.solution import VectorSolution

from src.evaluation import ObjIter, Stats
from src.local_search import StepFunction, LocalSearchSolution

obj_huge = 999999999999999999999999999999999999999


class MWCCPNeighborhoods(Enum):
    """
    Note that 1 and 2 are a subset of 3. However, 1 and 2 are distinct.
    """

    # Flip two adjacent vertices v1 v2 -> v2 v1
    flip_two_adjacent_vertices = 1

    # Flip three adjacent vertices (2 times flip two adjacent vertices):
    # v1 v2 v3 -> v2 v1 v3 -> v2 v3 v1
    flip_three_adjacent_vertices = 2

    # Flip four adjacent vertices (3 times flip two adjacent vertices:
    # v1 v2 v3 v4 -> v2 v1 v3 v4 -> v2 v3 v1 v4 -> v2 v3 v4 v1
    flip_four_adjacent_vertices = 3

    def __str__(self):
        if self == MWCCPNeighborhoods.flip_two_adjacent_vertices:
            return "flip_two_adj"
        elif self == MWCCPNeighborhoods.flip_three_adjacent_vertices:
            return "flip_three_adj"
        else:
            return "flip_four_adj"


class MWCCPInstance:
    """
    Minimum Weighted Crossings with Constraints problem (MWCCP) instance.

    Given an undirected weighted bipartite graph G=(U ∪ V,E), find an ordering of nodes V such that
    the weighted edge crossings are minimized while satisfying all constraints C.

    Attributes
        - U: number of vertices (first partition)
        - V: number of vertices (second partition)
        - C: set of constraints
        - E: set of edges (u,v)
    """

    path: str

    U: [int]
    V: [int]
    C: [(int, int)]
    E: [(int, int, int)]

    edges_from_u: dict[int, (int, int)]
    edges_from_v: dict[int, (int, int)]

    # precomputed values for the order of nodes.
    # I.e. if node v1 is left of v2 then we can precompute the sum of the edges of v1 and v2 that intersect.
    #       If v1 is right if v2, we can also precompute the sum.
    #       ->  When we want to compute the objective value of a solution, we just go over every pair of vertices
    #           and determine whether v1 is left or right of v2 and add the precomputed sum of intersecting edges.
    pre_comp_val: dict[int, dict]

    def __init__(self, path, U, V, C, E):
        self.path = path
        self.U = U
        self.V = V
        self.C = C
        self.E = E

        # Convert the list of edges E into an adjacency matrix
        self.adj_matrix = self.create_bipartite_adjacency_matrix()
        self.create_edges_from_u_and_v()

        # print("-- -- MWCCPInstance: " + "Calculating the precomputed values of pairs of vertices ...")
        # start = time.time()
        pre_comp_path = path + "_pre_comp_val.pkl"
        if not self.read_precomputed_values(pre_comp_path):
            # The precomputed values are not saved yet, precompute and save them.
            self.pre_comp_val = self.precompute_values_of_pairs_of_vertices()
            self.write_precomputed_values(pre_comp_path, self.pre_comp_val)
        # end = time.time()
        # print("-- -- MWCCPInstance: " + "Precomputed values of pairs of vertices finished in: " + f"{(end-start):.6f}s")

    def precompute_values_of_pairs_of_vertices(self):
        pre_comp_val: dict[int, dict] = {}

        if True:
            # Took 1.58
            for v1 in self.V:
                pre_comp_val[v1] = {}
                for v2 in self.V:
                    if v1 != v2:
                        pre_comp_val[v1][v2] = 0
                        # Assume v1 is left of v2, compute the resulting value.
                        # (Since we iterate over all pairs (v1 v2) in V, we get both combinations, e.g. (6,7) and (7,6)
                        #   in the case of just two vertices)
                        # Iterate over all pairs of edges adjacent to v1 and v2
                        for (u1, w1) in self.edges_from_v[v1]:
                            for (u2, w2) in self.edges_from_v[v2]:
                                if u1 > u2:
                                    pre_comp_val[v1][v2] += w1 + w2

        return pre_comp_val

    def read_precomputed_values(self, file_path):
        try:
            with open(file_path, "rb") as file:
                pre_compute_values = pickle.load(file)
                self.pre_comp_val = pre_compute_values
        except FileNotFoundError:
            return False

        return True

    def write_precomputed_values(self, file_path, pre_comp_val: dict[int, dict]):
        with open(file_path, "wb") as file:
            pickle.dump(pre_comp_val, file)

    def create_bipartite_adjacency_matrix(self):
        """
        Has the form e.g.:

               | v1 v2 .. .. vn
            ---|--------------
            u1 | 0  1
            u2 | 0  0
            .. |
            .. |
            un | 1  0        1

        """
        # |U| = |V| = n
        n = len(self.U)

        # Create an n x n matrix initialized with zeros
        # ( Note that we create a (n+1 + n+1) x (n+1) matrix since we leave the zero row and column empty.
        #   Furthermore, the vertices v start from n+1, therefore the rows 1..n are empty)
        adj_matrix = np.zeros(((n + 1) + (n + 1), n + 1), dtype=int)

        # Create a dictionary for fast lookup of vertex indices
        pos_U = {u: i + 1 for i, u in enumerate(self.U)}  # Position of U vertices
        pos_V = {v: i + 1 + n for i, v in enumerate(self.V)}  # Position of V vertices

        # Iterate through the edges (u, v, w) in E
        for (u, v, w) in self.E:
            if u in pos_U and v in pos_V:  # Ensure u is in U and v is in V
                pos_u = pos_U[u]  # Get index of u in U
                pos_v = pos_V[v]  # Get index of v in V

                # Populate the adjacency matrix
                adj_matrix[pos_v, pos_u] = w

        return adj_matrix

    def create_edges_from_u_and_v(self):
        self.edges_from_u = {}
        self.edges_from_v = {}

        for (u, v, w) in self.E:
            # add all edges that are adjacent to u
            if not u in self.edges_from_u:
                self.edges_from_u[u] = []
            self.edges_from_u[u].append((v, w))

            # add all edges that are adjacent to v
            if not v in self.edges_from_v:
                self.edges_from_v[v] = []
            self.edges_from_v[v].append((u, w))

        # If there is no edge adjacent to v, give it an empty list
        for v in self.V:
            if v not in self.edges_from_v:
                self.edges_from_v[v] = []
        # If there is no edge adjacent to u, give it an empty list
        for u in self.U:
            if u not in self.edges_from_u:
                self.edges_from_u[u] = []


class MWCCPSolution(VectorSolution, LocalSearchSolution):
    """
    Solution to a MWCCP instance.
    """

    to_maximize = False

    def __init__(self, inst: MWCCPInstance):
        super().__init__(len(inst.V), inst=inst)

        self.initialize(None)

    def copy(self):
        sol = MWCCPSolution(self.inst)
        sol.copy_from(self)
        return sol

    def copy_from(self, other: 'MWCCPSolution'):
        super().copy_from(other)

    def calc_objective_inefficient(self):
        # Precompute positions of each element in self.x
        pos_dict = {v: idx for idx, v in enumerate(self.x)}

        value = 0
        iteration = 1
        number_of_combinations = len(self.inst.E) * len(self.inst.E)
        # Loop over unique pairs of edges (u, v, w) and (u_, v_, w_)
        for (u, v, w) in self.inst.E:
            # Just iterate over u_ > u
            for u_ in range(u + 1, len(self.inst.E) + 1):
                # Check if there are any edges going from u_ to a vertex v_
                if u_ in self.inst.edges_from_u:
                    # Iterate over all edges that go from u_ to a vertex v_
                    for (v_, w_) in self.inst.edges_from_u[u_]:
                        if iteration % 100000000 == 0:
                            print("Iteration: " + str(iteration) + " of " + str(number_of_combinations) + " (" + str(
                                round(100 * (iteration / number_of_combinations), 2)) + "%)")
                        # Retrieve precomputed positions
                        pos_v = pos_dict.get(v)
                        pos_v_ = pos_dict.get(v_)

                        # Check the positions and update value
                        if pos_v > pos_v_:
                            value += w + w_
                        iteration += 1
        return value

    def calc_objective(self):
        value = 0
        # Iterate over all pairs of vertices in the current solution(self.x)
        for i in range(len(self.x)):
            v1 = self.x[i]
            for j in range(i + 1, len(self.x)):
                v2 = self.x[j]
                # Determine the contribution based on the order in self.x
                if v1 in self.inst.pre_comp_val and v2 in self.inst.pre_comp_val[v1]:
                    value += self.inst.pre_comp_val[v1][v2]

        return value

    def calc_objective_par(self, solution: [int]):
        value = 0
        # Iterate over all pairs of vertices in the current solution
        for i in range(len(solution)):
            v1 = solution[i]
            for j in range(i + 1, len(solution)):
                v2 = solution[j]
                # Determine the contribution based on the order in solution
                if v1 in self.inst.pre_comp_val and v2 in self.inst.pre_comp_val[v1]:
                    value += self.inst.pre_comp_val[v1][v2]

        return value

    def initialize(self, k):
        """
        Initialize the solution vector with the vertices from v.
        :param k: the value to initialize it with. If it is None, initialize it with V in ascending order.
        :return:
        """
        if k:
            for i in range(self.x.size):
                self.x[i] = k
        else:
            for i in range(self.x.size):
                self.x[i] = self.inst.V[i]

    def check(self):
        """
        Check if self.x is a valid solution.

        :raises ValueError: if a problem is detected
        """
        for v in self.inst.V:
            if len(np.where(self.x == v)[0]) < 1:
                raise ValueError(
                    "Vertex (" + str(v) + ") is not in the current solution: " + str(self.x))

        for (v_1, v_2) in self.inst.C:
            pos_v_1 = np.where(self.x == v_1)[0][0]
            pos_v_2 = np.where(self.x == v_2)[0][0]
            if pos_v_1 > pos_v_2:
                raise ValueError(
                    "Constraint (" + str(v_1) + "," + str(v_2) + ") is violated for the solution: " + str(self.x))

        super().check()

    def is_valid_solution(self, solution: [int]):
        """
        Checks if the solution is valid.
        :param solution:
        :return:
        """
        # Check if every vertex of V is in the solution
        for v in self.inst.V:
            if v not in solution:
                return False

        # Check if all constraints are satisfied
        for (v_1, v_2) in self.inst.C:
            pos_v_1 = solution.index(v_1)
            pos_v_2 = solution.index(v_2)
            if pos_v_1 > pos_v_2:
                return False

        return True

    def has_duplicates(self, solution: [int]):
        # ONLY for testing !!
        seen = set()
        for s in solution:
            if s in seen:
                print("Duplicate found: " + str(s))
                return True
            seen.add(s)
        return False

    def get_neighbor(self, current_solution: [int], current_obj: int, neighborhood: MWCCPNeighborhoods,
                     step_function: StepFunction) -> ([int], int):

        if step_function == StepFunction.first_improvement:
            """
            First improvement strategy
            """
            curr_sol = current_solution.copy()
            for i in range(len(current_solution) - 1):
                next_neighbor, next_obj = self.get_neighbor_neighborhood(curr_sol, current_obj, neighborhood, i)

                if not self.is_valid_solution(next_neighbor):
                    # If the next neighbor violates some constraints, move to the next one
                    continue

                if next_obj < current_obj:
                    # A better solution was found
                    return (next_neighbor, next_obj)
            # no better solution was found, return a bad objective value to indicate that the old solution is the
            # global maximum
            return (current_solution, (current_obj + 100) * 100)
        elif step_function == StepFunction.best_improvement:
            """
            Best improvement strategy
            """
            curr_sol = current_solution.copy()
            curr_obj = current_obj
            for i in range(len(current_solution) - 1):
                next_neighbor, next_obj = self.get_neighbor_neighborhood(curr_sol, current_obj, neighborhood, i)

                if not self.is_valid_solution(next_neighbor):
                    # If the next neighbor violates some constraints, move to the next one
                    continue

                if next_obj < current_obj:
                    curr_sol = next_neighbor
                    curr_obj = next_obj

            if curr_obj == current_obj:
                # no better solution was found, return a bad objective value to indicate that the old solution is the
                # global maximum
                return (current_solution, (current_obj + 100) * 100)
            # A better solution was found
            return (curr_sol, curr_obj)
        elif step_function == StepFunction.random:
            """
            Random strategy
            """
            curr_sol = current_solution.copy()
            j = random.randint(0, len(current_solution) - 1)
            for i in range(len(curr_sol) - 1):
                # Select a neighbor at random
                # If the current solution is not valid, we try the next possibility until we enumerated all possibilities
                next_index = (j + i) % (len(curr_sol) - 1)
                next_neighbor, next_obj = self.get_neighbor_neighborhood(curr_sol, current_obj, neighborhood,
                                                                         next_index)
                if self.is_valid_solution(next_neighbor):
                    return (next_neighbor, next_obj)

            # No better solution found
            return (current_solution, obj_huge)
        else:
            raise ValueError("Step function is not specified!")

    def get_neighbor_neighborhood(self, current_solution: [int], current_obj: int,
                                  neighborhood: MWCCPNeighborhoods, index: int):
        if neighborhood == MWCCPNeighborhoods.flip_two_adjacent_vertices:
            return self.flip_two_adjacent_vertices(current_solution, current_obj, index)
        elif neighborhood == MWCCPNeighborhoods.flip_three_adjacent_vertices:
            return self.flip_three_adjacent_vertices(current_solution, current_obj, index)
        elif neighborhood == MWCCPNeighborhoods.flip_four_adjacent_vertices:
            return self.flip_four_adjacent_vertices(current_solution, current_obj, index)
        else:
            raise ValueError("Neighborhood is not specified!")

    def flip_two_adjacent_vertices(self, sol_old, obj_old, i):
        """
        Get the objective value by using delta evaluation for the neighbor where two adjacent vertices were flipped.
        When flipping two adjacent vertices v1, v2, all the orders to the other vertices stay the same (e.g. if we have
        v1 v2 v3 v4, if we flip v2 and v3, they still both appear after v1 and before v4).
        I.e., we can compute the new objective value with:
            obj_old - pre_comp_value[v1][v2] + pre_comp_value[v2][v1]
        I.e., we use the precomputed values of the order of two vertices to efficiently compute the new value with
        delta evaluation.

        :param sol_old: old solution vector
        :param obj_old: old objective value
        :param i: first position of the two flipped vertices
        :return:  objective value using delta evaluation
        """
        next_neighbor = sol_old.copy()
        # Flip two adjacent neighbors
        next_neighbor[i] = sol_old[i + 1]
        next_neighbor[i + 1] = sol_old[i]

        v1 = sol_old[i]
        v2 = sol_old[i + 1]
        new_obj_val = obj_old - self.inst.pre_comp_val[v1][v2] + self.inst.pre_comp_val[v2][v1]

        return (next_neighbor, new_obj_val)

    def flip_three_adjacent_vertices(self, sol_old, obj_old, i):
        # Check if the index is too large
        last_index = len(sol_old) - 1
        index_for_flip_two = i + 1
        if index_for_flip_two + 1 > last_index:
            return (sol_old, obj_huge)

        # Flip three adjacent vertices (2 times flip two adjacent vertices):
        # v1 v2 v3 -> v2 v1 v3 -> v2 v3 v1
        temp_neighbor, temp_obj = self.flip_two_adjacent_vertices(sol_old, obj_old, i)
        return self.flip_two_adjacent_vertices(temp_neighbor, temp_obj, i + 1)

    def flip_four_adjacent_vertices(self, sol_old, obj_old, i):
        # Check if the index is too large
        last_index = len(sol_old) - 1
        index_for_flip_two = i + 2
        if index_for_flip_two + 1 > last_index:
            return (sol_old, obj_huge)

        # Flip four adjacent vertices (3 times flip two adjacent vertices:
        # v1 v2 v3 v4 -> v2 v1 v3 v4 -> v2 v3 v1 v4 -> v2 v3 v4 v1
        temp_neighbor, temp_obj = self.flip_three_adjacent_vertices(sol_old, obj_old, i)
        return self.flip_two_adjacent_vertices(temp_neighbor, temp_obj, i + 2)

    def local_search(self, initial_solution: [int], neighborhood: MWCCPNeighborhoods, step_function: StepFunction,
                     initial_obj: int = -1, max_iterations: int = -1, max_time_in_s: float = -1):
        solution: [int] = initial_solution.copy()

        if not self.is_valid_solution(initial_solution):
            raise ValueError("Initial solution is not valid!")
        # Calculate the obj value of the initial solution

        obj: int
        if initial_obj < 0:
            obj = self.calc_objective_par(initial_solution)
        else:
            obj = initial_obj

        obj_over_time: [ObjIter] = [ObjIter(obj, 0)]
        curr_iter = 1

        start = time.time()
        # =======================================
        while True:
            if max_iterations > 0:
                # If there is a limit on the number of iterations, break if it has been reached.
                if curr_iter > max_iterations:
                    break
            if max_time_in_s > 0:
                # If there is a limit on the amount of time it should run, break if it has been exceeded.
                curr_time = time.time()
                if curr_time - start > max_time_in_s:
                    break

            (next_neighbor, next_obj) = self.get_neighbor(solution, obj, neighborhood, step_function)

            if not self.is_valid_solution(next_neighbor):
                raise ValueError("The neighborhood '" + str(neighborhood) + "' with step function '" + str(
                    step_function) + "' returned a non-valid solution!")

            if next_obj <= obj:
                solution = next_neighbor
                obj = next_obj
            else:
                if step_function == step_function.first_improvement or step_function == step_function.best_improvement:
                    # If we have first or best improvement, that means that there is no better solution
                    # in the neighborhood -> we reached a local optimum which we can't escape from.
                    break

            obj_over_time.append(ObjIter(obj, curr_iter))
            curr_iter += 1
        # =======================================
        end = time.time()

        stats = Stats(title=str(step_function) + ", " + str(neighborhood), start_time=start, end_time=end,
                      iterations=curr_iter,
                      final_objective=obj, obj_over_time=obj_over_time)

        return solution, obj, stats

    def vnd(self, neighborhoods: [MWCCPNeighborhoods], step_function: StepFunction, max_iterations: int = -1,
            max_time_in_s: int = -1, initial_sol: ([], int) = None):
        if initial_sol is None:
            # Get the initial solution from the DCH
            sol, obj, _ = self.deterministic_construction_heuristic()
        else:
            sol, obj = initial_sol

        curr_solution: [int] = sol
        # The obj value of the initial solution
        curr_obj: int = obj

        obj_over_time: [ObjIter] = [ObjIter(curr_obj, 0)]

        l = 0
        l_max = len(neighborhoods) - 1
        curr_iter = 1

        start = time.time()
        # =======================================
        while l <= l_max:
            curr_neighborhood = neighborhoods[l]
            (next_neighbor, next_obj) = self.get_neighbor(curr_solution, curr_obj, curr_neighborhood, step_function)

            if not self.is_valid_solution(next_neighbor):
                raise ValueError("The neighborhood '" + str(curr_neighborhood) + "' with step function '" + str(
                    step_function) + "' returned a non-valid solution!")

            if next_obj < curr_obj:
                # better solution found
                curr_solution = next_neighbor
                curr_obj = next_obj
                l = 0
            else:
                l += 1

            obj_over_time.append(ObjIter(curr_obj, curr_iter))

            if max_iterations > 0:
                # If there is a limit on the number on iterations, break when it is reached
                if curr_iter >= max_iterations:
                    break
            if max_time_in_s > 0:
                # If there is a limit on the amount of time, break when it is reached
                curr_time = time.time()
                if curr_time - start > max_time_in_s:
                    break

            curr_iter += 1
        # =======================================
        end = time.time()

        stats = Stats(title="VND, " + str(step_function), start_time=start, end_time=end, iterations=curr_iter,
                      final_objective=curr_obj, obj_over_time=obj_over_time)

        return curr_solution, stats

    def grasp(self, neighborhood: MWCCPNeighborhoods, step_function: StepFunction, max_iterations: int = -1,
              max_time_in_s: int = -1, max_iter_local_search: int = -1):
        best_sol = None
        best_obj = obj_huge

        obj_over_time: [ObjIter] = []
        i = 0

        start = time.time()
        # =======================================
        while True:
            # Get the initial solution from the RCH
            sol, obj, _ = self.randomized_construction_heuristic()
            curr_solution: [int] = sol

            # Run a local search to get a local maximum
            loc_sol, loc_obj, _ = self.local_search(curr_solution, neighborhood, step_function, initial_obj=obj,
                                                    max_iterations=max_iter_local_search)

            if loc_obj < best_obj:
                best_sol = loc_sol
                best_obj = loc_obj

            obj_over_time.append(ObjIter(best_obj, i))

            if max_iterations > 0:
                if i >= max_iterations:
                    break
            if max_time_in_s > 0:
                curr_time = time.time()
                if curr_time - start > max_time_in_s:
                    break

            i += 1
        # ========================================
        end = time.time()

        stats = Stats(title="GRASP, " + str(step_function) + ", " + str(neighborhood), start_time=start, end_time=end,
                      iterations=i,
                      final_objective=best_obj, obj_over_time=obj_over_time)

        return best_sol, stats

    def genetic_algorithm(self, population_size: int = 100,
                          randomized_const_heuristic_initialization: str = "random_and_repair",
                          elitist_population: float = 0.2,
                          bot_population: float = 0.2,
                          k: int = 10,
                          crossover_range: int = 5,
                          mutation_prob: float = 0.05,
                          repair_percentage: float = 0.5,
                          penalize_factor: float = 1.5,
                          max_iterations: int = -1,
                          max_time_in_s: int = 10):
        """
        Genetic algorithm for MWCCP.
        Constraint handling: The algorithm assigns invalid solutions a worse fitness value.
        Inspiration: The algorithm behaves very similarly to the BRKGA. However, since not every permutation is a
                        valid solution, we use partially_matched_crossover to get a reasonable number of valid solutions.
                        Thus, we also do not use a bias. Instead, we use tournament selection to select the REST
                        candidates to be paired with the TOP candidates. This way, it is also possible that two parents
                        are from the TOP group.

        :param repair_percentage: The percentage of mid-solutions (from the crossover) that will be repaired.
        :param bot_population: Percentage of the bot population that is randomly generated in every iteration
        :param elitist_population: The percentage of the population that is considered to be the elite.
        :param mutation_prob: The probability of an allel to mutate
        :param penalize_factor: A factor how much worse invalid solutions should be (e.g. 1.5 means 1.5 times the value of the parent/original solution).
        :param crossover_range: The size of the crossover range of the partially matched crossover recombination
        :param population_size: the size of the population
        :param k: The number of individuals randomly chosen by the tournament selection.
        :param max_iterations: Maximum number of iterations.
        :param max_time_in_s: Maximum time the algorithm should run.
        :return: best found solution.
        """
        # Checks
        if population_size < k:
            raise ValueError("Population size must be greater than or equal to k")

        elitist_n = int(elitist_population * population_size)
        bot_n = int(bot_population * population_size)
        crossover_n = population_size - elitist_n - bot_n

        obj_over_time: [ObjIter] = []

        i = 0
        start = time.time()

        # initialize and evaluate P(t)
        # population is an array of solution-objective tuples, e.g. [([1,2,3], 9), ...]
        population: [([], int)] = []
        for j in range(population_size):
            if randomized_const_heuristic_initialization == "standard":
                sol, obj, _ = self.randomized_construction_heuristic()
            else:
                sol, obj = self.randomized_construction_heuristic_random_and_repair()
            population.append((sol, obj))

        population.sort(key=lambda x: x[1])

        obj_over_time.append(ObjIter(population[0][1], i))

        while i <= max_iterations or time.time() - start < max_time_in_s:
            i += 1

            # Select the top population
            top = population[:elitist_n]
            # Repair the top solutions if necessary
            top = self.repair(top, 1)

            # Do a tournament selection on the population to get the rest population
            rest = self.tournament_selection(population, k)

            # Do a partially matched crossover by choosing one parent of the top and one of the rest population
            mid = self.partially_matched_crossover(top, rest, crossover_n, crossover_range, penalize_factor)

            # Do insertion mutation
            mid = self.insertion_mutation(mid, mutation_prob, penalize_factor)

            mid = self.repair(mid, repair_percentage)

            # Replace the population with top, mid and random solutions
            p = self.replacement_brkga(top, mid, population_size)

            # set the new parents
            population = p

            obj_over_time.append(ObjIter(population[0][1], i))
        end = time.time()

        stats = Stats(title="Genetic Algorithm", start_time=start, end_time=end,
                      iterations=i,
                      final_objective=population[0][1], obj_over_time=obj_over_time)

        return population[0][0], stats

    def genetic_algorithm_with_vnd(self, population_size: int = 100,
                                   randomized_const_heuristic_initialization: str = "random_and_repair",
                                   elitist_population: float = 0.2,
                                   bot_population: float = 0.2,
                                   k: int = 10,
                                   crossover_range: int = 5,
                                   mutation_prob: float = 0.05,
                                   repair_percentage: float = 0.5,
                                   penalize_factor: float = 1.5,
                                   vnd_percentage: float = 0.5,
                                   vnd_max_runtime_in_s: int = 0.01,
                                   vnd_neighborhoods: [MWCCPNeighborhoods] = None,
                                   step_function: StepFunction = StepFunction.first_improvement,
                                   vnd_randomized_const_heuristic: str = "standard",
                                   max_iterations: int = -1,
                                   max_time_in_s: int = 10):
        """
        Hybrid approach that combines the genetic algorithm with VND.
        VND is used at the end of an iteration of the genetic algorithm on a random portion (on a part) of the population.

        :param randomized_const_heuristic_initialization: The randomized CH for the initialization phase
        :param vnd_randomized_const_heuristic: randomized construction heuristic that should be used for the BOT individuals.
        :param step_function: Stepfunction of VND
        :param vnd_neighborhoods: The neighborhoods that VND will use
        :param vnd_max_runtime_in_s: Maximal runtime of VND on one instance
        :param vnd_percentage: The percentage of the population to which vnd should be applied.

        :param repair_percentage: The percentage of mid-solutions (from the crossover) that will be repaired.
        :param bot_population: Percentage of the bot population that is randomly generated in every iteration
        :param elitist_population: The percentage of the population that is considered to be the elite.
        :param mutation_prob: The probability of an allel to mutate
        :param penalize_factor: A factor how much worse invalid solutions should be (e.g. 1.5 means 1.5 times the value of the solution this solution was created from).
        :param crossover_range: The size of the crossover range of the partially matched crossover recombination
        :param population_size: the size of the population
        :param k: The number of individuals randomly chosen by the tournament selection.
        :param max_iterations: Maximum number of iterations.
        :param max_time_in_s: Maximum time the algorithm should run.
        :return: best found solution.
        """
        # Checks
        if population_size < k:
            raise ValueError("Population size must be greater than or equal to k")

        if vnd_neighborhoods is None:
            vnd_neighborhoods = [
                MWCCPNeighborhoods.flip_two_adjacent_vertices,
                MWCCPNeighborhoods.flip_three_adjacent_vertices,
                MWCCPNeighborhoods.flip_four_adjacent_vertices]

        elitist_n = int(elitist_population * population_size)
        bot_n = int(bot_population * population_size)
        crossover_n = population_size - elitist_n - bot_n

        obj_over_time: [ObjIter] = []

        i = 0
        start = time.time()

        # initialize and evaluate P(t)
        # population is an array of solution-objective tuples, e.g. [([1,2,3], 9), ...]
        population: [([], int)] = []
        for j in range(population_size):
            if randomized_const_heuristic_initialization == "standard":
                sol, obj, _ = self.randomized_construction_heuristic()
            else:
                sol, obj = self.randomized_construction_heuristic_random_and_repair()
            population.append((sol, obj))

        population.sort(key=lambda x: x[1])

        obj_over_time.append(ObjIter(population[0][1], i))

        while i <= max_iterations or time.time() - start < max_time_in_s:
            i += 1

            # Select the top population
            top = population[:elitist_n]
            # Repair the top solutions if necessary
            top = self.repair(top, 1)

            # Do a tournament selection on the population to get the rest population
            rest = self.tournament_selection(population, k)

            # Do a partially matched crossover by choosing one parent of the top and one of the rest population
            mid = self.partially_matched_crossover(top, rest, crossover_n, crossover_range, penalize_factor)

            # Do insertion mutation
            mid = self.insertion_mutation(mid, mutation_prob, penalize_factor)

            mid = self.repair(mid, repair_percentage)

            # Replace the population with top, mid and random solutions.
            # NEW: The random solutions are enhanced with VND.
            p = self.replacement_brkga_with_VND(top, mid, population_size, vnd_percentage, vnd_neighborhoods,
                                                step_function, vnd_randomized_const_heuristic, vnd_max_runtime_in_s)

            # set the new parents
            population = p

            obj_over_time.append(ObjIter(population[0][1], i))
        end = time.time()

        stats = Stats(title="Genetic Algorithm", start_time=start, end_time=end,
                      iterations=i,
                      final_objective=population[0][1], obj_over_time=obj_over_time)

        return population[0][0], stats

    def repair(self, population: [([], int)], repair_percentage: float):
        population.sort(key=lambda x: x[1])
        repair_n = int(repair_percentage * len(population))
        for i in range(repair_n):
            sol, _ = population[i]

            # repair the solution
            sol = self.repair_solution(sol)

            if not self.is_valid_solution(sol):
                raise ValueError("Solution is not valid after repair!")

            population[i] = (sol, self.calc_objective_par(sol))
        return population

    def repair_solution(self, sol: []):
        violated_constraints = self.get_violated_constraints(sol)  # O(|C|)
        while violated_constraints:
            v_k, v_j = violated_constraints[0]
            # Move v_k at the position of v_j and push v_j and its successors to the right
            self.resolve_constraint(sol, v_k, v_j)

            violated_constraints = self.get_violated_constraints(sol)
        return sol

    def tournament_selection(self, population: [([], int)], k):
        selected_individuals = []
        for i in range(len(population)):
            # choose k individuals uniformly at random
            k_individuals = []
            for r in range(k):
                rand_individual = random.choice(population)
                k_individuals.append(rand_individual)

            # select the best individual
            best = k_individuals[0]
            for (sol, obj) in k_individuals:
                if obj < best[1]:
                    best = (sol, obj)
            selected_individuals.append(best)
        return selected_individuals

    def partially_matched_crossover(self, top: [([], int)], rest: [([], int)], size: int, crossover_range: int,
                                    penalize_factor: float):
        children: [([], int)] = []
        for i in range(int(size / 2)):
            # select 2 parents A and B
            (parent_a, obj_parent_a) = random.choice(top)
            (parent_b, obj_parent_b) = random.choice(rest)
            crossover_start = random.randint(0, len(parent_a) - 1 - crossover_range)

            # create children
            children_a = list.copy(parent_a)
            children_b = list.copy(parent_b)

            # For all genes in the crossover range:
            a_map_to: dict = {}
            b_map_to: dict = {}
            crossover_list_a = []
            crossover_list_b = []

            for j in range(crossover_start, crossover_start + crossover_range):
                # Swap the genes within the crossover range
                a_map_to[parent_b[j]] = children_a[j]
                children_a[j] = parent_b[j]
                crossover_list_a.append(children_a[j])

                b_map_to[parent_a[j]] = children_b[j]
                children_b[j] = parent_a[j]
                crossover_list_b.append(children_b[j])

            # Swap variables outside the crossover range with the old variables inside the crossover range.
            outside_left = range(0, crossover_start)
            outside_right = range(crossover_start + crossover_range, len(parent_a))
            for j in chain(outside_left, outside_right):
                gene_a = children_a[j]
                if gene_a in a_map_to:
                    repl_gene_a = a_map_to[gene_a]
                    while repl_gene_a in crossover_list_a:
                        # The gene is in the crossover list, we have to look what this should be replaced with
                        repl_gene_a = a_map_to[repl_gene_a]

                    children_a[j] = repl_gene_a

                gene_b = children_b[j]
                if gene_b in b_map_to:
                    repl_gene_b = b_map_to[gene_b]
                    while repl_gene_b in crossover_list_b:
                        # The gene is in the crossover list, we have to look what this should be replaced with
                        repl_gene_b = b_map_to[repl_gene_b]

                    children_b[j] = repl_gene_b

            # Calculate fitness of children (i.e. obj function)
            obj_a = obj_huge
            obj_b = obj_huge
            if self.is_valid_solution(children_a):
                obj_a = self.calc_objective_par(children_a)
            else:
                obj_a = int(obj_parent_a * penalize_factor)

            if self.is_valid_solution(children_b):
                obj_b = self.calc_objective_par(children_b)
            else:
                obj_b = int(obj_parent_b * penalize_factor)

            children.append((children_a, obj_a))
            children.append((children_b, obj_b))
        return children

    def insertion_mutation(self, solutions: [([], int)], mutation_prob: float, penalize_factor: float):
        mutated_solutions = list.copy(solutions)
        for j in range(len(mutated_solutions)):
            mut_solution, obj = mutated_solutions[j]
            for i in range(len(mut_solution)):
                if random.random() < mutation_prob:
                    # Mutate
                    new_loc = random.randint(0, len(mut_solution) - 1)
                    curr = mut_solution.pop(i)
                    mut_solution.insert(new_loc, curr)

            if self.is_valid_solution(mut_solution):
                obj_new = self.calc_objective_par(mut_solution)
            else:
                obj_new = int(obj * penalize_factor)
            mutated_solutions[j] = (mutated_solutions[j][0], obj_new)

        return mutated_solutions

    def replacement_brkga(self, top: [([], int)], mid: [([], int)], population_size: int):
        """
        Do the replacement according to the Biased Random Key Genetic Algorithm.
        I.e., we copy the top solutions, append the mid-solutions and append randomly generated solutions.
        """
        new_population = []

        # Append top and mid-solutions
        new_population = new_population + top + mid

        # Append random solutions
        while len(new_population) < population_size:
            rand_individual, obj, _ = self.randomized_construction_heuristic()
            new_population.append((rand_individual, obj))

        new_population.sort(key=lambda x: x[1])

        return new_population

    def replacement_brkga_with_VND(self, top: [([], int)], mid: [([], int)], population_size: int,
                                   vnd_percentage: float,
                                   vnd_neighborhoods, step_function, vnd_randomized_const_heuristic,
                                   vnd_max_runtime_in_s):
        new_population = []

        # Append top and mid-solutions
        new_population = new_population + top + mid

        # Get random solutions
        rand_sol = []
        while len(new_population) + len(rand_sol) < population_size:
            if vnd_randomized_const_heuristic == "standard":
                rand_individual, obj, _ = self.randomized_construction_heuristic()
            else:
                rand_individual, obj = self.randomized_construction_heuristic_random_and_repair()
            rand_sol.append((rand_individual, obj))

        # Improve (some) random solutions with vnd
        vnd_n = int(vnd_percentage * len(rand_sol))
        subset_indexes = random.sample(range(len(rand_sol)), vnd_n)
        for i in subset_indexes:
            rand_individual = rand_sol[i]
            sol, stats = self.vnd(vnd_neighborhoods, step_function, initial_sol=rand_individual,
                                  max_time_in_s=vnd_max_runtime_in_s)
            rand_sol[i] = (sol, stats.get_final_objective())

        # Append the improved random solutions
        new_population = new_population + rand_sol

        new_population.sort(key=lambda x: x[1])

        return new_population

    # =================================================================================
    #           Construction Heuristics
    # =================================================================================

    def deterministic_construction_heuristic(self):
        self.initialize(-1)
        x_temp: []
        x_temp = self.x.tolist()
        V_rem = self.inst.V.copy()
        u_i = 1
        start = time.time()
        while u_i <= len(self.inst.U):  # iterates |U| times
            # Get the vertex with maximum weight to the vertical counterpart u_i
            v_i = self.get_max_weight_vertex(V_rem, u_i)  # O(|E|)

            if not v_i:
                # If there is no vertex with an edge to u_i, choose the next vertex of V_rem
                v_i = V_rem[0]

            # Add v_i to the current position
            x_temp[u_i - 1] = v_i
            V_rem.remove(v_i)

            # While there are violated constraints of the form (v_k, v_j)
            violated_constraints = self.get_violated_constraints(x_temp)  # O(|C|)
            while violated_constraints:  # O(|violated_constraints| * |violated_constraints|)
                (v_k, v_j) = violated_constraints[0]
                # Move v_k at the position of v_j and push v_j and its successors to the right
                self.resolve_constraint(x_temp, v_k, v_j)  # O(1)

                violated_constraints.remove((v_k, v_j))
                # check, if other constraints could also be resolved
                self.remove_resolved_constraints(x_temp, violated_constraints)  # O(|violated_constraints|)

            u_i += 1
        end = time.time()

        self.x = np.array(x_temp)
        self.check()

        obj = self.calc_objective()

        return x_temp, obj, Stats(title="Deterministic CH", start_time=start, end_time=end,
                                  iterations=-1,
                                  final_objective=obj, obj_over_time=[obj, 0])

    def randomized_construction_heuristic(self):
        self.initialize(-1)
        x_temp: [] = self.x.tolist()
        V_rem = self.inst.V.copy()
        # -- Get a list of remaining elements that need a partner
        U_ = list(range(1, len(V_rem) + 1))
        # seed the randomness with the current time
        # --

        start = time.time()
        while U_:
            # -- Get a random element u_i from U_
            u_i = random.choice(U_)
            # --

            # Get the vertex with maximum weight to the vertical counterpart u_i
            v_i = self.get_max_weight_vertex(V_rem, u_i)

            if not v_i:
                # If there is no vertex with an edge to u_i, choose the next vertex of V_rem
                v_i = V_rem[0]

            # Add v_i to the current position
            x_temp[u_i - 1] = v_i
            V_rem.remove(v_i)

            # While there are violated constraints of the form (v_k, v_j)
            violated_constraints = self.get_violated_constraints(x_temp)
            while violated_constraints:
                v_k, v_j = violated_constraints[0]
                # Move v_k at the position of v_j and push v_j and its successors to the right
                self.resolve_constraint(x_temp, v_k, v_j)

                violated_constraints = self.get_violated_constraints(x_temp)

            u_i += 1
            # Update U_ (also in case items were moved because of violations of constraints)
            U_ = self.get_remaining_vertices_of_U(x_temp)
        end = time.time()

        self.x = np.array(x_temp)
        self.check()

        obj = self.calc_objective()

        return x_temp, obj, Stats(title="Randomized CH", start_time=start, end_time=end,
                                  iterations=-1,
                                  final_objective=obj, obj_over_time=[obj, 0])

    def randomized_construction_heuristic_random_and_repair(self):
        v: [] = self.x.tolist()
        # do a random permutation
        random.shuffle(v)

        # repair the solution
        repaired = self.repair_solution(v)

        if not self.is_valid_solution(repaired):
            raise ValueError("Repaired solution is not valid!")

        obj = self.calc_objective_par(repaired)

        return repaired, obj

    def get_max_weight_vertex(self, V_rem, u_i):
        best_v = None
        best_w = 0
        for (u, v, w) in self.inst.E:
            if u_i == u and v in V_rem:
                if w > best_w:
                    best_w = w
                    best_v = v

        return best_v

    def get_violated_constraints(self, x_temp):
        violated_constraints = []

        for (v_1, v_2) in self.inst.C:
            if v_1 in x_temp and v_2 in x_temp:
                pos_v_1 = x_temp.index(v_1)
                pos_v_2 = x_temp.index(v_2)
                # Both vertices have been set, check the constraint!
                if pos_v_1 > pos_v_2:
                    violated_constraints.append((v_1, v_2))

        return violated_constraints

    def resolve_constraint(self, x_temp, v_i, v_j):
        # Move v_i at the position of v_j and push v_j and its successors to the right
        pos_v_i = x_temp.index(v_i)
        pos_v_j = x_temp.index(v_j)

        # remove v_i
        x_temp.remove(v_i)

        # insert v_i at the position of v_j and move v_j and all other elements to the right.
        x_temp.insert(pos_v_j, v_i)

    def remove_resolved_constraints(self, x_temp, violated_constraints):
        for (v_1, v_2) in violated_constraints:
            if x_temp.index(v_1) < x_temp.index(v_2):
                violated_constraints.remove((v_1, v_2))

    def get_remaining_vertices_of_U(self, x_temp):
        U_ = []
        for i in range(len(x_temp)):
            if x_temp[i] == -1:
                U_.append(i + 1)
        return U_
