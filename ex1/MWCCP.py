import random
from enum import Enum

import numpy as np
from pymhlib.solution import VectorSolution

from ex1.local_search import StepFunction, LocalSearchSolution


class MWCCPNeighborhoods(Enum):
    """
    Note that 1 and 2 are a subset of 3. However, 1 and 2 are distinct.
    """

    # Flip two adjacent vertices v1 v2 -> v2 v1
    flip_two_adjacent_vertices = 1

    # Rotate the solution to the right
    rotate_to_the_right = 2

    # Move a vertex v from position i to position k (i != k)
    move_vertex_to_position = 3


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

    def __init__(self, U, V, C, E):
        self.U = U
        self.V = V
        self.C = C
        self.E = E

        # Convert the list of edges E into an adjacency matrix
        self.adj_matrix = self.create_bipartite_adjacency_matrix()

        self.create_edges_from_u_and_v()

        self.pre_comp_val = self.precompute_values_of_pairs_of_vertices()

    def precompute_values_of_pairs_of_vertices(self):
        pre_comp_val: dict[int, dict] = {}

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
        Check if valid solution.

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

    def get_neighbor(self, current_solution: [int], current_obj: int, neighborhood: MWCCPNeighborhoods,
                     step_function: StepFunction) -> ([int], int):
        if neighborhood == MWCCPNeighborhoods.flip_two_adjacent_vertices:
            return self.get_neighbor_flip_two_adjacent_vertices(current_solution, current_obj, step_function)
        elif neighborhood == MWCCPNeighborhoods.rotate_to_the_right:
            return self.get_neighbor_rotate_to_the_right(current_solution, current_obj, step_function)
        elif neighborhood == MWCCPNeighborhoods.move_vertex_to_position:
            return self.get_neighbor_move_vertex_to_position(current_solution, current_obj, step_function)
        else:
            raise ValueError("Neighborhood is not specified!")

    def get_neighbor_flip_two_adjacent_vertices(self, current_solution: [int], current_obj: int,
                                                step_function: StepFunction):
        if step_function.first_improvement:
            """
            First improvement strategy
            """
            current_sol = current_solution.copy()
            for i in range(len(current_solution) - 1):
                next_neighbor, next_obj = self.flip_two_adjacent_vertices(current_obj, current_sol, i)
                if next_obj < current_obj:
                    return (next_neighbor, next_obj)
            # no better solution was found, return the old one
            return (current_solution, current_obj)

    def flip_two_adjacent_vertices(self, obj_old, sol_old, i):
        """
        Get the objective value by using delta evaluation for the neighbor where two adjacent vertices were flipped.
        When flipping two adjacent vertices v1, v2, only an edge from v1 and an edge from v2 can change with respect
        to whether they intersect or do not intersect anymore. I.e., we can compute the new objective value with:
            obj_old - obj_old_v1_v2 + obj_new_v2_v1
        I.e. subtracting the objective value that the crossings of the edges going out from v1 and v2 yielded from the
        old objective value and then adding the new value that the crossings of the new constellation of v2 and v1
        yields.

        :param sol_old: old solution vector
        :param obj_old: old objective value
        :param i: first position of the two flipped vertices
        :return:  objective value using delta evaluation
        """
        next_neighbor = sol_old.copy()
        # Flip two adjacent neighbors
        next_neighbor[i] = sol_old[i + 1]
        next_neighbor[i + 1] = sol_old[i]

        # TODO rework this and do it more efficiently without the adjacency matrix
        # adj_matrix[v][u]
        adj_matrix = self.inst.adj_matrix

        v1 = sol_old[i]
        v2 = sol_old[i + 1]

        # (v1, u, w)
        edges_from_v1: [(int, int, int)] = []
        for u in range(len(adj_matrix[v1])):
            w = adj_matrix[v1][u]
            if w > 0:
                # Edge found
                edges_from_v1.append((v1, u, w))

        # (v2, u, w)
        edges_from_v2: [(int, int, int)] = []
        for u in range(len(adj_matrix[v2])):
            w = adj_matrix[v2][u]
            if w > 0:
                # Edge found
                edges_from_v2.append((v2, u, w))

        obj_old_v1_v2 = 0
        for (v1, u1, w1) in edges_from_v1:
            for (v2, u2, w2) in edges_from_v2:
                if u1 > u2:
                    # The edges cross
                    obj_old_v1_v2 += w1 + w2

        obj_new_v2_v1 = 0
        for (v2, u2, w2) in edges_from_v2:
            for (v1, u1, w1) in edges_from_v1:
                if u2 > u1:
                    # The edges cross
                    obj_new_v2_v1 += w2 + w1

        new_obj_val = obj_old - obj_old_v1_v2 + obj_new_v2_v1

        return (next_neighbor, new_obj_val)

    def get_neighbor_rotate_to_the_right(self, current_solution: [int], current_obj: int, step_function: StepFunction):
        # TODO
        raise NotImplementedError

    def get_neighbor_move_vertex_to_position(self, current_solution: [int], current_obj: int,
                                             step_function: StepFunction):
        # TODO
        raise NotImplementedError

    def run_local_search(self, neighborhood: MWCCPNeighborhoods, step_function: StepFunction, iterations: int):
        # Get the initial solution from the DCH
        self.deterministic_construction_heuristic()
        self.check()
        solution: [int] = self.x.tolist()
        # Calculate the obj value of the initial solution
        obj: int = self.calc_objective()
        print("Initial solution: " + str(solution))
        print("Initial obj value: " + str(obj))

        # TODO consider different StepFunctions: Random needs a number of iterations, while the other two reach a local
        # optima after x iterations whcih they can't escape from.
        for i in range(iterations):
            (next_neighbor, next_obj) = self.get_neighbor(solution, obj, neighborhood, step_function)
            if next_obj <= obj:
                solution = next_neighbor
                obj = next_obj

        return solution, obj

    def deterministic_construction_heuristic(self):
        self.initialize(-1)
        x_temp: []
        x_temp = self.x.tolist()
        V_rem = self.inst.V.copy()
        u_i = 1
        while u_i <= len(self.inst.U):  # iterates |U| times
            # Get the vertex with maximum weight to the vertical counterpart u_i
            v_i = self.get_max_weight_vertex(V_rem, u_i)  # O(|E|)

            if not v_i:
                # If there is no vertex with an edge to u_i, choose the next vertex of V_rem
                v_i = V_rem[0]

            # Add v_i to the current position
            x_temp[u_i - 1] = v_i
            V_rem.remove(v_i)

            # While v_i violates a constraint of the form (v_i, v_j)
            violated_constraints = self.get_violated_constraints(x_temp)  # O(|C|)
            while violated_constraints:  # O(|violated_constraints| * |violated_constraints|)
                _, v_j = violated_constraints[0]
                # Move v_i at the position of v_j and push v_j and its successors to the right
                self.resolve_constraint(x_temp, v_i, v_j)  # O(1)

                violated_constraints.remove((_, v_j))
                # check, if other constraints could also be resolved
                self.remove_resolved_constraints(x_temp, violated_constraints)  # O(|violated_constraints|)

            u_i += 1

        self.x = np.array(x_temp)

    def randomized_construction_heuristic(self):
        self.initialize(-1)
        x_temp: []
        x_temp = self.x.tolist()
        V_rem = self.inst.V.copy()
        # -- Get a list of remaining elements that need a partner
        U_ = list(range(1, len(V_rem) + 1))
        # seed the randomness with the current time
        # --
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

            # While v_i violates a constraint of the form (v_i, v_j)
            violated_constraints = self.get_violated_constraints(x_temp)
            while violated_constraints:
                v_i, v_j = violated_constraints[0]
                # Move v_i at the position of v_j and push v_j and its successors to the right
                self.resolve_constraint(x_temp, v_i, v_j)

                violated_constraints = self.get_violated_constraints(x_temp)

            u_i += 1
            # Update U_ (also in case items were moved because of violations of constraints)
            U_ = self.get_remaining_vertices_of_U(x_temp)

        self.x = np.array(x_temp)

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
