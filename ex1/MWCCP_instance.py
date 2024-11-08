import random

import numpy as np
from pymhlib.solution import VectorSolution


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

    def __init__(self, U, V, C, E):
        self.U = U
        self.V = V
        self.C = C
        self.E = E

        # Convert the list of edges E into an adjacency matrix
        self.adj_matrix = self.create_bipartite_adjacency_matrix()

        self.edges_from_u = self.create_edges_from_u()

    def create_bipartite_adjacency_matrix(self):
        # |U| = |V| = n
        n = len(self.U)

        # Create an n x n matrix initialized with zeros
        adj_matrix = np.zeros((n, n), dtype=int)

        # Create a dictionary for fast lookup of vertex indices
        pos_U = {v: i for i, v in enumerate(self.U)}  # Position of U vertices
        pos_V = {v: i for i, v in enumerate(self.V)}  # Position of V vertices

        # Iterate through the edges (u, v, w) in E
        for (u, v, w) in self.E:
            if u in pos_U and v in pos_V:  # Ensure u is in U and v is in V
                pos_u = pos_U[u]  # Get index of u in U
                pos_v = pos_V[v]  # Get index of v in V

                # Populate the adjacency matrix
                adj_matrix[pos_u, pos_v] = w

        return adj_matrix

    def create_edges_from_u(self):
        edges_from_u = {}

        for (u, v, w) in self.E:
            if not u in edges_from_u:
                edges_from_u[u] = []
            edges_from_u[u].append((v, w))
        return edges_from_u


class MWCCPSolution(VectorSolution):
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

    def calc_objective(self):
        # Precompute positions of each element in self.x
        pos_dict = {v: idx for idx, v in enumerate(self.x)}

        value = 0
        iteration = 1
        number_of_combinations = len(self.inst.E)*len(self.inst.E)
        # Loop over unique pairs of edges (u, v, w) and (u_, v_, w_)
        for (u, v, w) in self.inst.E:
            # Just iterate over u_ > u
            for u_ in range(u, len(self.inst.E)):
               # Check if there are any edges going from u_ to a vertex v_
                if u_ in self.inst.edges_from_u:
                    # Iterate over all edges that go from u_ to a vertex v_
                    for (v_, w_) in self.inst.edges_from_u[u_]:
                        if iteration % 100000000 == 0:
                            print("Iteration: " + str(iteration) + " of " + str(number_of_combinations) + " (" + str(round(100 * (iteration/number_of_combinations), 2)) + "%)")
                        # Retrieve precomputed positions
                        pos_v = pos_dict.get(v)
                        pos_v_ = pos_dict.get(v_)

                        # Check the positions and update value
                        if pos_v > pos_v_:
                            value += w + w_
                        iteration += 1
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

    def deterministic_construction_heuristic(self):
        self.initialize(-1)
        x_temp : []
        x_temp = self.x.tolist()
        V_rem = self.inst.V.copy()
        u_i = 1
        while u_i <= len(self.inst.U):  # iterates |U| times
            # Get the vertex with maximum weight to the vertical counterpart u_i
            v_i = self.get_max_weight_vertex(V_rem, u_i)     # O(|E|)

            if not v_i:
                # If there is no vertex with an edge to u_i, choose the next vertex of V_rem
                v_i = V_rem[0]

            # Add v_i to the current position
            x_temp[u_i - 1] = v_i
            V_rem.remove(v_i)

            # While v_i violates a constraint of the form (v_i, v_j)
            violated_constraints = self.get_violated_constraints(x_temp)    # O(|C|)
            while violated_constraints:     # O(|violated_constraints| * |violated_constraints|)
                _, v_j = violated_constraints[0]
                # Move v_i at the position of v_j and push v_j and its successors to the right
                self.resolve_constraint(x_temp, v_i, v_j)   # O(1)

                violated_constraints.remove((_, v_j))
                # check, if other constraints could also be resolved
                self.remove_resolved_constraints(x_temp, violated_constraints) # O(|violated_constraints|)

            u_i += 1

        self.x = np.array(x_temp)

    def randomized_construction_heuristic(self):
        # TODO not tested yet!!!
        self.initialize(-1)
        x_temp : []
        x_temp = self.x.tolist()
        V_rem = self.inst.V.copy()
        # -- Get a list of remaining elements that need a partner
        U_ = list(range(1, len(V_rem) + 1))
        # seed the randomness with the current time
        # --
        while U_:
            # -- Get a random element u_i from U_
            u_i = random.choice(U_)
            # remove u_i from U_
            U_.remove(u_i)
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
                self.resolve_constraint(x_temp, v_i, v_j) # TODO does not work for the randomized version

                violated_constraints = self.get_violated_constraints(x_temp)

            u_i += 1

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
