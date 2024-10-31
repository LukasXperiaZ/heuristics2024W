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
        value = 0
        for (u, v, w) in self.inst.E:
            for (u_, v_, w_) in self.inst.E:
                if u < u_:
                    pos_v = np.where(self.x == v)[0][0]
                    pos_v_ = np.where(self.x == v_)[0][0]
                    if pos_v > pos_v_:
                        value += w + w_
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
        V_rem = self.inst.V.copy()
        u_i = 1
        while u_i <= len(self.inst.U):
            # Get the vertex with maximum weight to the vertical counterpart u_i
            v_i = self.get_max_weight_vertex(V_rem, u_i)

            if not v_i:
                # If there is no vertex with an edge to u_i, choose the next vertex of V_rem
                v_i = V_rem[0]

            # Add v_i to the current position
            self.x[u_i - 1] = v_i
            V_rem.remove(v_i)

            # While v_i violates a constraint of the form (v_i, v_j)
            violated_constraints = self.get_violated_constraints()
            while violated_constraints:
                _, v_j = violated_constraints[0]
                # Move v_i at the position of v_j and push v_j and its successors to the right
                self.resolve_constraint(v_i, v_j)

                violated_constraints = self.get_violated_constraints()

            u_i += 1

    def get_max_weight_vertex(self, V_rem, u_i):
        best_v = None
        best_w = 0
        for (u, v, w) in self.inst.E:
            if u_i == u and v in V_rem:
                if w > best_w:
                    best_w = w
                    best_v = v

        return best_v

    def get_violated_constraints(self):
        violated_constraints = []

        for (v_1, v_2) in self.inst.C:
            pos_v_1 = np.where(self.x == v_1)[0]
            pos_v_2 = np.where(self.x == v_2)[0]
            if len(pos_v_1) > 0 and len(pos_v_2) > 0:
                pos_v_1 = np.where(self.x == v_1)[0][0]
                pos_v_2 = np.where(self.x == v_2)[0][0]
                # Both vertices have been set, check the constraint!
                if pos_v_1 > pos_v_2:
                    violated_constraints.append((v_1, v_2))

        return violated_constraints

    def resolve_constraint(self, v_i, v_j):
        pos_v_i = np.where(self.x == v_i)[0][0] # position of the last element
        pos_v_j = np.where(self.x == v_j)[0][0]

        # Push v_j and its successors to the right
        current = pos_v_i - 1
        while current >= pos_v_j:
            self.x[current + 1] = self.x[current]
            current -= 1

        # Move v_i at the position of v_j
        self.x[pos_v_j] = v_i
