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
        x_temp : []
        x_temp = self.x.tolist()
        V_rem = self.inst.V.copy()
        u_i = 1
        while u_i <= len(self.inst.U):
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
                _, v_j = violated_constraints[0]
                # Move v_i at the position of v_j and push v_j and its successors to the right
                self.resolve_constraint(x_temp, v_i, v_j)

                violated_constraints.remove((_, v_j))
                # check, if other constraints could also be resolved
                self.remove_resolved_constraints(x_temp, violated_constraints)

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

        # remove v_i, i.e. the last element
        x_temp.remove(v_i)

        # insert v_i at the position of v_j and move v_j and all other elements to the right.
        x_temp.insert(pos_v_j, v_i)

    def remove_resolved_constraints(self, x_temp, violated_constraints):
        for (v_1, v_2) in violated_constraints:
            if x_temp.index(v_1) < x_temp.index(v_2):
                violated_constraints.remove((v_1, v_2))
