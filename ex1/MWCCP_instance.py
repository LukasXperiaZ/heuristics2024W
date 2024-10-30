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

    def initialize(self, _k):
        """
        Initialize the solution vector with the vertices from v in ascending order.
        :param _k:
        :return:
        """
        for i in range(self.x.size):
            self.x[i] = self.inst.V[i]

    def check(self):
        """
        Check if valid solution.

        :raises ValueError: if a problem is detected
        """
        for (v_1, v_2) in self.inst.C:
            pos_v_1 = np.where(self.x == v_1)[0][0]
            pos_v_2 = np.where(self.x == v_2)[0][0]
            if pos_v_1 > pos_v_2:
                raise ValueError("Constraint (" + str(v_1) + "," + str(v_2) + ") is violated for the solution: " + str(self.x))


        super().check()

        pass
