from networkx.classes import edges

from ex1.MWCCP import MWCCPInstance


def read_instance(path: str) -> MWCCPInstance:
    with open(path, 'r') as instance:
        lines = instance.read().splitlines()

    # --- Process first line ---
    u_n, v_n, c_n, e_n = map(int, lines[0].split())

    U = list(range(1, u_n + 1))  # Create list of U
    V = list(range(u_n + 1, u_n + 1 + v_n))  # Create list of V
    C: [(int, int)] = []
    E: [(int, int, int)] = []
    # --- --- --- --- --- ---

    # --- Process the other lines ---
    constraints_s = False
    edges_s = False
    for line in lines[1:]:
        if line.startswith('#constraints'):
            constraints_s = True
            continue

        if line.startswith("#edges"):
            constraints_s = False
            edges_s = True
            continue

        if constraints_s:
            # read the constraints
            n1, n2 = map(int, line.split())
            C.append((n1, n2))

        elif edges_s:
            # read the edges
            u, v, w = map(int, line.split())
            E.append((u, v, w))
    # --- --- --- --- --- ---

    return MWCCPInstance(U, V, C, E)