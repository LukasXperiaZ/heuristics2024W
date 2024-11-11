from networkx.classes import edges

from ex1.MWCCP import MWCCPInstance


def read_instance(path: str) -> MWCCPInstance:
    instance = open(path, 'r')

    # --- Process first line ---
    first_line = instance.readline()
    u_n, v_n, c_n, e_n = first_line.split()
    u_n = int(u_n)
    v_n = int(v_n)
    c_n = int(c_n)
    e_n = int(e_n)

    U: [int] = []
    V: [int] = []
    C: [(int, int)] = []
    E: [(int, int, int)] = []

    for u in range(1, u_n + 1):
        U.append(u)

    for v in range(u_n + 1, u_n + 1 + v_n):
        V.append(v)
    # --- --- --- --- --- ---

    # --- Process the other lines ---
    constraints_title_s = True
    constraints_s = False
    edges_s = False
    for line in instance:
        if constraints_title_s:
            # assert that now the constraints begin
            constraints_title = line
            assert(constraints_title.startswith("#constraints"))
            constraints_title_s = False
            constraints_s = True
            continue

        if constraints_s:
            # check if there are more constraints
            if line.startswith("#edges"):
                constraints_s = False
                edges_s = True
                continue

            # read the constraints
            n1, n2 = line.split()
            n1 = int(n1)
            n2 = int(n2)

            C.append((n1, n2))

        if edges_s:
            # read the edges
            u, v, w = line.split()
            u = int(u)
            v = int(v)
            w = int(w)
            E.append((u, v, w))
    # --- --- --- --- --- ---
    instance.close()

    return MWCCPInstance(U, V, C, E)