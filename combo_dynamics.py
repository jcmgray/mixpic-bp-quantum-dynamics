"""
Source code for

Relevant measurements are (note you need trailing comma between pauli types):

- 'Z62'
- 'X13,29,31,Y9,30,Z8,12,17,28,32'
- 'X37,41,52,56,57,58,62,79,Y38,40,42,63,72,80,90,91,Z75'
- 'X37,41,52,56,57,58,62,79,Y75,Z38,40,42,63,72,80,90,91'
- 'M' (magnetization, not implemented for all methods)

Memory usage is approx:
- 8 * 30 *  2 * chi**3 bytes for PEPS
- 8 * 30 *  4 * chi**3 bytes for PEPO
- 8 * 30 * 12 * chi**3 bytes for MIX

"""

import collections
import functools
import itertools
import os
import time
from math import pi, log2, log10

import xyzpy as xyz
import autoray as ar
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.circuit import Gate

from quimb.experimental.belief_propagation.l2bp import (
    L2BP,
    compress_l2bp,
    contract_l2bp,
)
from quimb.experimental.belief_propagation.l1bp import (
    contract_l1bp,
)


def get_optimizer_exact(
    target_size=2**27,
    minimize="combo",
    max_time="rate:1e8",
    optlib="nevergrad",
    directory=True,
    progbar=False,
    **kwargs,
):
    import cotengra as ctg

    if "parallel" not in kwargs:
        if "OMP_NUM_THREADS" in os.environ:
            parallel = int(os.environ["OMP_NUM_THREADS"])
        else:
            import multiprocessing as mp

            parallel = mp.cpu_count()

        if parallel == 1:
            parallel = False

        kwargs["parallel"] = parallel

    if target_size is not None:
        kwargs["slicing_reconf_opts"] = dict(target_size=target_size)
    else:
        kwargs["reconf_opts"] = {}

    return ctg.ReusableHyperOptimizer(
        progbar=progbar,
        minimize=minimize,
        max_time=max_time,
        optlib=optlib,
        directory=directory,
        **kwargs,
    )


def choose_optimizer(optimize, chi, version, target_size=2**27):
    if optimize is None:
        if ("square" in version) or ("torus" in version):
            # square lattices
            if chi <= 12:
                return "auto-hq"

        elif chi <= 64:
            # heavy hex or loops
            return "auto-hq"

    return get_optimizer_exact(target_size=target_size)


ibm_kyiv_edges = {
    # Row 0
    (0, 1): "g",
    (1, 2): "r",
    (2, 3): "b",
    (3, 4): "g",
    (4, 5): "r",
    (5, 6): "b",
    (6, 7): "r",
    (7, 8): "g",
    (8, 9): "r",
    (9, 10): "b",
    (10, 11): "g",
    (11, 12): "b",
    (12, 13): "r",
    # Connect rows 0-1
    (0, 14): "r",
    (14, 18): "b",
    (4, 15): "b",
    (15, 22): "r",
    (8, 16): "b",
    (16, 26): "g",
    (12, 17): "g",
    (17, 30): "r",
    # Row-1
    (18, 19): "g",
    (19, 20): "b",
    (20, 21): "r",
    (21, 22): "b",
    (22, 23): "g",
    (23, 24): "r",
    (24, 25): "g",
    (25, 26): "r",
    (26, 27): "b",
    (27, 28): "g",
    (28, 29): "b",
    (29, 30): "g",
    (30, 31): "b",
    (31, 32): "r",
    # Connect rows 1-2
    (20, 33): "g",
    (33, 39): "r",
    (24, 34): "b",
    (34, 43): "g",
    (28, 35): "r",
    (35, 47): "b",
    (32, 36): "g",
    (36, 51): "r",
    # Row-2
    (37, 38): "g",
    (38, 39): "b",
    (39, 40): "g",
    (40, 41): "b",
    (41, 42): "g",
    (42, 43): "b",
    (43, 44): "r",
    (44, 45): "b",
    (45, 46): "r",
    (46, 47): "g",
    (47, 48): "r",
    (48, 49): "b",
    (49, 50): "r",
    (50, 51): "g",
    # Connect rows 2-3
    (37, 52): "r",
    (52, 56): "g",
    (41, 53): "r",
    (53, 60): "b",
    (45, 54): "g",
    (54, 64): "b",
    (49, 55): "g",
    (55, 68): "r",
    # Row-3
    (56, 57): "r",
    (57, 58): "b",
    (58, 59): "g",
    (59, 60): "r",
    (60, 61): "g",
    (61, 62): "r",
    (62, 63): "b",
    (63, 64): "r",
    (64, 65): "g",
    (65, 66): "b",
    (66, 67): "r",
    (67, 68): "b",
    (68, 69): "g",
    (69, 70): "b",
    # Connect rows 3-4
    (58, 71): "r",
    (71, 77): "b",
    (62, 72): "g",
    (72, 81): "r",
    (66, 73): "g",
    (73, 85): "r",
    (70, 74): "r",
    (74, 89): "b",
    # Row-4
    (75, 76): "b",
    (76, 77): "r",
    (77, 78): "g",
    (78, 79): "r",
    (79, 80): "b",
    (80, 81): "g",
    (81, 82): "b",
    (82, 83): "g",
    (83, 84): "r",
    (84, 85): "b",
    (85, 86): "g",
    (86, 87): "b",
    (87, 88): "r",
    (88, 89): "g",
    # Connect rows 4-5
    (75, 90): "g",
    (90, 94): "b",
    (79, 91): "g",
    (91, 98): "b",
    (83, 92): "b",
    (92, 102): "r",
    (87, 93): "g",
    (93, 106): "b",
    # Row-5
    (94, 95): "r",
    (95, 96): "g",
    (96, 97): "r",
    (97, 98): "g",
    (98, 99): "r",
    (99, 100): "b",
    (100, 101): "g",
    (101, 102): "b",
    (102, 103): "g",
    (103, 104): "r",
    (104, 105): "g",
    (105, 106): "r",
    (106, 107): "g",
    (107, 108): "r",
    # Connect rows 5-6
    (96, 109): "b",
    (109, 114): "r",
    (100, 110): "r",
    (110, 118): "b",
    (104, 111): "b",
    (111, 122): "g",
    (108, 112): "b",
    (112, 126): "g",
    # Row-6
    (113, 114): "b",
    (114, 115): "g",
    (115, 116): "r",
    (116, 117): "b",
    (117, 118): "g",
    (118, 119): "r",
    (119, 120): "g",
    (120, 121): "r",
    (121, 122): "b",
    (122, 123): "r",
    (123, 124): "b",
    (124, 125): "g",
    (125, 126): "b",
}

loop_31_edges = [
    (0, 1),
    (2, 5),
    (3, 6),
    (4, 8),
    (7, 13),
    (14, 20),
    (17, 23),
    (18, 24),
    (21, 25),
    (22, 27),
    (26, 29),
    (0, 2),
    (4, 7),
    (5, 9),
    (6, 12),
    (10, 16),
    (11, 17),
    (15, 21),
    (19, 25),
    (20, 26),
    (24, 27),
    (0, 3),
    (1, 4),
    (5, 10),
    (6, 11),
    (8, 14),
    (9, 15),
    (12, 18),
    (13, 19),
    (16, 22),
    (23, 26),
    (25, 28),
    (27, 30),
]


loop_12_edges = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 0),
]

loop_21_edges = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 0),
    (0, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (16, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (20, 2),
]

coo2int = collections.defaultdict(itertools.count().__next__)
square_16_edges = tuple((coo2int[i], coo2int[j]) for i, j in qtn.edges_2d_square(4, 4))

coo2int = collections.defaultdict(itertools.count().__next__)
square_20_edges = tuple((coo2int[i], coo2int[j]) for i, j in qtn.edges_2d_square(5, 4))

coo2int = collections.defaultdict(itertools.count().__next__)
torus_16_edges = tuple(
    (coo2int[i], coo2int[j]) for i, j in qtn.edges_2d_square(4, 4, cyclic=True)
)

coo2int = collections.defaultdict(itertools.count().__next__)
torus_20_edges = tuple(
    (coo2int[i], coo2int[j]) for i, j in qtn.edges_2d_square(5, 4, cyclic=True)
)


layer_0 = []
layer_1 = []
layer_2 = []

for (na, nb), c in ibm_kyiv_edges.items():
    if c == "r":
        layer_0.append((na, nb))
    elif c == "b":
        layer_1.append((na, nb))
    elif c == "g":
        layer_2.append((na, nb))


def make_gates(
    steps=20,
    theta_h=pi / 4,
    theta_j=-pi / 2,
    lightcone=None,
    group=False,
    final_rx=False,
):
    gates = []
    for r in range(steps):
        for i in range(127):
            gates.append(Gate("rx", [theta_h], [i], round=r))

        for layer in [layer_0, layer_1, layer_2]:
            for na, nb in layer:
                gates.append(Gate("rzz", [theta_j], [na, nb], round=r))

    if final_rx:
        for i in range(127):
            gates.append(Gate("rx", [theta_h], [i], round=r))

    if lightcone is not None:
        lgates = []
        lightcone = set(lightcone)
        for g in reversed(gates):
            qs = set(g.qubits)
            if qs & lightcone:
                lgates.append(g)
                lightcone |= qs

        gates = lgates[::-1]

    if group:
        import cytoolz

        gates = cytoolz.groupby(lambda g: g.round, gates)

    return gates


def make_gates_from_edges(
    steps=20,
    theta_h=pi / 4,
    theta_j=pi / 2,
    lightcone=None,
    group=False,
    final_rx=False,
    edges=(),
):
    sites = sorted({i for edge in edges for i in edge})

    gates = []
    for round in range(steps):
        for i in sites:
            gates.append(Gate("rx", [theta_h], [i], round=round))
        for na, nb in edges:
            gates.append(Gate("rzz", [theta_j], [na, nb], round=round))
    if final_rx:
        for i in sites:
            gates.append(Gate("rx", [theta_h], [i], round=round))

    if lightcone is not None:
        lgates = []
        lightcone = set(lightcone)
        for g in reversed(gates):
            qs = set(g.qubits)
            if qs & lightcone:
                lgates.append(g)
                lightcone |= qs

        gates = lgates[::-1]

    if group:
        import cytoolz

        gates = cytoolz.groupby(lambda g: g.round, gates)

    return gates


versions = {
    "loop12": (
        12,
        functools.partial(make_gates_from_edges, edges=loop_12_edges),
    ),
    "loop21": (
        21,
        functools.partial(make_gates_from_edges, edges=loop_21_edges),
    ),
    "loop31": (
        31,
        functools.partial(make_gates_from_edges, edges=loop_31_edges),
    ),
    "square16": (
        16,
        functools.partial(make_gates_from_edges, edges=square_16_edges),
    ),
    "square20": (
        20,
        functools.partial(make_gates_from_edges, edges=square_20_edges),
    ),
    "torus16": (
        16,
        functools.partial(make_gates_from_edges, edges=torus_16_edges),
    ),
    "torus20": (
        20,
        functools.partial(make_gates_from_edges, edges=torus_20_edges),
    ),
}


def make_circuit(
    steps=20,
    theta_h=pi / 4,
    theta_j=-pi / 2,
    final_rx=False,
    lightcone=None,
    version="ibm_kyiv",
    **kwargs,
):
    if version == "ibm_kyiv":
        circ = qtn.Circuit(127, **kwargs)
        gates = make_gates(
            steps=steps,
            theta_h=theta_h,
            theta_j=theta_j,
            final_rx=final_rx,
            lightcone=lightcone,
        )
    else:
        n, _make_gates = versions[version]
        circ = qtn.Circuit(n, **kwargs)
        gates = _make_gates(
            steps=steps,
            theta_h=theta_h,
            theta_j=theta_j,
            final_rx=final_rx,
            lightcone=lightcone,
        )

    for gate in gates:
        circ.apply_gate(gate)

    return circ


def get_to_backend(backend):
    if "torch" in backend:
        import torch

        if "cpu" in backend:
            device = "cpu"
        elif "gpu" in backend:
            device = "cuda"
        elif "mps" in backend:
            device = "mps"
        else:
            # default to gpu
            device = "cuda"

        if "double" in backend:
            dtype = torch.complex128
        else:
            # default to single precision
            dtype = torch.complex64

        def to_backend(x):
            return torch.tensor(x, dtype=dtype, device=device)

    elif "cupy" in backend:
        import cupy

        if "double" in backend:
            dtype = "complex128"
        else:
            dtype = "complex64"

        def to_backend(x):
            return cupy.asarray(x, dtype=dtype)

    elif "tensorflow" in backend:
        import tensorflow as tf

        if "cpu" in backend:
            device = "/cpu:0"
        elif "gpu" in backend:
            device = "/gpu:0"
        else:
            # default to gpu
            device = "/gpu:0"

        if "double" in backend:
            dtype = tf.complex128
        else:
            # default to single precision
            dtype = tf.complex64

        def to_backend(x):
            with tf.device(device):
                return tf.constant(x, dtype=dtype)

    else:  # assume numpy
        if "double" in backend:
            dtype = "complex128"
        else:
            dtype = "complex64"

        def to_backend(x):
            return x.astype(dtype)

    return to_backend


def torch_to_gpu(x):
    return x.cuda()


def torch_to_cpu(x):
    return x.cpu()


def parse_string_to_paulis(s):
    s = s.replace(" ", "")
    ops = s.split(",")
    which = None
    pstring = {}
    for op in ops:
        if op[0].upper() in "XYZ":
            which = op[0]
            op = op[1:]
            pstring[which] = []
        pstring[which].append(int(op))
    return pstring


def run_exact(
    steps,
    theta_h=8 * pi / 32,
    theta_j=-pi / 2,
    measure="Z62",
    version="ibm_kyiv",
    atol=1e-12,
    method=None,
    backend="numpy",
    optimize=None,
    progbar=False,
):
    if optimize is None:
        optimize = get_optimizer_exact(progbar=progbar)

    final_rx = measure == "X37,41,52,56,57,58,62,79,Y38,40,42,63,72,80,90,91,Z75"

    circ = make_circuit(
        steps,
        theta_h=theta_h,
        theta_j=theta_j,
        final_rx=final_rx,
        version=version,
    )

    if measure == "M":
        # XXX: this bit doesn't use advanced simplification or backend yet
        Z = 0.0
        for i in range(circ.N):
            Z = Z + circ.local_expectation(qu.pauli("z"), i, optimize=optimize)

        Z = Z / circ.N
        real = ar.to_numpy(Z.real)
        imag = ar.to_numpy(Z.imag)
        return real, imag

    pstring = parse_string_to_paulis(measure)
    all_qubits = [q for qs in pstring.values() for q in qs]
    ket = circ.get_psi_reverse_lightcone(all_qubits)

    if method == "dense":
        # directly contract the full wavefunction
        to_backend = get_to_backend(backend)
        ket.apply_to_arrays(to_backend)
        tket = ket.contract(optimize=optimize)
        tketG = tket.copy()
        for s, where in pstring.items():
            for q in where:
                tketG.gate_(to_backend(qu.pauli(s)), ket.site_ind(q))
        tket.conj_()
        Z = tketG @ tket
        real = ar.to_numpy(Z.real)
        imag = ar.to_numpy(Z.imag)
        return real, imag

    # lazily form the full expectation TN to simplify
    bra = ket.conj()
    for s, where in pstring.items():
        for q in where:
            ket.gate_(qu.pauli(s), q, contract=True)

    tn = ket & bra

    # simplify: this is only 'compressed/approximate' up to tolerance atol
    tn.compress_simplify_((), atol=atol, progbar=progbar, equalize_norms=False)

    if tn.num_tensors == 1:
        # fully simplified
        Z = tn.arrays[0]
    else:
        tree = tn.contraction_tree(optimize=optimize)

        to_backend = get_to_backend(backend)
        tn.apply_to_arrays(to_backend)
        Z = tree.contract(tn.arrays, progbar=progbar)

    real = ar.to_numpy(Z.real)
    imag = ar.to_numpy(Z.imag)
    return real, imag


def run_bppeps(
    steps=9,
    theta_h=8 * pi / 32,
    theta_j=-pi / 2,
    measure="Z62",
    measure_local_check="auto",
    version="ibm_kyiv",
    chi=8,
    cutoff=5e-6,
    cutoff_mode="rsum2",
    tol=None,
    tol_final=None,
    max_iterations=256,
    last_step=-1,
    block=1,
    use_lightcone=True,
    backend="numpy-double",
    precision="double",  # dummy kwarg to allow as xyzpy dimension
    optimize=None,
    target_size=2**27,
    return_tn_after=None,
    compute_exact=False,
    progbar=False,
    **bp_opts,
):
    t0 = time.time()

    if tol is None:
        # set BP tolerance same as cutoff
        tol = cutoff
    if tol_final is None:
        tol_final = 10 * tol

    optimize = choose_optimizer(optimize, chi, version, target_size)

    pstring = parse_string_to_paulis(measure)
    all_qubits = [q for qs in pstring.values() for q in qs]
    final_rx = measure == "X37,41,52,56,57,58,62,79,Y38,40,42,63,72,80,90,91,Z75"
    circ = make_circuit(
        steps,
        theta_h=theta_h,
        theta_j=theta_j,
        tags="ROUND_0",
        final_rx=final_rx,
        lightcone=all_qubits if use_lightcone else None,
        version=version,
    )
    psi = circ.psi

    to_backend = get_to_backend(backend)
    psi.apply_to_arrays(to_backend)

    if "viagpu" in backend:
        bp_opts["via"] = (torch_to_gpu, torch_to_cpu)

    if measure_local_check == "auto":
        # check if `measure` is of form "Z{int}"
        if tuple(pstring.keys()) == ("Z",) and len(pstring["Z"]) == 1:
            (measure_local_check,) = pstring["Z"]
        else:
            # otherwise just pick first qubit
            measure_local_check = 0

    rblock = 0
    for r in range(steps + last_step):
        # add another layer to the inner PEPO
        psi.retag_({f"ROUND_{r}": "OUTER"})

        rblock += 1
        if rblock < block:
            # keep accumulating into blocks
            continue
        else:
            # reset block counter
            rblock = 0

        if progbar:
            print(f"compressing layer {r}")

        psi, tn_outer = psi.partition("OUTER", inplace=True)
        compress_l2bp(
            tn_outer,
            max_bond=chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            tol=cutoff,
            optimize=optimize,
            progbar=progbar,
            inplace=True,
            **bp_opts,
        )

        if progbar:
            print("max_bond:", tn_outer.max_bond())
            print(xyz.report_memory())
            print(xyz.report_memory_gpu())

        psi.add_tensor_network(tn_outer, virtual=True)
        psi.check()

        if progbar:
            print("----------------------------------------------")

    if return_tn_after == "final":
        return psi

    if progbar:
        print(f"... contract norm and estimate Z{measure_local_check} directly ...")

    if compute_exact:
        Nex = complex((psi.H & psi).contract(optimize=optimize))

    bp = L2BP(psi, optimize=optimize, **bp_opts)
    bp.run(max_iterations=max_iterations, tol=tol, progbar=progbar)
    est_norm = complex(bp.contract())

    rho = bp.partial_trace(measure_local_check)
    Zcheck = complex(ar.do("trace", rho @ to_backend(qu.pauli("Z"))))

    if progbar:
        print(f"-> norm:{est_norm} Z{measure_local_check}:{Zcheck}")
        print(xyz.report_memory())
        print(xyz.report_memory_gpu())
        print(f"... contracting obs after compressing layer {r} ...")

    psiG = psi.copy()
    for s, where in pstring.items():
        for q in where:
            psiG.gate_(to_backend(qu.pauli(s)), q, contract=True)
    expec = psi.H & psiG

    if compute_exact:
        Oex = complex(expec.contract(optimize=optimize))

    info = {}
    obs = complex(
        contract_l1bp(
            expec,
            tol=tol_final,
            max_iterations=max_iterations,
            optimize=optimize,
            info=info,
            progbar=progbar,
            **bp_opts,
        )
    )

    if progbar:
        print(f"-> obs:{obs}")
        print(xyz.report_memory())
        print(xyz.report_memory_gpu())

    res = {
        "O": obs.real,
        "Oim": obs.imag,
        "N": est_norm.real,
        "Nim": est_norm.imag,
        "Z": Zcheck.real,
        "time": time.time() - t0,
        "O_converged": info["converged"],
    }
    if compute_exact:
        res["Oex"] = Oex.real
        res["Nex"] = Nex.imag

    return res


def run_bppepo(
    steps=9,
    theta_h=8 * pi / 32,
    theta_j=-pi / 2,
    measure="Z62",
    version="ibm_kyiv",
    chi=8,
    cutoff=5e-6,
    cutoff_mode="rsum2",
    tol=None,
    tol_final=None,
    max_iterations=256,
    last_step=-1,
    block=1,
    use_lightcone=True,
    backend="numpy-double",
    precision="double",  # dummy kwarg to allow as xyzpy dimension
    optimize=None,
    target_size=2**27,
    return_tn_after=None,
    compute_norm=True,
    compute_exact=False,
    progbar=False,
    **bp_opts,
):
    t0 = time.time()

    if tol is None:
        tol = cutoff
    if tol_final is None:
        tol_final = 10 * tol
    if last_step == -1:
        # match message sizes in BP2 and BP1 stages
        last_step = max(1, int(log2(chi) / 2))

    optimize = choose_optimizer(optimize, chi, version, target_size)

    pstring = parse_string_to_paulis(measure)
    all_qubits = [q for qs in pstring.values() for q in qs]

    final_rx = measure == "X37,41,52,56,57,58,62,79,Y38,40,42,63,72,80,90,91,Z75"
    circ = make_circuit(
        steps,
        theta_h=theta_h,
        theta_j=theta_j,
        tags="ROUND_0",
        final_rx=final_rx,
        lightcone=all_qubits if use_lightcone else None,
        version=version,
    )
    ket = circ.psi
    bra = ket.conj()

    ket.add_tag("KET")
    bra.add_tag("BRA")

    # add expectation gates
    for which, qubits in parse_string_to_paulis(measure).items():
        for q in qubits:
            ket.gate_inds_(qu.pauli(which), [f"k{q}"], contract=True)

    # form full sandwich
    tn = ket & bra

    to_backend = get_to_backend(backend)
    tn.apply_to_arrays(to_backend)

    if "viagpu" in backend:
        bp_opts["via"] = (torch_to_gpu, torch_to_cpu)

    if return_tn_after == "initial":
        return tn

    r = 0
    rblock = 0
    for r in reversed(range(last_step, steps)):
        # add another layer to the inner PEPO
        tn.retag_({f"ROUND_{r}": "INNER"})
        rblock += 1

        if rblock < block:
            # keep accumulating into blocks
            continue
        else:
            # reset block counter
            rblock = 0

        if return_tn_after == f"applied_{r}":
            return tn.select("INNER")

        # split out middle pepo
        tn, tn_inner = tn.partition("INNER", inplace=True)

        if r == return_tn_after:
            return tn_inner

        if r != 0:
            if progbar:
                print(f"compressing layer {r}")

            tn_inner.equalize_norms_()
            compress_l2bp(
                tn_inner,
                max_bond=chi,
                tol=tol,
                max_iterations=max_iterations,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                progbar=progbar,
                optimize=optimize,
                inplace=True,
                **bp_opts,
            )

        tn_inner.equalize_norms_()

        if progbar:
            print("max bond:", tn_inner.max_bond())
            print(xyz.report_memory())
            print(xyz.report_memory_gpu())
            print("----------------------------------------------")

        # recombine
        tn.add_tensor_network(tn_inner, check_collisions=False, virtual=True)
        tn.check()

    if return_tn_after == "final":
        return tn

    if compute_norm:
        if progbar:
            print("computing norm...")

        try:
            tn_inner = tn.select("INNER")
            bp = L2BP(tn_inner, optimize=optimize, **bp_opts)
            bp.run(tol=tol, max_iterations=max_iterations, progbar=progbar)
            _, norm_exponent = bp.contract(strip_exponent=True)
            N = float(10 ** ((norm_exponent - (len(bp.local_tns) * log10(2))) / 2))
        except KeyError:
            # no INNER layer becuase small steps value
            N = 1.0
    else:
        N = None

    if compute_exact:
        tree = tn.contraction_tree(optimize)
        if tree.contraction_cost() < 1e15:
            Zex = tree.contract(tn.arrays, progbar=progbar)
            real_ex = ar.to_numpy(Zex.real)
            imag_ex = ar.to_numpy(Zex.imag)
        else:
            import warnings

            warnings.warn("Tree too large to contract.")
            real_ex = imag_ex = float("nan")
    else:
        real_ex = imag_ex = None

    if progbar:
        print(xyz.report_memory())
        print(xyz.report_memory_gpu())
        print(f"contracting after compressing layer {r}")

    info = {}
    obs = complex(
        contract_l1bp(
            tn,
            tol=tol_final,
            max_iterations=max_iterations,
            optimize=optimize,
            info=info,
            progbar=progbar,
            **bp_opts,
        )
    )

    if progbar:
        print(f"-> {obs}")
        print(xyz.report_memory())
        print(xyz.report_memory_gpu())

    res = {
        "O": obs.real,
        "Oim": obs.imag,
        "O_converged": info["converged"],
        "time": time.time() - t0,
    }
    if compute_norm:
        res["N"] = N
    if compute_exact:
        res["Oex"] = real_ex
        res["Oimex"] = imag_ex

    return res


def run_bpmixed(
    steps=9,
    theta_h=8 * pi / 32,
    theta_j=-pi / 2,
    measure="Z62",
    version="ibm_kyiv",
    fraction=1 / 2,
    peps_step=None,
    pepo_step=None,
    chi=8,
    cutoff=5e-6,
    cutoff_mode="rsum2",
    tol=None,
    tol_final=None,
    max_iterations=256,
    block=1,
    use_lightcone=True,
    backend="numpy-double",
    precision="double",  # dummy kwarg to allow as xyzpy dimension
    optimize=None,
    target_size=2**27,
    return_tn_after=None,
    progbar=False,
    **bp_opts,
):
    t0 = time.time()

    if tol is None:
        tol = cutoff
    if tol_final is None:
        tol_final = 10 * tol

    optimize = choose_optimizer(optimize, chi, version, target_size)

    # get circuit and measurement
    pstring = parse_string_to_paulis(measure)
    all_qubits = [q for qs in pstring.values() for q in qs]
    final_rx = measure == "X37,41,52,56,57,58,62,79,Y38,40,42,63,72,80,90,91,Z75"
    circ = make_circuit(
        steps,
        theta_h=theta_h,
        theta_j=theta_j,
        tags="ROUND_0",  # group initial state into round 0
        final_rx=final_rx,
        lightcone=all_qubits if use_lightcone else None,
        version=version,
    )

    if peps_step is None:
        peps_step = int(steps * fraction)
    if pepo_step is None:
        pepo_step = peps_step

    # get initial psi wavefunction
    to_backend = get_to_backend(backend)
    if "viagpu" in backend:
        bp_opts["via"] = (torch_to_gpu, torch_to_cpu)

    psi = circ.psi
    psi.apply_to_arrays(to_backend)

    # PEPS evolution steps
    tn_outer = None
    rblock = 0
    for r in range(peps_step):
        # add another layer to the inner PEPO
        psi.retag_({f"ROUND_{r}": "OUTER"})

        rblock += 1
        if rblock < block:
            # keep accumulating into blocks
            continue
        else:
            # reset block counter
            rblock = 0

        if progbar:
            print(f"compressing PEPS layer {r}")

        psi, tn_outer = psi.partition("OUTER", inplace=True)
        compress_l2bp(
            tn_outer,
            chi,
            cutoff,
            cutoff_mode=cutoff_mode,
            tol=tol,
            progbar=progbar,
            inplace=True,
            optimize=optimize,
            **bp_opts,
        )

        if progbar:
            print("max_bond:", tn_outer.max_bond())
            print(xyz.report_memory())
            print(xyz.report_memory_gpu())
            print("----------------------------------------------")

        psi.add_tensor_network(tn_outer, virtual=True)
        psi.check()

    if progbar:
        print("... contracting PEPS norm")

    if tn_outer is not None:
        norm_peps = contract_l2bp(
            tn_outer,
            tol=tol,
            max_iterations=max_iterations,
            optimize=optimize,
            progbar=progbar,
            **bp_opts,
        )
    else:
        # too short depth to have an outer PEPS
        norm_peps = 1.0

    if progbar:
        print(f"-> {norm_peps}")
        print(xyz.report_memory())
        print(xyz.report_memory_gpu())
        print("----------------------------------------------")

    # PEPO evolution steps
    # form full sandwich, with obs gates applied to ket side
    ket = psi
    bra = ket.conj()
    # add measuring gates
    for which, qubits in parse_string_to_paulis(measure).items():
        for q in qubits:
            ket.gate_inds_(to_backend(qu.pauli(which)), [f"k{q}"], contract=True)

    # form full sandwich
    tn = ket | bra

    rblock = 0
    tn_inner = None

    for r in reversed(range(pepo_step, steps)):
        tn.retag_({f"ROUND_{r}": "INNER"})
        rblock += 1

        if rblock < block:
            # keep accumulating into blocks
            continue
        else:
            # reset block counter
            rblock = 0

        # split out middle pepo
        tn, tn_inner = tn.partition("INNER", inplace=True)

        if progbar:
            print(f"compressing PEPO layer {r}")

        compress_l2bp(
            tn_inner,
            max_bond=chi,
            tol=tol,
            max_iterations=max_iterations,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            progbar=progbar,
            optimize=optimize,
            inplace=True,
            **bp_opts,
        )

        if progbar:
            print("max_bond:", tn_inner.max_bond())
            print(xyz.report_memory())
            print(xyz.report_memory_gpu())
            print("----------------------------------------------")

        tn.add_tensor_network(tn_inner, check_collisions=False, virtual=True)
        tn.check()

    if return_tn_after == "final":
        return tn

    if progbar:
        print("... contracting PEPO norm")

    if tn_inner is not None:
        bp = L2BP(tn_inner, optimize=optimize, **bp_opts)
        bp.run(
            tol=tol,
            max_iterations=max_iterations,
            progbar=progbar,
        )
        mantissa, norm_exponent = bp.contract(strip_exponent=True)
        # add contributions from identities missing due to lightcone
        norm_pepo = (
            mantissa * 10 ** (norm_exponent - len(bp.local_tns) * log10(2))
        ) ** 0.5
    else:
        # too short depth to have an inner PEPO
        norm_pepo = 1.0

    if progbar:
        print(f"-> {norm_pepo}")
        print(xyz.report_memory())
        print(xyz.report_memory_gpu())
        print("!!! contracting observable !!!")

    # now run 1-norm BP on the triple sandwich
    info = {}
    obs = complex(
        contract_l1bp(
            tn,
            optimize=optimize,
            tol=tol_final,
            max_iterations=max_iterations,
            info=info,
            progbar=progbar,
            **bp_opts,
        )
    )

    if progbar:
        print(f"-> {obs}")
        print(xyz.report_memory())
        print(xyz.report_memory_gpu())

    norm_peps = complex(norm_peps)
    norm_pepo = complex(norm_pepo)
    norm = norm_peps * norm_pepo

    return {
        "O": obs.real,
        "Oim": obs.imag,
        "O_converged": info["converged"],
        "Npeps": norm_peps.real,
        "Npeps_im": norm_peps.imag,
        "Npepo": norm_pepo.real,
        "Npepo_im": norm_pepo.imag,
        "N": norm.real,
        "Nim": norm.imag,
        "time": time.time() - t0,
    }
