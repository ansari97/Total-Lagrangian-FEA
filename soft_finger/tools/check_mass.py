#!/usr/bin/env python3
"""
check_mass.py
=============

Check the scalar mass matrix m = [m_ij] exported by the FOM driver
(--dump_mass) and compare it with the exact consistent T10 mass matrix
computed in closed form from the same mesh.

Usage:
  python3 check_mass.py --mass ../runs/mass_coo.txt \
      --mesh ../meshes/finger_100x15x15mm_n20x2x2/finger_100x15x15mm_n20x2x2.npz \
      --density 1080
"""

import argparse

import numpy as np

# Exact T10 element mass pattern: m^E = rho * V_E / 420 * C (standard order:
# corners 0-3, then midside nodes of edges 01, 12, 02, 03, 13, 23).
EDGES = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]


def exact_pattern():
    C = np.zeros((10, 10))
    for i in range(4):
        for j in range(4):
            C[i, j] = 6.0 if i == j else 1.0
    for a in range(4):
        for k, (p, q) in enumerate(EDGES):
            C[a, 4 + k] = C[4 + k, a] = -4.0 if a in (p, q) else -6.0
    for k, ek in enumerate(EDGES):
        for l, el in enumerate(EDGES):
            if k == l:
                C[4 + k, 4 + l] = 32.0
            elif set(ek) & set(el):
                C[4 + k, 4 + l] = 16.0
            else:
                C[4 + k, 4 + l] = 8.0
    return C


def exact_mass(nodes, elements, rho):
    n = nodes.shape[0]
    C = exact_pattern()
    P0, P1, P2, P3 = (nodes[elements[:, k]] for k in range(4))
    V = np.einsum("ij,ij->i", P1 - P0, np.cross(P2 - P0, P3 - P0)) / 6.0
    m = np.zeros((n, n))
    for e, Ve in zip(elements, V):
        m[np.ix_(e, e)] += rho * Ve / 420.0 * C
    return m


def summarize(name, m):
    n = m.shape[0]
    ev = np.linalg.eigvalsh(m)
    scale = np.abs(ev).max()
    print(f"{name}:")
    print(f"  size {n} x {n}, nonzeros {np.count_nonzero(m)}")
    print(f"  symmetry   max|m - m^T| = {np.abs(m - m.T).max():.3e}")
    print(f"  total mass sum(m_ij)    = {m.sum():.6e} kg")
    print(f"  eigenvalues: min {ev.min():.3e}, max {ev.max():.3e}")
    print(f"  #negative = {(ev < -1e-12 * scale).sum()}, "
          f"#~zero = {(np.abs(ev) <= 1e-12 * scale).sum()}, "
          f"#positive = {(ev > 1e-12 * scale).sum()}")
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", required=True, help="COO file from --dump_mass")
    ap.add_argument("--mesh", required=True, help="mesh .npz file")
    ap.add_argument("--density", type=float, default=1080.0)
    a = ap.parse_args()

    mesh = np.load(a.mesh)
    nodes, elements = mesh["nodes"], mesh["elements"]
    n = nodes.shape[0]

    coo = np.loadtxt(a.mass, comments="#")
    rows, cols, vals = coo[:, 0].astype(int), coo[:, 1].astype(int), coo[:, 2]
    m_code = np.zeros((n, n))
    np.add.at(m_code, (rows, cols), vals)

    m_exact = exact_mass(nodes, elements, a.density)

    summarize("TL-FEA mass matrix (5-point rule)", m_code)
    summarize("Exact consistent mass matrix", m_exact)

    diff = np.linalg.norm(m_code - m_exact) / np.linalg.norm(m_exact)
    rs_diff = np.abs(m_code.sum(1) - m_exact.sum(1)).max()
    print("Comparison:")
    print(f"  relative difference ||m_code - m_exact|| / ||m_exact|| = {diff:.3e}")
    print(f"  max row-sum difference (gravity loads) = {rs_diff:.3e} kg")


if __name__ == "__main__":
    main()