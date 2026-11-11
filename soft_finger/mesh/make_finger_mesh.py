#!/usr/bin/env python3
"""
make_finger_mesh.py
===================

Structured, mirror-symmetric T10 (10-node quadratic tetrahedron) mesh of a
rectangular soft finger, written in the TetGen text format read by TL-FEA
(ANCFCPUUtils::FEAT10_read_nodes / FEAT10_read_elements).

Geometry and coordinate convention (SI units, meters)
-----------------------------------------------------
  * Finger axis along z. Base (clamped) face at z = 0, finger hangs down to
    z = -L (gravity along -z).
  * Cross-section centered on the axis: x in [-Sx/2, Sx/2], y in [-Sy/2, Sy/2].
  * Bending happens in the x-z plane; y = 0 is the mirror-symmetry plane.

Mesh construction
-----------------
  * The finger is divided into nz x nx x ny boxes (z, x, y).
  * Each box is split into 6 tetrahedra along one body diagonal (Kuhn split).
    Using the same pattern in every box makes neighboring tets share faces
    exactly (conforming mesh).
  * Boxes with y < 0 use the mirror image of the pattern used for y > 0, so
    the mesh is exactly mirror-symmetric about y = 0 (ny must be even).
  * T10 midside nodes sit at exact edge midpoints and are shared between
    neighboring elements. All nodes live on a (2nx+1) x (2ny+1) x (2nz+1)
    lattice.

Element node ordering written to the .ele file ("standard" order, which the
TL-FEA reader detects automatically):
    v0, v1, v2, v3, m(0-1), m(1-2), m(0-2), m(0-3), m(1-3), m(2-3)

Output files (in <outdir>/<name>/)
-----------------------------------
  <name>.1.node          node coordinates (1-based ids, TetGen format)
  <name>.1.ele           T10 connectivity (1-based ids, TetGen format)
  <name>.base.txt        0-based indices of clamped base nodes (z = 0)
  <name>.tip_weights.txt 0-based tip-face node indices and traction weights
                         (nodal force = weight * F; weights sum to 1)
  <name>.track.txt       0-based index of the tracking node (tip-face center)
  <name>.midplane.txt    0-based indices of nodes on the symmetry plane y = 0
  <name>.npz             everything above as NumPy arrays (for Python/Colab)
  <name>.info.json       parameters, counts, and check results

All node indices in the companion files are 0-based, matching the arrays
used inside TL-FEA after it reads the mesh.

Usage examples
--------------
  python3 make_finger_mesh.py                         # defaults below
  python3 make_finger_mesh.py --nz 40 --nx 4 --ny 4   # production mesh
  python3 make_finger_mesh.py --length 0.12 --size-x 0.02 --size-y 0.015
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a structured, mirror-symmetric T10 finger mesh "
                    "in TetGen format for TL-FEA.")
    p.add_argument("--length", type=float, default=0.100,
                   help="finger length L along z, in meters (default 0.100)")
    p.add_argument("--size-x", type=float, default=0.015,
                   help="cross-section size along x (bending direction), "
                        "in meters (default 0.015)")
    p.add_argument("--size-y", type=float, default=0.015,
                   help="cross-section size along y (out-of-plane), "
                        "in meters (default 0.015)")
    p.add_argument("--nz", type=int, default=20,
                   help="number of boxes along the length (default 20)")
    p.add_argument("--nx", type=int, default=2,
                   help="number of boxes along x (default 2)")
    p.add_argument("--ny", type=int, default=2,
                   help="number of boxes along y, must be even (default 2)")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--outdir", type=str,
                   default=os.path.normpath(os.path.join(script_dir, "..",
                                                         "meshes")),
                   help="parent output folder (default: ../meshes next to "
                        "this script)")
    p.add_argument("--name", type=str, default=None,
                   help="mesh name (default built from dimensions and "
                        "resolution, e.g. finger_100x15x15mm_n20x2x2)")
    return p.parse_args()


def default_name(a):
    def mm(v):
        s = f"{v * 1000:.3f}".rstrip("0").rstrip(".")
        return s.replace(".", "p")
    return (f"finger_{mm(a.length)}x{mm(a.size_x)}x{mm(a.size_y)}mm_"
            f"n{a.nz}x{a.nx}x{a.ny}")


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------
# Kuhn split of the unit cube: one tet per permutation of the axes, following
# the path (0,0,0) -> e_p0 -> e_p0 + e_p1 -> (1,1,1).
def kuhn_tets():
    tets = []
    for perm in itertools.permutations(range(3)):
        v = np.zeros(3, dtype=int)
        path = [v.copy()]
        for ax in perm:
            v[ax] = 1
            path.append(v.copy())
        tets.append(path)               # 4 local corners, each in {0,1}^3
    return tets


# Standard T10 edge order for midside nodes 4..9.
EDGES_STD = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]


def build_mesh(L, Sx, Sy, nz, nx, ny):
    """Return node coordinates (N x 3), elements (E x 10, 0-based, standard
    order) and the integer lattice coordinates (N x 3) of every node."""
    NI, NJ, NK = 2 * nx + 1, 2 * ny + 1, 2 * nz + 1   # lattice sizes

    # Lattice spacing. Coordinates are computed as (index - center) * h so the
    # symmetry plane is exactly y = 0 and mirrored nodes are exact negatives.
    hx, hy, hz = Sx / (2 * nx), Sy / (2 * ny), L / (2 * nz)

    # Node numbering: z (base -> tip) outermost, then y, then x.
    def lattice_id(I, J, K):
        return (K * NJ + J) * NI + I

    local_tets = kuhn_tets()
    elements = []
    for bz in range(nz):
        for by in range(ny):
            mirrored = by < ny // 2          # box lies in y < 0
            for bx in range(nx):
                for tet in local_tets:
                    corners = []
                    for (dx, dy, dz) in tet:
                        if mirrored:
                            dy = 1 - dy      # mirror image of the pattern
                        corners.append(np.array(
                            [2 * (bx + dx), 2 * (by + dy), 2 * (bz + dz)]))
                    elements.append(corners)

    # Turn lattice corners into T10 connectivity (lattice ids for now).
    conn = []
    for corners in elements:
        c = list(corners)
        # Fix orientation: the parent T10 element (nodes at (0,0,0), (1,0,0),
        # (0,1,0), (0,0,1)) has positive volume; enforce the same here.
        # Physical position uses z = -K*hz, so evaluate the sign physically.
        P = [np.array([(v[0] - nx) * hx, (v[1] - ny) * hy, -v[2] * hz])
             for v in c]
        vol6 = np.linalg.det(np.column_stack([P[1] - P[0], P[2] - P[0],
                                              P[3] - P[0]]))
        if vol6 < 0:
            c[1], c[2] = c[2], c[1]
        ids = [lattice_id(*v) for v in c]
        for (i, j) in EDGES_STD:
            m = (c[i] + c[j]) // 2            # exact: corners are even
            ids.append(lattice_id(*m))
        conn.append(ids)
    conn = np.array(conn, dtype=np.int64)

    # Keep only lattice points that are actually used, renumbered 0..N-1
    # in lattice order (so base nodes come first).
    used = np.unique(conn)
    new_id = -np.ones(NI * NJ * NK, dtype=np.int64)
    new_id[used] = np.arange(used.size)
    elements = new_id[conn]

    K, rem = np.divmod(used, NI * NJ)
    J, I = np.divmod(rem, NI)
    lattice = np.column_stack([I, J, K])
    nodes = np.column_stack([(I - nx) * hx, (J - ny) * hy, -K * hz])
    return nodes, elements, lattice


# ---------------------------------------------------------------------------
# Tip traction weights (consistent nodal loads for a uniform traction)
# ---------------------------------------------------------------------------
# Face j of a tet is the face opposite corner j; listed with its midside
# node positions (in standard order) for each corner pair.
FACES = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]
EDGE_TO_MID = {frozenset(e): 4 + k for k, e in enumerate(EDGES_STD)}


def tip_traction_weights(nodes, elements, lattice, nz):
    """For a uniform traction on the tip face (z = -L), the consistent nodal
    loads of each 6-node face triangle are: 0 at the 3 corners, A/3 at each of
    the 3 midside nodes (A = triangle area). Weights are normalized by the
    total tip area so that nodal force = weight * total force."""
    K_tip = 2 * nz
    on_tip = lattice[:, 2] == K_tip
    w = np.zeros(nodes.shape[0])
    total_area = 0.0
    for e in elements:
        for face in FACES:
            corner_nodes = [e[c] for c in face]
            if all(on_tip[n] for n in corner_nodes):
                p0, p1, p2 = (nodes[n] for n in corner_nodes)
                area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
                total_area += area
                for a, b in [(face[0], face[1]), (face[1], face[2]),
                             (face[0], face[2])]:
                    w[e[EDGE_TO_MID[frozenset((a, b))]]] += area / 3.0
    tip_nodes = np.nonzero(w)[0]
    weights = w[tip_nodes] / total_area
    return tip_nodes, weights, total_area


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def tet_volumes(nodes, elements):
    P0, P1, P2, P3 = (nodes[elements[:, k]] for k in range(4))
    return np.einsum("ij,ij->i", P1 - P0, np.cross(P2 - P0, P3 - P0)) / 6.0


def run_checks(nodes, elements, lattice, args, tip_nodes, weights, tip_area,
               base, midplane, track):
    L, Sx, Sy = args.length, args.size_x, args.size_y
    nx, ny, nz = args.nx, args.ny, args.nz
    tol = 1e-12
    results = {}

    def check(name, ok, detail=""):
        results[name] = {"ok": bool(ok), "detail": detail}
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" +
              (f"  ({detail})" if detail else ""))

    print("Checks:")

    # 1. Node count matches the full lattice (every lattice point used).
    expect_n = (2 * nx + 1) * (2 * ny + 1) * (2 * nz + 1)
    check("node count", nodes.shape[0] == expect_n,
          f"{nodes.shape[0]} nodes, expected {expect_n}")

    # 2. Positive element volumes, total = L * Sx * Sy.
    vol = tet_volumes(nodes, elements)
    check("all element volumes positive", np.all(vol > 0),
          f"min volume {vol.min():.3e} m^3")
    v_exact = L * Sx * Sy
    check("total volume", abs(vol.sum() - v_exact) <= tol * v_exact,
          f"{vol.sum():.12e} vs exact {v_exact:.12e} m^3")

    # 3. Midside nodes exactly at edge midpoints.
    err = 0.0
    for k, (i, j) in enumerate(EDGES_STD):
        mid = 0.5 * (nodes[elements[:, i]] + nodes[elements[:, j]])
        err = max(err, np.abs(nodes[elements[:, 4 + k]] - mid).max())
    check("midside nodes at edge midpoints", err <= tol * max(L, Sx, Sy),
          f"max error {err:.2e} m")

    # 4. Conformity: each face shared by at most 2 tets, and the boundary
    #    faces (used once) add up to the box surface area.
    face_count = {}
    face_area = {}
    for e in elements:
        for face in FACES:
            key = tuple(sorted(e[c] for c in face))
            face_count[key] = face_count.get(key, 0) + 1
            if key not in face_area:
                p0, p1, p2 = (nodes[n] for n in key)
                face_area[key] = 0.5 * np.linalg.norm(np.cross(p1 - p0,
                                                               p2 - p0))
    max_share = max(face_count.values())
    bnd_area = sum(face_area[k] for k, c in face_count.items() if c == 1)
    a_exact = 2 * (L * Sx + L * Sy + Sx * Sy)
    check("conforming faces (each face used by 1 or 2 tets)", max_share <= 2,
          f"max tets per face {max_share}")
    check("boundary surface area", abs(bnd_area - a_exact) <= 1e-10 * a_exact,
          f"{bnd_area:.12e} vs exact {a_exact:.12e} m^2")

    # 5. Exact mirror symmetry about y = 0 (nodes and elements).
    key_to_id = {tuple(l): i for i, l in enumerate(lattice)}
    mirror = np.array([key_to_id.get((I, 2 * ny - J, K), -1)
                       for (I, J, K) in lattice])
    ok_nodes = np.all(mirror >= 0) and np.array_equal(
        nodes[mirror], nodes * np.array([1.0, -1.0, 1.0]))
    elem_sets = {tuple(sorted(e[:4])) for e in elements}
    mirrored_sets = {tuple(sorted(mirror[e[:4]])) for e in elements}
    check("mirror symmetry about y = 0", ok_nodes and
          elem_sets == mirrored_sets,
          "every node and element has an exact mirror partner")

    # 6. Tip traction weights.
    check("tip weights sum to 1", abs(weights.sum() - 1.0) <= 1e-12,
          f"sum = {weights.sum():.15f}")
    check("tip weights non-negative", np.all(weights >= 0))
    mx = np.dot(weights, nodes[tip_nodes, 0])
    my = np.dot(weights, nodes[tip_nodes, 1])
    check("tip load acts at the face center (no extra moment)",
          abs(mx) <= 1e-12 * Sx and abs(my) <= 1e-12 * Sy,
          f"load centroid x = {mx:.1e}, y = {my:.1e} m")
    check("tip face area", abs(tip_area - Sx * Sy) <= 1e-12 * Sx * Sy,
          f"{tip_area:.12e} m^2")

    # 7. Node sets.
    check("base node count", base.size == (2 * nx + 1) * (2 * ny + 1),
          f"{base.size} nodes at z = 0")
    check("tracking node at tip-face center",
          np.allclose(nodes[track], [0.0, 0.0, -L], atol=1e-15),
          f"node {track} at {nodes[track]}")
    check("mid-plane nodes on y = 0", np.all(nodes[midplane, 1] == 0.0),
          f"{midplane.size} nodes")

    # 8. Reproduce the TL-FEA reader's node-order detection on these
    #    elements: it should report "Standard" (error 0).
    TET_EDGES = [(2, 3), (0, 3), (0, 1), (1, 2), (1, 3), (0, 2)]
    sample = elements[:200]
    err_tet = err_std = 0.0
    for j in range(6):
        m = nodes[sample[:, 4 + j]]
        t0, t1 = TET_EDGES[j]
        s0, s1 = EDGES_STD[j]
        err_tet += np.linalg.norm(
            m - 0.5 * (nodes[sample[:, t0]] + nodes[sample[:, t1]]),
            axis=1).sum()
        err_std += np.linalg.norm(
            m - 0.5 * (nodes[sample[:, s0]] + nodes[sample[:, s1]]),
            axis=1).sum()
    check("TL-FEA reader will detect 'Standard' node order",
          err_tet > err_std,
          f"tetgen error {err_tet:.3e}, standard error {err_std:.3e}")

    all_ok = all(r["ok"] for r in results.values())
    return all_ok, results, mirror


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_node_file(path, nodes):
    with open(path, "w") as f:
        f.write(f"{nodes.shape[0]}  3  0  0\n")
        for i, (x, y, z) in enumerate(nodes, start=1):
            f.write(f"{i} {x:.17g} {y:.17g} {z:.17g}\n")


def write_ele_file(path, elements):
    with open(path, "w") as f:
        f.write(f"{elements.shape[0]}  10  0\n")
        for i, e in enumerate(elements, start=1):
            f.write(f"{i} " + " ".join(str(n + 1) for n in e) + "\n")


def write_index_file(path, header, indices):
    with open(path, "w") as f:
        f.write(f"# {header}\n# count: {len(indices)}\n")
        for n in indices:
            f.write(f"{n}\n")


def write_weights_file(path, tip_nodes, weights):
    with open(path, "w") as f:
        f.write("# Tip-face traction weights (0-based node index, weight).\n"
                "# Uniform traction on the tip face: nodal force = weight * F"
                " for each force component.\n"
                f"# count: {tip_nodes.size}\n")
        for n, w in zip(tip_nodes, weights):
            f.write(f"{n} {w:.17g}\n")


# ---------------------------------------------------------------------------
def main():
    a = parse_args()
    if min(a.length, a.size_x, a.size_y) <= 0:
        sys.exit("error: dimensions must be positive")
    if min(a.nz, a.nx, a.ny) < 1:
        sys.exit("error: --nz, --nx, --ny must be at least 1")
    if a.ny % 2 != 0:
        sys.exit("error: --ny must be even for mirror symmetry about y = 0")

    name = a.name or default_name(a)
    outdir = os.path.join(a.outdir, name)
    os.makedirs(outdir, exist_ok=True)

    print(f"Finger mesh '{name}'")
    print(f"  size  L x Sx x Sy = {a.length} x {a.size_x} x {a.size_y} m")
    print(f"  boxes nz x nx x ny = {a.nz} x {a.nx} x {a.ny}")

    nodes, elements, lattice = build_mesh(a.length, a.size_x, a.size_y,
                                          a.nz, a.nx, a.ny)
    tip_nodes, weights, tip_area = tip_traction_weights(nodes, elements,
                                                        lattice, a.nz)
    base = np.nonzero(lattice[:, 2] == 0)[0]
    midplane = np.nonzero(lattice[:, 1] == a.ny)[0]
    track = int(np.nonzero((lattice[:, 0] == a.nx) &
                           (lattice[:, 1] == a.ny) &
                           (lattice[:, 2] == 2 * a.nz))[0][0])

    n_nodes, n_elem = nodes.shape[0], elements.shape[0]
    n_free = 3 * (n_nodes - base.size)
    print(f"  {n_nodes} nodes, {n_elem} T10 elements, "
          f"{3 * n_nodes} DOFs ({n_free} free after clamping the base)")

    all_ok, results, mirror = run_checks(nodes, elements, lattice, a,
                                         tip_nodes, weights, tip_area,
                                         base, midplane, track)
    if not all_ok:
        sys.exit("\nSome checks failed: no files written.")

    stem = os.path.join(outdir, name)
    write_node_file(stem + ".1.node", nodes)
    write_ele_file(stem + ".1.ele", elements)
    write_index_file(stem + ".base.txt",
                     "Clamped base nodes (z = 0), 0-based indices", base)
    write_weights_file(stem + ".tip_weights.txt", tip_nodes, weights)
    write_index_file(stem + ".track.txt",
                     "Tracking node at tip-face center (0, 0, -L), "
                     "0-based index", [track])
    write_index_file(stem + ".midplane.txt",
                     "Nodes on the symmetry plane y = 0, 0-based indices",
                     midplane)
    np.savez(stem + ".npz", nodes=nodes, elements=elements, base=base,
             tip_nodes=tip_nodes, tip_weights=weights, track=track,
             midplane=midplane, mirror=mirror, lattice=lattice)
    info = {
        "name": name,
        "length_m": a.length, "size_x_m": a.size_x, "size_y_m": a.size_y,
        "nz": a.nz, "nx": a.nx, "ny": a.ny,
        "n_nodes": int(n_nodes), "n_elements": int(n_elem),
        "n_dofs": int(3 * n_nodes), "n_free_dofs": int(n_free),
        "n_base_nodes": int(base.size), "n_tip_nodes": int(tip_nodes.size),
        "track_node": track, "n_midplane_nodes": int(midplane.size),
        "coordinates": "base at z=0 (clamped), tip at z=-L, bending in x-z, "
                       "symmetry plane y=0",
        "element_node_order": "v0 v1 v2 v3 m01 m12 m02 m03 m13 m23 "
                              "(standard)",
        "companion_indices": "0-based",
        "checks": results,
    }
    with open(stem + ".info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nWrote files to {outdir}/")
    for ext in [".1.node", ".1.ele", ".base.txt", ".tip_weights.txt",
                ".track.txt", ".midplane.txt", ".npz", ".info.json"]:
        print(f"  {name}{ext}")


if __name__ == "__main__":
    main()