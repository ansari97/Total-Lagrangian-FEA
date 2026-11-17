#!/usr/bin/env python3
"""
check_run.py
============

Consistency checks for one FOM run folder written by finger_fom --out=...

  1. Discrete equation of motion on the free rows:
         M a_n + fint_n - fbdy - B u_n = 0      (to Newton's tolerance)
  2. Viscous power v_n . (fint_n - fint_e_n) >= 0 (Kelvin-Voigt dissipates)
  3. Clamped nodes stay at their reference positions.
  4. Tracking-node trajectory summary.

Usage:
  python3 check_run.py --run ../runs/test_ramp \
      --mesh ../meshes/finger_100x15x15mm_n20x2x2/finger_100x15x15mm_n20x2x2
"""

import argparse
import json
import os

import numpy as np


def read_indices(path):
    return np.loadtxt(path, comments="#", dtype=int, ndmin=1)


def load_mass(path, n_nodes):
    coo = np.loadtxt(path, comments="#")
    m = np.zeros((n_nodes, n_nodes))
    np.add.at(m, (coo[:, 0].astype(int), coo[:, 1].astype(int)), coo[:, 2])
    return m


def mass_times(m, a):
    """(m kron I3) a for each row of a (rows = snapshots, layout x0,y0,z0,...)."""
    ns, nd = a.shape
    a3 = a.reshape(ns, nd // 3, 3)                 # (ns, n_nodes, 3)
    return np.einsum("ij,sjc->sic", m, a3).reshape(ns, nd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder (finger_fom --out)")
    ap.add_argument("--mesh", required=True, help="mesh path without extension")
    a = ap.parse_args()

    info = json.load(open(os.path.join(a.run, "info.json")))
    load = lambda name: np.load(os.path.join(a.run, name))
    t, e, v, acc, u = (load(f) for f in ["t.npy", "e.npy", "v.npy", "a.npy", "u.npy"])
    fbdy, B, e_ref = load("fbdy.npy"), load("B.npy"), load("e_ref.npy")
    n_nodes = info["n_nodes"]
    print(f"run {a.run}: {len(t)} snapshots, t = {t[0]:.4g} ... {t[-1]:.4g} s, "
          f"dt = {info['dt']}, save_every = {info['save_every']}")

    base = read_indices(a.mesh + ".base.txt")
    track = int(read_indices(a.mesh + ".track.txt")[0])
    clamped_dofs = np.concatenate([3 * base + c for c in range(3)])
    free = np.setdiff1d(np.arange(3 * n_nodes), clamped_dofs)

    # 1. Discrete equation of motion (needs the internal-force snapshots).
    if info["save_fint"]:
        fint, fint_e = load("fint.npy"), load("fint_e.npy")
        m = load_mass(os.path.join(a.run, "mass_coo.txt"), n_nodes)
        f_ext = fbdy[None, :] + u @ B.T                     # (ns, 3n)
        res = (mass_times(m, acc) + fint - f_ext)[:, free]
        scale = np.abs(f_ext[:, free]).max()
        print(f"1. EOM residual (free rows): max |M a + fint - fext| = "
              f"{np.abs(res).max():.3e} N  (force scale {scale:.3e} N)")

        # 2. Viscous power: must be >= 0 (up to rounding).
        f_visc = fint - fint_e
        power = np.einsum("sk,sk->s", v[:, free], f_visc[:, free])
        print(f"2. viscous power v.f_v: min {power.min():.3e} W, "
              f"max {power.max():.3e} W "
              f"({'OK, dissipative' if power.min() > -1e-12 * max(power.max(), 1e-300) else 'NEGATIVE values!'})")
    else:
        print("1-2. skipped (run without --save_fint)")

    # 3. Clamped nodes stay put.
    drift = np.abs(e[:, clamped_dofs] - e_ref[clamped_dofs]).max()
    print(f"3. clamped nodes: max |e - e_ref| = {drift:.3e} m")

    # 4. Tracking node.
    d = e[:, 3 * track:3 * track + 3] - e_ref[3 * track:3 * track + 3]
    print(f"4. tracking node {track}: final disp = ({d[-1, 0]:.4e}, "
          f"{d[-1, 1]:.4e}, {d[-1, 2]:.4e}) m; max |x| = {np.abs(d[:, 0]).max():.4e} m")


if __name__ == "__main__":
    main()