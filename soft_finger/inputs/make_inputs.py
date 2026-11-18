#!/usr/bin/env python3
"""
make_inputs.py
==============

Generate tip-force input trajectories (t, Fx, Fz) for the FOM driver as CSV
files, plus a manifest listing every file and whether it is for training or
testing.

Sign convention: +z points up, toward the clamped base. Fz < 0 pulls the tip
down (tension, harmless); Fz > 0 pushes it up (compression, kept small).

Families
--------
  steps   : ramps between random force levels, holds, final free decay
  chirp_x : swept sine in Fx (f0 -> f1), Fz = 0
  chirp_z : swept sine in Fz around a tension bias, Fx = 0
  random  : smooth random signal (spline through random knots), Fx and Fz
  aprbs   : random levels held for random durations, smooth transitions
  task    : smooth periodic pattern (test set only)

Every signal starts at exactly zero (the runs start from the gravity
equilibrium) and stays inside the force limits.

Usage:
  python3 make_inputs.py                 # default suite (train + test)
  python3 make_inputs.py --duration 2    # shorter runs
"""

import argparse
import csv
import os

import numpy as np


# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------
def smoothstep(s):
    """0 for s <= 0, 1 for s >= 1, smooth (zero slope at both ends) between."""
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def ramp_in(t, t_ramp):
    """Multiplier that grows smoothly from 0 to 1 over t_ramp seconds."""
    return smoothstep(t / t_ramp)


def piecewise_levels(t, times, levels, transition):
    """Hold levels[k] from times[k]; smooth transitions of `transition` s."""
    f = np.full_like(t, levels[0])
    for k in range(1, len(levels)):
        f += (levels[k] - levels[k - 1]) * smoothstep((t - times[k]) / transition)
    return f


def catmull_rom(t, knot_dt, knots):
    """Smooth curve through knot values placed every knot_dt seconds."""
    k = np.concatenate([[knots[0]], knots, [knots[-1], knots[-1]]])
    s = t / knot_dt
    i = np.minimum(np.floor(s).astype(int), len(knots) - 2)
    u = s - i
    p0, p1, p2, p3 = k[i], k[i + 1], k[i + 2], k[i + 3]
    return 0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u**2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * u**3)


# ---------------------------------------------------------------------------
# Families. Each returns (Fx, Fz, metadata dict).
# ---------------------------------------------------------------------------
def make_steps(t, rng, lim):
    T = t[-1]
    times, lx, lz = [0.0], [0.0], [0.0]
    tc = 0.2
    while tc < T - 1.2:                       # leave >= 1 s for free decay
        times.append(tc)
        lx.append(rng.uniform(-lim["fx"], lim["fx"]))
        lz.append(rng.uniform(lim["fz_min"], lim["fz_max"]))
        tc += rng.uniform(0.5, 1.0)
    times.append(T - 1.0)                     # back to zero: free decay
    lx.append(0.0)
    lz.append(0.0)
    tr = rng.uniform(0.05, 0.2)
    fx = piecewise_levels(t, times, lx, tr)
    fz = piecewise_levels(t, times, lz, tr)
    return fx, fz, {"transition_s": round(tr, 4), "n_levels": len(times) - 2}


def chirp(t, f0, f1):
    """Linear sweep from f0 to f1 Hz over the full duration."""
    T = t[-1]
    return np.sin(2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t**2 / T))


def make_chirp_x(t, rng, lim, amp=0.03, f0=0.5, f1=25.0):
    fx = amp * chirp(t, f0, f1) * ramp_in(t, 0.2)
    return fx, np.zeros_like(t), {"amp_N": amp, "f0_Hz": f0, "f1_Hz": f1}


def make_chirp_z(t, rng, lim, amp=0.05, f0=0.5, f1=25.0):
    bias = -amp                                # oscillate in [-2 amp, 0]: tension
    fz = (bias + amp * chirp(t, f0, f1)) * ramp_in(t, 0.2)
    return np.zeros_like(t), fz, {"amp_N": amp, "bias_N": bias,
                                   "f0_Hz": f0, "f1_Hz": f1}


def make_random(t, rng, lim):
    knot_dt = rng.uniform(0.05, 0.25)
    scale = min(1.0, knot_dt / 0.25)           # faster signals -> smaller amplitude
    n = int(np.ceil(t[-1] / knot_dt)) + 2
    kx = rng.uniform(-1, 1, n) * lim["fx"] * scale
    kz = rng.uniform(lim["fz_min"], lim["fz_max"], n) * scale
    w = ramp_in(t, 0.2)
    fx = catmull_rom(t, knot_dt, kx) * w
    fz = catmull_rom(t, knot_dt, kz) * w
    return fx, fz, {"knot_dt_s": round(knot_dt, 4), "amp_scale": round(scale, 3)}


def make_aprbs(t, rng, lim):
    def levels(lo, hi):
        times, vals, tc = [0.0], [0.0], rng.uniform(0.05, 0.2)
        while tc < t[-1]:
            times.append(tc)
            vals.append(rng.uniform(lo, hi))
            tc += rng.uniform(0.1, 0.6)
        return times, vals
    tr = 0.03
    tx, vx = levels(-lim["fx"], lim["fx"])
    tz, vz = levels(lim["fz_min"], lim["fz_max"])
    fx = piecewise_levels(t, tx, vx, tr)
    fz = piecewise_levels(t, tz, vz, tr)
    return fx, fz, {"hold_s": "0.1-0.6", "transition_s": tr}


def make_task(t, rng, lim, freq=0.5, ax=0.2, bz=-0.08, cz=0.06):
    w = ramp_in(t, 0.5)
    fx = ax * np.sin(2 * np.pi * freq * t) * w
    fz = (bz + cz * np.cos(2 * np.pi * freq * t)) * w
    return fx, fz, {"freq_Hz": freq, "ax_N": ax, "bz_N": bz, "cz_N": cz}


FAMILIES = {"steps": make_steps, "chirp_x": make_chirp_x, "chirp_z": make_chirp_z,
            "random": make_random, "aprbs": make_aprbs, "task": make_task}

# The default suite: (name, set, family, seed, extra keyword arguments).
SUITE = [
    ("steps_01",   "train", "steps",   101, {}),
    ("steps_02",   "train", "steps",   102, {}),
    ("chirp_x_01", "train", "chirp_x", 0,   {}),
    ("chirp_z_01", "train", "chirp_z", 0,   {}),
    ("random_01",  "train", "random",  201, {}),
    ("random_02",  "train", "random",  202, {}),
    ("random_03",  "train", "random",  203, {}),
    ("random_04",  "train", "random",  204, {}),
    ("aprbs_01",   "train", "aprbs",   301, {}),
    ("aprbs_02",   "train", "aprbs",   302, {}),
    ("aprbs_03",   "train", "aprbs",   303, {}),
    ("aprbs_04",   "train", "aprbs",   304, {}),
    ("random_t01", "test",  "random",  901, {}),
    ("aprbs_t01",  "test",  "aprbs",   902, {}),
    ("steps_t01",  "test",  "steps",   903, {}),
    ("task_t01",   "test",  "task",    0,   {"freq": 0.5, "ax": 0.2}),
    ("task_t02",   "test",  "task",    0,   {"freq": 1.5, "ax": 0.12}),
]


# ---------------------------------------------------------------------------
def write_csv(path, t, fx, fz, header):
    with open(path, "w", newline="") as f:
        for key, val in header.items():
            f.write(f"# {key}: {val}\n")
        f.write("t,Fx,Fz\n")
        for row in zip(t, fx, fz):
            f.write(f"{row[0]:.6f},{row[1]:.9g},{row[2]:.9g}\n")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Generate FOM input CSVs.")
    ap.add_argument("--out_dir", default=os.path.join(here, "csv"),
                    help="output folder (default: ./csv next to this script)")
    ap.add_argument("--duration", type=float, default=4.0, help="seconds")
    ap.add_argument("--sample", type=float, default=2e-4,
                    help="CSV sampling interval (s); keep <= the FOM dt")
    ap.add_argument("--fx_max", type=float, default=0.3, help="|Fx| limit (N)")
    ap.add_argument("--fz_min", type=float, default=-0.2,
                    help="most negative Fz (tension, N)")
    ap.add_argument("--fz_max", type=float, default=0.05,
                    help="most positive Fz (compression, N)")
    a = ap.parse_args()

    lim = {"fx": a.fx_max, "fz_min": a.fz_min, "fz_max": a.fz_max}
    os.makedirs(a.out_dir, exist_ok=True)
    n = int(round(a.duration / a.sample))
    t = np.arange(n + 1) * a.sample

    manifest = []
    print(f"{'name':12s} {'set':5s} {'family':8s}  {'Fx range (N)':>20s}  "
          f"{'Fz range (N)':>20s}")
    for name, split, family, seed, kw in SUITE:
        rng = np.random.default_rng(seed)
        fx, fz, meta = FAMILIES[family](t, rng, lim, **kw)
        fx = np.clip(fx, -lim["fx"], lim["fx"])
        fz = np.clip(fz, lim["fz_min"], lim["fz_max"])
        fx[0] = fz[0] = 0.0                      # start exactly from rest
        header = {"name": name, "set": split, "family": family, "seed": seed,
                  "duration_s": a.duration, "sample_s": a.sample,
                  "fx_max_N": a.fx_max, "fz_min_N": a.fz_min,
                  "fz_max_N": a.fz_max, **meta}
        path = os.path.join(a.out_dir, name + ".csv")
        write_csv(path, t, fx, fz, header)
        manifest.append({"name": name, "set": split, "family": family,
                         "seed": seed, "duration_s": a.duration,
                         "csv": os.path.relpath(path, a.out_dir)})
        print(f"{name:12s} {split:5s} {family:8s}  "
              f"[{fx.min():+.3f}, {fx.max():+.3f}]  [{fz.min():+.3f}, {fz.max():+.3f}]")

    with open(os.path.join(a.out_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(manifest)
    print(f"\nwrote {len(manifest)} CSV files and manifest.csv to {a.out_dir}")


if __name__ == "__main__":
    main()