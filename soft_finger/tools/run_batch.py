#!/usr/bin/env python3
"""
run_batch.py
============

Run the FOM driver once per input CSV listed in a manifest
(soft_finger/inputs/csv/manifest.csv), writing each run into its own folder:

    soft_finger/runs/<dataset>/<set>/<name>/   (data files + info.json + log.txt)

Runs that already finished (info.json present) are skipped, so the batch can
be stopped and restarted. Build the driver first:

    bazel build //soft_finger/fom:finger_fom

Usage (from anywhere):
  python3 run_batch.py --dataset dev_n20x2x2 --dry_run    # show commands only
  python3 run_batch.py --dataset dev_n20x2x2              # run everything
  python3 run_batch.py --dataset dev_n20x2x2 --sets test
  python3 run_batch.py --dataset dev_n20x2x2 --no_fint     # skip f_int files
"""

import argparse
import csv
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
DEV_MESH = ("soft_finger/meshes/finger_100x15x15mm_n20x2x2/"
            "finger_100x15x15mm_n20x2x2")
DEV_EQ = ("soft_finger/runs/equilibrium/"
          "finger_100x15x15mm_n20x2x2_eq.txt")


def main():
    ap = argparse.ArgumentParser(description="Run the FOM for every input CSV.")
    ap.add_argument("--dataset", required=True,
                    help="name of this data set (folder under soft_finger/runs)")
    ap.add_argument("--manifest", default="soft_finger/inputs/csv/manifest.csv")
    ap.add_argument("--mesh", default=DEV_MESH)
    ap.add_argument("--init_state", default=DEV_EQ)
    ap.add_argument("--dt", type=float, default=2e-4)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--no_fint", action="store_true",
                    help="do not save internal-force snapshots (saved by default)")
    ap.add_argument("--sets", default="train,test",
                    help="comma-separated sets to run (train, test)")
    ap.add_argument("--binary", default="bazel-bin/soft_finger/fom/finger_fom")
    ap.add_argument("--repo_root", default=DEFAULT_ROOT)
    ap.add_argument("--dry_run", action="store_true",
                    help="print the commands without running them")
    a = ap.parse_args()

    root = a.repo_root
    sets = set(s.strip() for s in a.sets.split(","))
    manifest_path = os.path.join(root, a.manifest)
    csv_dir = os.path.dirname(manifest_path)
    with open(manifest_path, newline="") as f:
        runs = [r for r in csv.DictReader(f) if r["set"] in sets]

    if not a.dry_run and not os.path.isfile(os.path.join(root, a.binary)):
        raise SystemExit(f"driver not found: {a.binary} "
                         "(run: bazel build //soft_finger/fom:finger_fom)")

    print(f"{len(runs)} runs -> soft_finger/runs/{a.dataset}/")
    t_all = time.time()
    for k, r in enumerate(runs, 1):
        out = os.path.join("soft_finger", "runs", a.dataset, r["set"], r["name"])
        csv_path = os.path.relpath(os.path.join(csv_dir, r["csv"]), root)
        cmd = [a.binary, f"--mesh={a.mesh}", f"--init_state={a.init_state}",
               f"--input={csv_path}", f"--dt={a.dt}",
               f"--save_every={a.save_every}", f"--out={out}",
               "--print_every=1000000000"]
        if not a.no_fint:
            cmd.append("--save_fint")

        tag = f"[{k}/{len(runs)}] {r['set']}/{r['name']}"
        if os.path.isfile(os.path.join(root, out, "info.json")):
            print(f"{tag}: already done, skipped")
            continue
        if a.dry_run:
            print(f"{tag}:\n    " + " ".join(cmd))
            continue

        os.makedirs(os.path.join(root, out), exist_ok=True)
        t0 = time.time()
        with open(os.path.join(root, out, "log.txt"), "w") as log:
            ret = subprocess.run(cmd, cwd=root, stdout=log,
                                 stderr=subprocess.STDOUT).returncode
        status = "ok" if ret == 0 else f"FAILED (exit {ret}, see log.txt)"
        print(f"{tag}: {status}  ({time.time() - t0:.1f} s)")

    print(f"total: {time.time() - t_all:.1f} s")


if __name__ == "__main__":
    main()