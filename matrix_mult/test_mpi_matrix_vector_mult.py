#!/usr/bin/env python3
"""
test_mpi_matrix_vector_mult.py

Usage:
  python3 test_mpi_matrix_vector_mult.py <n_start> <n_end> <n_step> <p_start> <p_end> <p_power_limit> <p_linear_step>

What it does:
  - For each n in [start..end] step n_increment:
      * makes A: n x n, B: n x 1 once via ./make-matrix
      * for each p in a generated list (see below):
          - runs: mpirun -np p ./mpi-matrix-vector-multiply A.bin B.bin C.bin
          - parses: TIMING total_s=... read_s=... compute_s=... write_s=... m=... n=... p=...
  - Process counts (p) are generated in two phases:
      1. Powers of 2, starting from <p_start> (must be 1 or power of 2) up to <p_power_limit>.
      2. Linearly, adding <p_linear_step> to the last generated p, up to <p_end>.
  - Writes CSV to ./mpi_results/mpi_results.csv with raw times plus calculated speedup and efficiency.
  - Produces 6 combined plots (one curve per n):
      * overall_times_vs_p.png, overall_speedup_vs_p.png, overall_efficiency_vs_p.png
      * compute_times_vs_p.png, compute_speedup_vs_p.png, compute_efficiency_vs_p.png

Example:
  # Test n=1024 with p = 1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128
  python3 test_mpi_matrix_vector_mult.py 1024 1024 0 1 128 32 16
"""

import sys
import os
import subprocess
import tempfile
import re
import csv
from pathlib import Path
from math import isfinite

import matplotlib.pyplot as plt

MAKE_MATRIX = "./make-matrix"
MPI_MATVEC  = "./mpi-matrix-vector-multiply"
MPIRUN      = os.environ.get("MPIRUN", "mpirun")

TIMING_RE = re.compile(
    r"TIMING\s+total_s=(?P<total>\d+\.\d+)\s+read_s=(?P<read>\d+\.\d+)\s+compute_s=(?P<compute>\d+\.\d+)\s+write_s=(?P<write>\d+\.\d+)\s+m=(?P<m>\d+)\s+n=(?P<n>\d+)\s+p=(?P<p>\d+)"
)

def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)

def run_cmd(cmd, cwd=None):
    res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.stdout, res.stderr, res.returncode

def check_tools():
    for exe in (MAKE_MATRIX, MPI_MATVEC):
        if not (os.path.isfile(exe) and os.access(exe, os.X_OK)):
            die(f"Required executable not found or not executable: {exe}")
    # Best-effort mpirun check
    _o, _e, rc = run_cmd([MPIRUN, "--version"])
    if rc != 0:
        print("WARNING: 'mpirun --version' failed; ensure MPI is available (set MPIRUN env if needed).", file=sys.stderr)

def ensure_results_dir():
    d = Path("./mpi_results")
    d.mkdir(parents=True, exist_ok=True)
    return d

def parse_timing(text_out):
    for line in text_out.splitlines():
        m = TIMING_RE.search(line)
        if m:
            d = m.groupdict()
            return {
                "total": float(d["total"]),
                "read": float(d["read"]),
                "compute": float(d["compute"]),
                "write": float(d["write"]),
                "m": int(d["m"]),
                "n": int(d["n"]),
                "p": int(d["p"]),
            }
    return None

def make_matrix(path, rows, cols, lower=-1.0, upper=1.0):
    cmd = [MAKE_MATRIX, "-rows", str(rows), "-cols", str(cols),
           "-l", str(lower), "-u", str(upper), "-o", str(path)]
    out, err, rc = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"make-matrix failed\nCMD: {' '.join(cmd)}\nSTDOUT:\n{out}\nSTDERR:\n{err}")

def run_mpi_matvec(p, A_path, B_path, C_path):
    cmd = [MPIRUN, "-np", str(p), MPI_MATVEC, str(A_path), str(B_path), str(C_path)]
    out, err, rc = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"mpi-matrix-vector-multiply failed (p={p})\nCMD: {' '.join(cmd)}\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    timing = parse_timing(out) or parse_timing(err)
    if timing is None:
        raise RuntimeError(f"Could not parse TIMING line for p={p}\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return timing

def calculate_scalability_metrics(rows):
    """
    Augments rows with speedup and efficiency metrics.
    Calculations are done per-n, using the run with the smallest p as the baseline.
    """
    if not rows:
        return []

    # 1. Group rows by n and find baseline for each n (minimum p)
    baselines = {}
    rows_by_n = {}
    for r in rows:
        n = r['n']
        rows_by_n.setdefault(n, []).append(r)

    for n, n_rows in rows_by_n.items():
        baseline_row = min(n_rows, key=lambda x: x['p'])
        baselines[n] = {
            'p': baseline_row['p'],
            'total_s': baseline_row['total_s'],
            'compute_s': baseline_row['compute_s'],
        }
        if baseline_row['p'] != 1:
            print(f"NOTE: For n={n}, p=1 not sampled. Using p={baseline_row['p']} as baseline for CSV speedup/efficiency.", file=sys.stderr)

    # 2. Calculate metrics for each row and return new list
    augmented_rows = []
    for r in rows:
        n, p = r['n'], r['p']
        baseline = baselines[n]
        
        # Avoid division by zero
        total_s = r['total_s']
        compute_s = r['compute_s']

        total_speedup = (baseline['total_s'] / total_s) if total_s > 0 else 0.0
        total_efficiency = (total_speedup / p) if p > 0 else 0.0
        compute_speedup = (baseline['compute_s'] / compute_s) if compute_s > 0 else 0.0
        compute_efficiency = (compute_speedup / p) if p > 0 else 0.0
        
        new_row = r.copy()
        new_row.update({
            'total_speedup': total_speedup,
            'total_efficiency': total_efficiency,
            'compute_speedup': compute_speedup,
            'compute_efficiency': compute_efficiency,
        })
        augmented_rows.append(new_row)
        
    return augmented_rows

# ---------- plotting helpers ----------

def per_n_series(rows, key_time):
    """
    Build per-n series: {n: (ps_sorted, times_sorted)} using key_time ('total' or 'compute').
    rows: list of dicts with keys n, p, total, read, compute, write
    """
    by_n = {}
    for r in rows:
        n = r["n"]; p = r["p"]; t = r[key_time]
        by_n.setdefault(n, []).append((p, t))
    series = {}
    for n, lst in by_n.items():
        lst_sorted = sorted(lst, key=lambda x: x[0])
        ps = [p for p,_ in lst_sorted]
        ts = [t for _,t in lst_sorted]
        series[n] = (ps, ts)
    return series

def plot_multi_times(series_by_n, title, ylabel, outpath):
    plt.figure()
    for n in sorted(series_by_n.keys()):
        ps, ts = series_by_n[n]
        plt.plot(ps, ts, marker="o", linewidth=2, label=f"n={n}")
    plt.title(title)
    plt.xlabel("Processes (p)")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="Matrix size", ncol=2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def build_speedup_efficiency(ps, ts, n, label_for_warning):
    """Given vectors p, time, compute speedup & efficiency with per-n baseline."""
    if not ps:
        return [], [], 1
    if 1 in ps:
        base_idx = ps.index(1)
        base_p = 1
    else:
        base_idx = 0
        base_p = ps[0]
        print(f"NOTE: p=1 not sampled for n={n} ({label_for_warning}); using p={base_p} as baseline for plot.", file=sys.stderr)
    base_t = ts[base_idx]
    speedup = []
    efficiency = []
    for p, t in zip(ps, ts):
        s = (base_t / t) if t > 0 and isfinite(t) else 0.0
        e = (s / p) if p > 0 else 0.0
        speedup.append(s)
        efficiency.append(e)
    return speedup, efficiency, base_p

def plot_multi_speedup(series_by_n, title, outpath, label_for_warning):
    plt.figure()
    for n in sorted(series_by_n.keys()):
        ps, ts = series_by_n[n]
        s, _e, base_p = build_speedup_efficiency(ps, ts, n, label_for_warning)
        plt.plot(ps, s, marker="o", linewidth=2, label=f"n={n} (base p={base_p})")
    plt.title(title)
    plt.xlabel("Processes (p)")
    plt.ylabel("Speedup")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="Matrix size", ncol=2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_multi_efficiency(series_by_n, title, outpath, label_for_warning):
    plt.figure()
    for n in sorted(series_by_n.keys()):
        ps, ts = series_by_n[n]
        _s, e, base_p = build_speedup_efficiency(ps, ts, n, label_for_warning)
        plt.plot(ps, e, marker="o", linewidth=2, label=f"n={n} (base p={base_p})")
    plt.title(title)
    plt.xlabel("Processes (p)")
    plt.ylabel("Efficiency (Speedup / p)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="Matrix size", ncol=2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# ---------- main ----------

def main():
    if len(sys.argv) != 8: # prog, n_start, n_end, n_step, p_start, p_end, p_power_limit, p_linear_step
        print(__doc__)
        sys.exit(2)

    try:
        n_start     = int(sys.argv[1])
        n_end       = int(sys.argv[2])
        n_step      = int(sys.argv[3])
    except ValueError:
        die("n_start, n_end, n_step must be integers.")

    if n_start <= 0 or n_end <= 0 or (n_step < 0):
        die("n_start, n_end must be positive; n_step must be non-negative.")
    if n_step == 0 and n_start != n_end:
        n_step = n_end - n_start # Allows for a single n run, e.g., 1024 1024 0
    if n_start > n_end: die("starting_n must be <= ending_n")
    
    ns = list(range(n_start, n_end + 1, n_step)) if n_step > 0 else [n_start]

    # --- New Argument Parsing for p ---
    try:
        p_start       = int(sys.argv[4])
        p_end         = int(sys.argv[5])
        p_power_limit = int(sys.argv[6])
        p_linear_step = int(sys.argv[7])
    except ValueError:
        die("p_start, p_end, p_power_limit, and p_linear_step must be integers.")
    
    if not (p_start > 0 and p_end > 0 and p_power_limit > 0 and p_linear_step > 0):
        die("All p arguments (start, end, power_limit, linear_step) must be positive.")
    if p_start > p_end: die("p_start must be <= p_end")
    if p_power_limit > p_end:
        print(f"NOTE: p_power_limit ({p_power_limit}) > p_end ({p_end}). Power-of-2 phase will not exceed {p_end}.", file=sys.stderr)
        p_power_limit = p_end
    if p_start > p_power_limit: die("p_start must be <= p_power_limit")
    # Check if p_start is 1 or a power of 2
    if (p_start & (p_start - 1) != 0) and p_start != 1:
            die(f"p_start ({p_start}) must be 1 or a power of 2 for this mode.")

    p_values = []
    current_p = p_start
    last_p_generated = 0

    # Phase 1: Powers of 2
    while current_p <= p_power_limit:
        p_values.append(current_p)
        last_p_generated = current_p
        if current_p == 0: break # Should be blocked by > 0 check
        current_p *= 2
        
    # Phase 2: Linear range
    # Start from the *next* step after the last generated value
    current_p = last_p_generated + p_linear_step
    while current_p <= p_end:
        p_values.append(current_p)
        current_p += p_linear_step
    
    ps_sorted = sorted(list(set(p_values))) # Use set to remove potential duplicates
    # --- End New Argument Parsing ---


    check_tools()
    outdir = ensure_results_dir()
    csv_path = outdir / "mpi_results.csv"
    
    print(f"Sweep: n in {ns}, p in {ps_sorted}")

    # Gather all rows
    rows = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for n in ns:
            A_path = tmpdir / f"A_{n}.bin"
            B_path = tmpdir / f"B_{n}.bin"
            C_path = tmpdir / f"C_{n}.bin"

            print(f"\n[ n = {n} ] Building inputs ... ", end="", flush=True)
            make_matrix(A_path, n, n, -1.0, 1.0)
            make_matrix(B_path, n, 1, -1.0, 1.0)
            print("done.")

            for p in ps_sorted:
                print(f"  p={p:<4} running ... ", end="", flush=True)
                try:
                    t = run_mpi_matvec(p, A_path, B_path, C_path)
                    print(f"total={t['total']:.4f}s")
                    rows.append({
                        "n": t["n"], "p": t["p"],
                        "total_s": t["total"], "read_s": t["read"],
                        "compute_s": t["compute"], "write_s": t["write"],
                    })
                except RuntimeError as e:
                    print(f"FAILED!\n{e}")

    # Add speedup/efficiency calculations before writing
    rows_with_metrics = calculate_scalability_metrics(rows)

    # Write CSV
    if rows_with_metrics:
        with open(csv_path, "w", newline="") as f:
            fieldnames = [
                "n", "p", "total_s", "read_s", "compute_s", "write_s",
                "total_speedup", "total_efficiency",
                "compute_speedup", "compute_efficiency"
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_with_metrics)
    else:
        print("No successful runs to write to CSV.")
        sys.exit(1)

    # Build per-n series for overall and compute-only
    rows_for_plots = [{"n": r["n"], "p": r["p"], "total": r["total_s"], "compute": r["compute_s"]} for r in rows]
    overall_series = per_n_series(rows_for_plots, key_time="total")
    compute_series = per_n_series(rows_for_plots, key_time="compute")

    # Produce 6 combined plots (one curve per n per plot)
    plot_multi_times(
        overall_series,
        title="Overall time vs processes (p) — one curve per n",
        ylabel="Time (seconds)",
        outpath=outdir / "overall_times_vs_p.png",
    )
    plot_multi_speedup(
        overall_series,
        title="Overall speedup vs processes (p) — one curve per n",
        outpath=outdir / "overall_speedup_vs_p.png",
        label_for_warning="total",
    )
    plot_multi_efficiency(
        overall_series,
        title="Overall efficiency vs processes (p) — one curve per n",
        outpath=outdir / "overall_efficiency_vs_p.png",
        label_for_warning="total",
    )
    plot_multi_times(
        compute_series,
        title="Compute-only time vs processes (p) — one curve per n",
        ylabel="Time (seconds)",
        outpath=outdir / "compute_times_vs_p.png",
    )
    plot_multi_speedup(
        compute_series,
        title="Compute-only speedup vs processes (p) — one curve per n",
        outpath=outdir / "compute_speedup_vs_p.png",
        label_for_warning="compute",
    )
    plot_multi_efficiency(
        compute_series,
        title="Compute-only efficiency vs processes (p) — one curve per n",
        outpath=outdir / "compute_efficiency_vs_p.png",
        label_for_warning="compute",
    )

    print("\nDone.")
    print(f"- Results CSV: {csv_path}")
    print(f"- Plots written to: {outdir}/")
    for fn in [
        "overall_times_vs_p.png", "overall_speedup_vs_p.png", "overall_efficiency_vs_p.png",
        "compute_times_vs_p.png", "compute_speedup_vs_p.png", "compute_efficiency_vs_p.png",
    ]:
        print(f"  - {fn}")

if __name__ == "__main__":
    main()