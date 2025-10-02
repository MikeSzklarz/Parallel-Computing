#!/usr/bin/env python3
"""
test_matrix_vector_mult.py

Usage:
  python3 test_matrix_vector_mult.py starting_n ending_n n_increment

What it does:
  - For each n in [start..end] step increment:
      * builds A: n x n, B: n x 1 with ./make-matrix (once per n)
      * runs ./matrix-vector-multiply A B C repeatedly
        until (trials >= 5) AND (cumulative total_time >= 2.0s)
      * parses the one-line "TIMING ..." report for each trial
      * writes per-trial rows to ./results/trial_results.csv
      * writes per-n summary (mean/stdev) to ./results/summary_results.csv
  - Plots:
      * overall_time_vs_n.png, read_time_vs_n.png, compute_time_vs_n.png, write_time_vs_n.png
      * combined_linear.png, combined_log.png
      * total_gflops_vs_n.png, compute_gflops_vs_n.png, combined_gflops_vs_n.png
  - Shows a live status bar with ETA that adapts using total ≈ a·n² + b.
"""

import sys
import os
import subprocess
import tempfile
import re
import csv
import time
from pathlib import Path
from math import ceil, isfinite
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np

TIMING_RE = re.compile(
    r"TIMING\s+total_s=(?P<total>\d+\.\d+)\s+read_s=(?P<read>\d+\.\d+)\s+compute_s=(?P<compute>\d+\.\d+)\s+write_s=(?P<write>\d+\.\d+)\s+m=(?P<m>\d+)\s+n=(?P<n>\d+)\s+total_gflops=(?P<total_gflops>\d+\.\d+)\s+compute_gflops=(?P<compute_gflops>\d+\.\d+)"
)


MAKE_MATRIX = "./make-matrix"
MATVEC = "./matrix-vector-multiply"

MIN_TRIALS = 5
TARGET_CUM_SEC = 2.0  # run trials until cumulative total >= 2s (and MIN_TRIALS met)

def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)

def check_tools():
    for tool in (MAKE_MATRIX, MATVEC):
        if not os.path.isfile(tool) or not os.access(tool, os.X_OK):
            die(f"Required executable not found or not executable: {tool}")

def run_cmd(cmd, cwd=None):
    result = subprocess.run(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return result.stdout, result.stderr, result.returncode

def parse_timing(s):
    for line in s.splitlines():
        m = TIMING_RE.search(line)
        if m:
            d = m.groupdict()
            return {
                "total": float(d["total"]),
                "read": float(d["read"]),
                "compute": float(d["compute"]),
                "write": float(d["write"]),
                "m": int(d["m"]),
                "n_inside": int(d["n"]),
                "total_gflops": float(d["total_gflops"]),
                "compute_gflops": float(d["compute_gflops"]),
            }
    return None

def ensure_results_dir():
    p = Path("./results")
    p.mkdir(parents=True, exist_ok=True)
    return p

def make_matrix(path, rows, cols, lower=-1.0, upper=1.0):
    cmd = [MAKE_MATRIX, "-rows", str(rows), "-cols", str(cols),
           "-l", str(lower), "-u", str(upper), "-o", str(path)]
    out, err, rc = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"make-matrix failed (rc={rc})\nSTDOUT:\n{out}\nSTDERR:\n{err}")

def matvec(A_path, B_path, C_path):
    cmd = [MATVEC, str(A_path), str(B_path), str(C_path)]
    out, err, rc = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"matrix-vector-multiply failed (rc={rc})\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    t = parse_timing(out) or parse_timing(err)
    if t is None:
        raise RuntimeError(f"Could not parse TIMING line.\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return t

# ---------- ETA model: total_s ~ a*(n^2) + b ----------
def fit_time_n2(ns, totals):
    """Return (a,b) minimizing least-squares for y ≈ a*x + b with x=n^2."""
    if len(ns) == 0:
        return 0.0, 0.0
    if len(ns) == 1:
        x = ns[0] * ns[0]
        a = totals[0] / x if x > 0 else totals[0]
        b = 0.0
        return a, b
    xs = np.array([n*n for n in ns], dtype=float)
    ys = np.array(totals, dtype=float)
    A = np.vstack([xs, np.ones_like(xs)]).T
    # least-squares
    sol, *_ = np.linalg.lstsq(A, ys, rcond=None)
    a, b = sol
    return float(a), float(b)

def predict_time_for_n(a, b, n):
    y = a * (n*n) + b
    if not isfinite(y) or y < 0:
        y = 0.0
    return y

def predict_trials_needed(per_trial_s):
    """Trials required to satisfy both constraints."""
    if per_trial_s <= 0:
        return MIN_TRIALS
    return max(MIN_TRIALS, int(ceil(TARGET_CUM_SEC / per_trial_s)))

def render_progress(done_items, total_items, n, trial_idx, trials_needed, last_total_s, eta_s, bar_width=36):
    frac = (done_items / total_items) if total_items else 1.0
    filled = int(bar_width * frac + 0.5)
    bar = "█" * filled + "·" * (bar_width - filled)
    def fmt_sec(s):
        if s >= 3600:
            return f"{int(s//3600)}h {int((s%3600)//60)}m {int(s%60)}s"
        if s >= 60:
            return f"{int(s//60)}m {int(s%60)}s"
        return f"{s:.1f}s"
    line = (f"[{bar}] {done_items}/{total_items}  "
            f"n={n:<6}  trial={trial_idx}/{trials_needed}  "
            f"last={last_total_s:>7.3f}s  ETA={fmt_sec(eta_s):>8}")
    print("\r" + line, end="", flush=True)

def setup_plot(title, xlabel, ylabel):
    """Set up a standard plot with aesthetic styles."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    return fig, ax

def setup_plot(title, xlabel, ylabel):
    """Set up a standard plot with aesthetic styles for LaTeX reports."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
        'legend.title_fontsize': 14
    })
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax

def finalize_plot(fig, ax, legend_title=None, outpath=None):
    """Finalize a plot with an optional legend and save it."""
    if legend_title:
        ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_series(xs, ys, title, ylabel, filename):
    fig, ax = setup_plot(title, "Matrix size n", ylabel)
    ax.plot(xs, ys, marker="o", linewidth=2, color='royalblue')
    finalize_plot(fig, ax, outpath=filename)

def plot_combined(ns, total_mean, read_mean, compute_mean, write_mean, total_std, filename, ylog=False):
    fig, ax = setup_plot("Timing vs. Matrix Size (n)" + (" [log Y]" if ylog else ""), "Matrix size n", "Time (seconds)")
    
    ax.plot(ns, total_mean, marker="o", linewidth=2, label="Total Time")
    ax.plot(ns, read_mean, marker="o", linewidth=2, label="Read Time")
    ax.plot(ns, compute_mean, marker="o", linewidth=2, label="Multiply Time")
    ax.plot(ns, write_mean, marker="o", linewidth=2, label="Write Time")
    
    total_mean, total_std = np.array(total_mean), np.array(total_std)
    ax.fill_between(ns, total_mean - total_std, total_mean + total_std, alpha=0.2, label="Total Time ±1σ")
    
    if ylog: ax.set_yscale("log")
    
    finalize_plot(fig, ax, legend_title="Time Component", outpath=filename)

def plot_combined_gflops(ns, compute_gflops, total_gflops, filename):
    fig, ax = setup_plot("GFLOPS Performance vs. Matrix Size (n)", "Matrix size n", "GFLOPS")
    ax.plot(ns, compute_gflops, marker="o", linewidth=2, label="Compute GFLOPS")
    ax.plot(ns, total_gflops, marker="o", linewidth=2, label="Total GFLOPS")
    finalize_plot(fig, ax, legend_title="Metric", outpath=filename)

def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)

    try:
        n_start = int(sys.argv[1])
        n_end = int(sys.argv[2])
        n_step = int(sys.argv[3])
    except ValueError:
        die("starting_n, ending_n, n_increment must be integers")
    if n_start <= 0 or n_end <= 0 or n_step <= 0:
        die("All parameters must be positive integers.")
    if n_start > n_end:
        die("starting_n must be <= ending_n")

    check_tools()
    results_dir = ensure_results_dir()
    trial_csv_path = results_dir / "trial_results.csv"
    summary_csv_path = results_dir / "summary_results.csv"

    ns_all = list(range(n_start, n_end + 1, n_step))

    # For ETA:
    measured_ns = []          # per-trial ns (for model)
    measured_totals = []      # per-trial totals (for model)

    # CSV writers
    trial_fields = ["n", "trial_index", "total_s", "read_s", "compute_s", "write_s", "total_gflops", "compute_gflops"]
    summary_fields = ["n", "trials", "total_mean_s", "total_std_s",
                      "read_mean_s", "compute_mean_s", "write_mean_s",
                      "total_gflops_mean", "compute_gflops_mean"]


    # Prepare containers for plots
    ns_summary = []
    total_means = []
    total_stds = []
    read_means = []
    compute_means = []
    write_means = []
    total_gflops_means = []
    compute_gflops_means = []


    t0_wall = time.time()

    # Pre-compute an initial total_items estimate for the progress bar:
    total_items_est = len(ns_all) * MIN_TRIALS
    done_items = 0

    print(f"Running sweep: n from {n_start} to {n_end} by {n_step}")

    with open(trial_csv_path, "w", newline="") as f_trial, \
         open(summary_csv_path, "w", newline="") as f_summary, \
         tempfile.TemporaryDirectory() as tmpdir:

        trial_writer = csv.DictWriter(f_trial, fieldnames=trial_fields)
        summary_writer = csv.DictWriter(f_summary, fieldnames=summary_fields)
        trial_writer.writeheader()
        summary_writer.writeheader()

        tmpdir = Path(tmpdir)

        for n in ns_all:
            A_path = tmpdir / f"A_{n}.bin"
            B_path = tmpdir / f"B_{n}.bin"
            C_path = tmpdir / f"C_{n}.bin"

            make_matrix(A_path, n, n, -1.0, 1.0)
            make_matrix(B_path, n, 1, -1.0, 1.0)

            totals, reads, computes, writes = [], [], [], []
            total_gflops, compute_gflops = [], []
            cumulative_total = 0.0
            trial_idx = 0

            a, b = fit_time_n2(measured_ns, measured_totals)
            per_trial_pred = predict_time_for_n(a, b, n)
            trials_needed = predict_trials_needed(per_trial_pred)

            while (trial_idx < MIN_TRIALS) or (cumulative_total < TARGET_CUM_SEC):
                trial_idx += 1
                t = matvec(A_path, B_path, C_path)

                totals.append(t["total"])
                reads.append(t["read"])
                computes.append(t["compute"])
                writes.append(t["write"])
                total_gflops.append(t["total_gflops"])
                compute_gflops.append(t["compute_gflops"])
                cumulative_total += t["total"]

                trial_writer.writerow({
                    "n": n, "trial_index": trial_idx, "total_s": t["total"],
                    "read_s": t["read"], "compute_s": t["compute"], "write_s": t["write"],
                    "total_gflops": t["total_gflops"], "compute_gflops": t["compute_gflops"],
                })

                measured_ns.append(n)
                measured_totals.append(t["total"])
                a, b = fit_time_n2(measured_ns, measured_totals)
                per_trial_pred = predict_time_for_n(a, b, n)
                trials_needed = max(trial_idx, predict_trials_needed(per_trial_pred))
                
                eta_this_n = max(0, trials_needed - trial_idx) * per_trial_pred
                eta_future = sum(predict_trials_needed(predict_time_for_n(a, b, nf)) * predict_time_for_n(a, b, nf) for nf in ns_all[ns_all.index(n)+1:])
                eta_s = eta_this_n + eta_future

                total_items_est = sum(max(MIN_TRIALS, predict_trials_needed(predict_time_for_n(a, b, nf))) for nf in ns_all)
                done_items += 1
                render_progress(done_items, total_items_est, n, trial_idx, trials_needed, t["total"], eta_s)

            # Summarize results for this n
            ns_summary.append(n)
            total_means.append(mean(totals))
            total_stds.append(pstdev(totals) if len(totals) > 1 else 0.0)
            read_means.append(mean(reads))
            compute_means.append(mean(computes))
            write_means.append(mean(writes))
            total_gflops_means.append(mean(total_gflops))
            compute_gflops_means.append(mean(compute_gflops))

            summary_writer.writerow({
                "n": n, "trials": trial_idx, "total_mean_s": total_means[-1],
                "total_std_s": total_stds[-1], "read_mean_s": read_means[-1],
                "compute_mean_s": compute_means[-1], "write_mean_s": write_means[-1],
                "total_gflops_mean": total_gflops_means[-1],
                "compute_gflops_mean": compute_gflops_means[-1],
            })

    print() # Newline after progress bar

    # Generate plots
    plot_series(ns_summary, total_means, "Overall Elapsed Time vs. Matrix Size (n)", "Time (seconds)", results_dir / "overall_time_vs_n.png")
    plot_series(ns_summary, read_means, "Read Time vs. Matrix Size (n)", "Time (seconds)", results_dir / "read_time_vs_n.png")
    plot_series(ns_summary, compute_means, "Multiply Time vs. Matrix Size (n)", "Time (seconds)", results_dir / "compute_time_vs_n.png")
    plot_series(ns_summary, write_means, "Write Time vs. Matrix Size (n)", "Time (seconds)", results_dir / "write_time_vs_n.png")
    plot_series(ns_summary, total_gflops_means, "Total GFLOPS vs. Matrix Size (n)", "GFLOPS", results_dir / "total_gflops_vs_n.png")
    plot_series(ns_summary, compute_gflops_means, "Compute GFLOPS vs. Matrix Size (n)", "GFLOPS", results_dir / "compute_gflops_vs_n.png")
    
    plot_combined(ns_summary, total_means, read_means, compute_means, write_means, total_stds, results_dir / "combined_linear.png", ylog=False)
    plot_combined(ns_summary, total_means, read_means, compute_means, write_means, total_stds, results_dir / "combined_log.png", ylog=True)
    plot_combined_gflops(ns_summary, compute_gflops_means, total_gflops_means, results_dir / "combined_gflops_vs_n.png")

    elapsed = time.time() - t0_wall
    print("Sweep complete.")
    print(f"- Per-trial CSV: {trial_csv_path}")
    print(f"- Summary CSV:   {summary_csv_path}")
    print(f"- Plots saved in: {results_dir}/")
    for fn in [
        "overall_time_vs_n.png", "read_time_vs_n.png", "compute_time_vs_n.png",
        "write_time_vs_n.png", "combined_linear.png", "combined_log.png",
        "total_gflops_vs_n.png", "compute_gflops_vs_n.png", "combined_gflops_vs_n.png",
    ]:
        print(f"  - {fn}")
    print(f"- Wall time: {elapsed:.3f}s")

if __name__ == "__main__":
    main()