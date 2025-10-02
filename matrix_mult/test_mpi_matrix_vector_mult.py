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
  - Writes CSV to ./mpi_results/mpi_results.csv with raw times, speedup, efficiency, and GFLOPS.
  - Produces 7 aesthetically-pleasing, comparative plots:
      * combined_times_vs_p.png
      * combined_speedup_vs_p.png
      * combined_efficiency_vs_p.png
      * total_gflops_vs_p.png, compute_gflops_vs_p.png, combined_gflops_vs_p.png

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
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np


MAKE_MATRIX = "./make-matrix"
MPI_MATVEC  = "./mpi-matrix-vector-multiply"
MPIRUN      = os.environ.get("MPIRUN", "mpirun")

TIMING_RE = re.compile(
    r"TIMING\s+total_s=(?P<total>\d+\.\d+)\s+read_s=(?P<read>\d+\.\d+)\s+compute_s=(?P<compute>\d+\.\d+)\s+write_s=(?P<write>\d+\.\d+)\s+m=(?P<m>\d+)\s+n=(?P<n>\d+)\s+p=(?P<p>\d+)\s+total_gflops=(?P<total_gflops>\d+\.\d+)\s+compute_gflops=(?P<compute_gflops>\d+\.\d+)"
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
    _o, _e, rc = run_cmd([MPIRUN, "--version"])
    if rc != 0:
        print("WARNING: 'mpirun --version' failed; ensure MPI is available.", file=sys.stderr)

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
                "total": float(d["total"]), "read": float(d["read"]),
                "compute": float(d["compute"]), "write": float(d["write"]),
                "m": int(d["m"]), "n": int(d["n"]), "p": int(d["p"]),
                "total_gflops": float(d["total_gflops"]),
                "compute_gflops": float(d["compute_gflops"]),
            }
    return None

def make_matrix(path, rows, cols, lower=-1.0, upper=1.0):
    cmd = [MAKE_MATRIX, "-rows", str(rows), "-cols", str(cols), "-l", str(lower), "-u", str(upper), "-o", str(path)]
    out, err, rc = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"make-matrix failed\nCMD: {' '.join(cmd)}\nSTDOUT:\n{out}\nSTDERR:\n{err}")

def run_mpi_matvec(p, A_path, B_path, C_path):
    cmd = [MPIRUN, "-np", str(p), MPI_MATVEC, str(A_path), str(B_path), str(C_path)]
    out, err, rc = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"mpi run failed (p={p})\nCMD: {' '.join(cmd)}\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    timing = parse_timing(out) or parse_timing(err)
    if timing is None:
        raise RuntimeError(f"Could not parse TIMING line for p={p}\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return timing

def calculate_scalability_metrics(rows):
    if not rows: return []
    rows_by_n = {}
    for r in rows:
        rows_by_n.setdefault(r['n'], []).append(r)

    augmented_rows = []
    for n, n_rows in rows_by_n.items():
        baseline = min(n_rows, key=lambda x: x['p'])
        if baseline['p'] != 1:
            print(f"NOTE: For n={n}, p=1 not sampled. Using p={baseline['p']} as baseline.", file=sys.stderr)
        
        for r in n_rows:
            new_row = r.copy()
            for key in ['total', 'compute']:
                time_val = r[f'{key}_s']
                baseline_time = baseline[f'{key}_s']
                speedup = (baseline_time / time_val) if time_val > 0 else 0.0
                efficiency = (speedup / r['p']) if r['p'] > 0 else 0.0
                new_row[f'{key}_speedup'] = speedup
                new_row[f'{key}_efficiency'] = efficiency
            augmented_rows.append(new_row)
    return augmented_rows

# ---------- plotting helpers ----------

def format_n_label(n):
    # This condition for exact millions remains the same
    if n >= 1000000 and n % 1000000 == 0:
        return f"{n // 1000000}M"
    
    # Handle all numbers in the thousands
    if n >= 1000:
        # If n is NOT an exact multiple of 1000 (e.g., 15500)
        if n % 1000 != 0:
            # Format with one decimal place
            return f"{n / 1000:.1f}k"
        # If n IS an exact multiple of 1000 (e.g., 8000)
        else:
            # Format as a whole number
            return f"{n // 1000}k"
            
    # For numbers below 1000
    return str(n)

def per_n_series(rows, key_template):
    by_n = {}
    for r in rows:
        key = key_template.format(**r)
        by_n.setdefault(r['n'], []).append((r['p'], r[key]))
    
    series = {}
    for n, lst in by_n.items():
        lst_sorted = sorted(lst, key=lambda x: x[0])
        series[n] = ([p for p, _ in lst_sorted], [v for _, v in lst_sorted])
    return series

def setup_plot(title, xlabel, ylabel):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11,
        'legend.title_fontsize': 13
    })
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax

def plot_gflops(series_by_n, title, outpath):
    fig, ax = setup_plot(title, "Processes (p)", "GFLOPS")
    colors = plt.cm.viridis(np.linspace(0, 1, len(series_by_n)))
    for i, n in enumerate(sorted(series_by_n.keys())):
        ps, gflops = series_by_n[n]
        ax.plot(ps, gflops, marker="o", linewidth=2, label=format_n_label(n), color=colors[i])
    
    # Use standard finalize_plot for single-metric plots
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Matrix Size (n)", bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_combined_metric(series_dict, title, ylabel, outpath):
    fig, ax = setup_plot(title, "Processes (p)", ylabel)
    # Change 'viridis' to 'Paired'
    colors = plt.cm.tab20(np.linspace(0, 1, len(series_dict['compute'])))

    for i, n in enumerate(sorted(series_dict['compute'].keys())):
        # Both lines for the same 'n' will now share the same color
        color = colors[i]
        
        # Plot "Compute" metric (solid line, 'o' marker)
        ps_c, val_c = series_dict['compute'][n]
        ax.plot(ps_c, val_c, marker='o', linewidth=2, color=color)

        # Plot "Overall" metric (dotted line, 'o' marker)
        if n in series_dict['total']:
            ps_t, val_t = series_dict['total'][n]
            ax.plot(ps_t, val_t, marker='o', linestyle='dotted', linewidth=2, color=color)

    # Custom legend with two columns
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, marker='o', label='Compute'),
        Line2D([0], [0], color='black', lw=2, marker='o', linestyle='dotted', label='Overall')
    ]
    style_legend = ax.legend(handles=legend_elements, title="Metric Type", bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.add_artist(style_legend)

    main_legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=format_n_label(n)) for i, n in enumerate(sorted(series_dict['compute'].keys()))]
    ax.legend(handles=main_legend_elements, title="Matrix Size (n)", bbox_to_anchor=(1.02, 0.65), loc='upper left')

    if "Efficiency" in title:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_ylim(0, 1.1)

    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_combined_gflops(compute_series, total_series, title, outpath):
    fig, ax = setup_plot(title, "Processes (p)", "GFLOPS")
    # Change 'viridis' to 'Paired'
    colors = plt.cm.tab20(np.linspace(0, 1, len(compute_series)))

    for i, n in enumerate(sorted(compute_series.keys())):
        # Both lines for the same 'n' will now share the same color
        color = colors[i]

        # Plot Compute GFLOPS (solid line, 'o' marker)
        ps_c, gflops_c = compute_series[n]
        ax.plot(ps_c, gflops_c, marker="o", lw=2, color=color)

        # Plot Total GFLOPS (dotted line, 'o' marker)
        if n in total_series:
            ps_t, gflops_t = total_series[n]
            ax.plot(ps_t, gflops_t, marker="o", ls="dotted", lw=2, color=color)

    # Custom two-part legend
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, marker='o', label='Compute'),
        Line2D([0], [0], color='black', lw=2, marker='o', linestyle='dotted', label='Overall')
    ]
    style_legend = ax.legend(handles=legend_elements, title="Metric Type", bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.add_artist(style_legend)

    main_legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=format_n_label(n)) for i, n in enumerate(sorted(compute_series.keys()))]
    ax.legend(handles=main_legend_elements, title="Matrix Size (n)", bbox_to_anchor=(1.02, 0.65), loc='upper left')

    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
# ---------- main ----------

def main():
    if len(sys.argv) != 8:
        print(__doc__)
        sys.exit(2)

    try:
        n_start, n_end, n_step = [int(arg) for arg in sys.argv[1:4]]
        p_start, p_end, p_power_limit, p_linear_step = [int(arg) for arg in sys.argv[4:8]]
    except ValueError:
        die("All arguments must be integers.")

    if not (n_start > 0 and n_end >= n_start and n_step >= 0 and p_start > 0 and p_end >= p_start and p_power_limit > 0 and p_linear_step > 0):
        die("Invalid range or step value provided.")
    
    ns = list(range(n_start, n_end + 1, n_step)) if n_step > 0 else [n_start]
    
    # Generate p values
    p_values = set()
    p = p_start
    while p <= p_power_limit and p <= p_end:
        p_values.add(p)
        p *= 2
    
    p = (max(p_values) if p_values else p_start-p_linear_step) + p_linear_step
    while p <= p_end:
        p_values.add(p)
        p += p_linear_step
    ps_sorted = sorted(list(p_values))

    check_tools()
    outdir = ensure_results_dir()
    csv_path = outdir / "mpi_results.csv"
    
    print(f"Sweep: n in {ns}, p in {ps_sorted}")
    rows = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for n in ns:
            A_path, B_path, C_path = tmpdir/f"A_{n}.bin", tmpdir/f"B_{n}.bin", tmpdir/f"C_{n}.bin"
            print(f"\n[ n = {n} ] Building inputs ... ", end="", flush=True)
            make_matrix(A_path, n, n)
            make_matrix(B_path, n, 1)
            print("done.")

            for p in ps_sorted:
                print(f"  p={p:<4} running ... ", end="", flush=True)
                try:
                    timing_data = run_mpi_matvec(p, A_path, B_path, C_path)
                    print(f"total={timing_data['total']:.4f}s")
                    # Rename keys for CSV consistency
                    timing_data['total_s'] = timing_data.pop('total')
                    timing_data['compute_s'] = timing_data.pop('compute')
                    timing_data['read_s'] = timing_data.pop('read')
                    timing_data['write_s'] = timing_data.pop('write')
                    rows.append(timing_data)
                except RuntimeError as e:
                    print(f"FAILED!\n{e}")
    
    if not rows:
        print("No successful runs to analyze.")
        sys.exit(1)

    rows_with_metrics = calculate_scalability_metrics(rows)

    with open(csv_path, "w", newline="") as f:
        fieldnames = list(rows_with_metrics[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_with_metrics)

    # Prepare data for plotting
    time_series = {
        'compute': per_n_series(rows_with_metrics, 'compute_s'),
        'total': per_n_series(rows_with_metrics, 'total_s')
    }
    speedup_series = {
        'compute': per_n_series(rows_with_metrics, 'compute_speedup'),
        'total': per_n_series(rows_with_metrics, 'total_speedup')
    }
    efficiency_series = {
        'compute': per_n_series(rows_with_metrics, 'compute_efficiency'),
        'total': per_n_series(rows_with_metrics, 'total_efficiency')
    }
    
    # Generate combined plots
    plot_combined_metric(time_series, "Execution Time vs. Processes", "Time (seconds)", outdir / "combined_times_vs_p.png")
    plot_combined_metric(speedup_series, "Speedup vs. Processes", "Speedup", outdir / "combined_speedup_vs_p.png")
    plot_combined_metric(efficiency_series, "Efficiency vs. Processes", "Efficiency", outdir / "combined_efficiency_vs_p.png")
    
    # Generate GFLOPS plots
    total_gflops_series = per_n_series(rows_with_metrics, 'total_gflops')
    compute_gflops_series = per_n_series(rows_with_metrics, 'compute_gflops')
    plot_gflops(total_gflops_series, "Total GFLOPS vs. Processes", outdir / "total_gflops_vs_p.png")
    plot_gflops(compute_gflops_series, "Compute GFLOPS vs. Processes", outdir / "compute_gflops_vs_p.png")
    plot_combined_gflops(compute_gflops_series, total_gflops_series, "GFLOPS Performance vs. Processes", outdir / "combined_gflops_vs_p.png")

    print("\nDone.")
    print(f"- Results CSV: {csv_path}")
    print(f"- Plots written to: {outdir}/")
    for fn in sorted(os.listdir(outdir)):
        if fn.endswith('.png'):
            print(f"  - {fn}")

if __name__ == "__main__":
    main()