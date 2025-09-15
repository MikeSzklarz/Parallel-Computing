#!/usr/bin/env python3
"""
mpi_trap_sweep.py

Run an MPI trapezoid program over a parameter sweep of n (trapezoids)
and p (number of MPI processes), collect timing, and produce:
  1) A neatly aligned table to stdout,
  2) A CSV file of all results,
  3) Plots: timing.png, speedup.png, efficiency.png,
  4) A live ASCII progress bar while the sweep is running.

EXPECTED MPI PROGRAM OUTPUT (rank 0 only, one line):
  "<answer> <a> <b> <n> <np> <total_time_seconds>"

Example:
  2.333333334999992e+00 1 2 10000 2 0.010487

USAGE
-----
  python mpi_trap_sweep.py a b n1 n2 n_increment p1 p2
    [--p-increment INCREMENT]
    [--p-extra-increment THRESHOLD INCREMENT]
    [--exe ./mpi_trap_modified]
    [--mpirun mpirun]
    [--csv mpi_trap_sweep.csv]
    [--extra-mpirun-args "-bind-to none"]
    [--timing-png timing.png]
    [--speedup-png speedup.png]
    [--efficiency-png efficiency.png]

Positional args:
  a, b                 : Integration range (floats)
  n1, n2, n_increment  : Sweep for n (ints; inclusive; increment > 0)
  p1, p2               : Start and end bounds for the number of processes (p).

Optional args:
  --p-increment        : Use a linear increment for p. If omitted, p is
                         generated using powers of 2 by default.
  --p-extra-increment  : (Power-of-2 mode only) Adds extra p values. Above
                         THRESHOLD, it adds p values every INCREMENT up to p2.

Outputs:
  - CSV columns: n,p,a,b,answer,time_seconds,speedup,efficiency
  - timing.png:      Y=time(s), X=#processes, one curve per n
  - speedup.png:     Y=speedup(p), X=#processes, one curve per n (requires p=1 baseline)
  - efficiency.png:  Y=efficiency(p), X=#processes, one curve per n (requires p=1 baseline)
"""

import argparse
import csv
import shlex
import subprocess
import sys
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


# ----------------------- Utilities -----------------------

def inc_range(start: int, stop: int, step: int) -> List[int]:
    """Inclusive integer range with positive step."""
    if step <= 0:
        raise ValueError("Increment must be > 0")
    vals = []
    x = start
    while x <= stop:
        vals.append(x)
        x += step
    return vals

def generate_p_values(p_start: int, p_end: int, extra_params: Optional[List[int]] = None) -> List[int]:
    """
    Generates a list of process counts (p) for the sweep.
    Includes powers of 2 within the [p_start, p_end] range.
    Optionally adds extra values at a fixed increment above a certain threshold.
    """
    if p_start < 1 or p_end < p_start:
        raise ValueError("p range must be valid (p1 >= 1, p2 >= p1)")

    # Generate base power-of-2 values
    p_vals = set()
    p = 1
    while p <= p_end:
        if p >= p_start:
            p_vals.add(p)
        # Handle case where p is already max int to avoid overflow
        if p > p_end / 2:
            break
        p *= 2

    # Add extra incremental values if specified
    if extra_params:
        threshold, increment = extra_params
        if threshold < 1 or increment < 1:
            raise ValueError("Extra increment threshold and value must be >= 1")
        
        current_p = threshold
        while current_p <= p_end:
            if current_p >= p_start:
                p_vals.add(current_p)
            # Avoid overflow
            if current_p > p_end - increment:
                break
            current_p += increment

    return sorted(list(p_vals))


def parse_program_output(line: str) -> Tuple[float, float, float, int, int, float]:
    """
    Parse: "<answer> <a> <b> <n> <np> <total_time_seconds>"
    Returns: (answer, a, b, n, np, time_sec)
    Raises ValueError on mismatch.
    """
    parts = line.strip().split()
    if len(parts) != 6:
        raise ValueError(f"Expected 6 fields, got {len(parts)}: {line!r}")
    try:
        answer  = float(parts[0])
        a_val   = float(parts[1])
        b_val   = float(parts[2])
        n_val   = int(parts[3])
        np_val  = int(parts[4])
        t_sec   = float(parts[5])
    except Exception as e:
        raise ValueError(f"Failed to parse fields from line {line!r}: {e}")
    return answer, a_val, b_val, n_val, np_val, t_sec


def run_once(mpirun: str, extra_args: str, exe: str, a: float, b: float, n: int, p: int
             ) -> Optional[Tuple[float, float, float, int, int, float]]:
    """
    Run the MPI program once with given parameters. Returns parsed tuple on success,
    or None on failure (errors are printed to stderr).
    """
    cmd = [mpirun, "-np", str(p)]
    if extra_args.strip():
        cmd += shlex.split(extra_args)
    cmd += [exe, str(a), str(b), str(n)]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as e:
        print(f"[ERROR] Failed to execute {mpirun!r} or program {exe!r}: {e}", file=sys.stderr)
        return None

    if res.returncode != 0:
        msg = res.stderr.strip() or res.stdout.strip()
        print(f"[WARN] Command failed (rc={res.returncode}) for n={n}, p={p}: {msg}", file=sys.stderr)
        return None

    # Use the last non-empty stdout line
    out_line = ""
    for line in res.stdout.splitlines():
        s = line.strip()
        if s:
            out_line = s
    if not out_line:
        print(f"[WARN] No output for n={n}, p={p}. Skipping.", file=sys.stderr)
        return None

    try:
        return parse_program_output(out_line)
    except ValueError as e:
        print(f"[WARN] Could not parse output for n={n}, p={p}: {e}", file=sys.stderr)
        return None


def print_table(results: List[dict]):
    """
    Print a neatly aligned table:
      n, p, answer, time(s), speedup, eff
    """
    hdr = f"{'n':>8} {'p':>5} {'answer':>20} {'time(s)':>12} {'speedup':>10} {'eff':>8}"
    print(hdr)
    for row in sorted(results, key=lambda r: (int(r["n"]), int(r["p"]))):
        n = int(row["n"])
        p = int(row["p"])
        ans = float(row["answer"])
        t = float(row["time_seconds"])
        s = row["speedup"]
        e = row["efficiency"]
        s_str = f"{s:.3f}" if isinstance(s, float) else "-"
        e_str = f"{e:.3f}" if isinstance(e, float) else "-"
        print(f"{n:8d} {p:5d} {ans:20.6e} {t:12.6f} {s_str:>10} {e_str:>8}")


def progress_bar(iteration: int, total: int, prefix: str = "Progress", bar_width: int = 40):
    """
    Render a single-line ASCII progress bar.
    Example:
      Progress [##############------------------------]   25/100   25.0%
    """
    if total <= 0:
        total = 1
    pct = iteration / total
    filled = int(round(pct * bar_width))
    bar = "#" * filled + "-" * (bar_width - filled)
    sys.stdout.write(f"\r{prefix} [{bar}] {iteration:>4}/{total:<4} {pct*100:5.1f}%")
    sys.stdout.flush()
    if iteration >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


# ----------------------- Plotting -----------------------

def save_plot_time(results: List[dict], out_png: str, a: float, b: float):
    """
    Plot time vs processes (one curve per n).
    """
    plt.rcParams.update({"font.size": 14})  # axis/title font size
    plt.figure()
    # Group times by n
    by_n: Dict[int, List[Tuple[int, float]]] = {}
    for r in results:
        n = int(r["n"]); p = int(r["p"]); t = float(r["time_seconds"])
        by_n.setdefault(n, []).append((p, t))
    # Plot
    for n, pts in sorted(by_n.items()):
        pts.sort(key=lambda x: x[0])
        xs = [p for p, _ in pts]
        ys = [t for _, t in pts]
        plt.plot(xs, ys, marker='o', label=f"n={n}")
    plt.xlabel("Number of processes (p)")
    plt.ylabel("Time (seconds)")
    plt.title(f"Trapezoidal Rule Timing (a={a}, b={b})")
    plt.legend(title="Trapezoids", loc="upper right", fontsize=10, title_fontsize=11)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_plot_speedup(results: List[dict], out_png: str, a: float, b: float):
    """
    Plot speedup vs processes (one curve per n), with an ideal speedup line.
    Requires p=1 baseline per n.
    """
    plt.rcParams.update({"font.size": 14})
    plt.figure()
    by_n: Dict[int, List[Tuple[int, float]]] = {}
    all_p_values = set()

    for r in results:
        if isinstance(r["speedup"], float):
            n = int(r["n"]); p = int(r["p"]); s = float(r["speedup"])
            by_n.setdefault(n, []).append((p, s))
            all_p_values.add(p)

    if not by_n:
        plt.close()
        return

    # Plot measured speedup curves
    for n, pts in sorted(by_n.items()):
        pts.sort(key=lambda x: x[0])
        xs = [p for p, _ in pts]
        ys = [s for _, s in pts]
        plt.plot(xs, ys, marker='o', label=f"n={n}")

    # Plot ideal speedup line S(p) = p
    if all_p_values:
        p_min, p_max = min(all_p_values), max(all_p_values)
        ideal_xs = [p_min, p_max]
        plt.plot(ideal_xs, ideal_xs, linestyle='--', color='k', label="Ideal")

    plt.xlabel("Number of processes (p)")
    plt.ylabel("Speedup")
    plt.title(f"Speedup vs Processes (a={a}, b={b})")
    plt.legend(title="Trapezoids", loc="upper left", fontsize=10, title_fontsize=11)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def save_plot_efficiency(results: List[dict], out_png: str, a: float, b: float):
    """
    Plot efficiency vs processes (one curve per n), with an ideal efficiency line.
    Requires p=1 baseline per n.
    """
    plt.rcParams.update({"font.size": 14})
    plt.figure()
    by_n: Dict[int, List[Tuple[int, float]]] = {}
    for r in results:
        if isinstance(r["efficiency"], float):
            n = int(r["n"]); p = int(r["p"]); e = float(r["efficiency"])
            by_n.setdefault(n, []).append((p, e))

    if not by_n:
        plt.close()
        return

    # Plot measured efficiency curves
    for n, pts in sorted(by_n.items()):
        pts.sort(key=lambda x: x[0])
        xs = [p for p, _ in pts]
        ys = [e for _, e in pts]
        plt.plot(xs, ys, marker='o', label=f"n={n}")

    # Plot ideal efficiency line E(p) = 1.0
    plt.axhline(y=1.0, linestyle='--', color='k', label="Ideal (1.0)")

    plt.xlabel("Number of processes (p)")
    plt.ylabel("Efficiency")
    plt.ylim(0, 1.2)  # Set y-axis for better visibility around 1.0
    plt.title(f"Efficiency vs Processes (a={a}, b={b})")
    plt.legend(title="Trapezoids", loc="lower left", fontsize=10, title_fontsize=11)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ----------------------- Main -----------------------

def main():
    ap = argparse.ArgumentParser(
        prog="mpi_trap_sweep.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Parameter sweep runner for an MPI trapezoid program producing a single-line result."
    )
    ap.add_argument("a", type=float, help="Left endpoint of integration")
    ap.add_argument("b", type=float, help="Right endpoint of integration")
    ap.add_argument("n1", type=int, help="Lower bound for n (inclusive)")
    ap.add_argument("n2", type=int, help="Upper bound for n (inclusive)")
    ap.add_argument("n_increment", type=int, help="Increment for n (must be > 0)")
    ap.add_argument("p1", type=int, help="Lower bound for p (inclusive)")
    ap.add_argument("p2", type=int, help="Upper bound for p (inclusive)")
    
    ap.add_argument("--p-increment", type=int,
                    help="Use a linear increment for p. Overrides the default power-of-2 method.")
    ap.add_argument("--p-extra-increment", type=int, nargs=2, metavar=('THRESHOLD', 'INCREMENT'),
                    help="(Power-of-2 mode only) Add extra p values with INCREMENT starting from THRESHOLD")

    ap.add_argument("--exe", default="./mpi_trap_modified",
                    help="Path to MPI program (default: ./mpi_trap_modified)")
    ap.add_argument("--mpirun", default="mpirun",
                    help="MPI launcher command (default: mpirun; e.g., mpiexec)")
    ap.add_argument("--csv", default="mpi_trap_sweep.csv",
                    help="CSV output file (default: mpi_trap_sweep.csv)")
    ap.add_argument("--extra-mpirun-args", default="",
                    help="Extra arguments passed to the MPI launcher (quote as one string)")

    ap.add_argument("--timing-png", default="timing.png",
                    help="Timing plot output (default: timing.png)")
    ap.add_argument("--speedup-png", default="speedup.png",
                    help="Speedup plot output (default: speedup.png)")
    ap.add_argument("--efficiency-png", default="efficiency.png",
                    help="Efficiency plot output (default: efficiency.png)")

    args = ap.parse_args()

    # Build sweep sets
    try:
        n_vals = inc_range(args.n1, args.n2, args.n_increment)

        # Decide which mode to use for generating p values
        if args.p_increment is not None:
            # --- Linear Increment Mode ---
            if args.p_extra_increment is not None:
                print("[WARN] --p-extra-increment is ignored when --p-increment is used.", file=sys.stderr)
            print("--> Generating p values with linear increment.")
            p_vals = inc_range(args.p1, args.p2, args.p_increment)
        else:
            # --- Default Power-of-2 Mode ---
            print("--> Generating p values using powers of 2.")
            p_vals = generate_p_values(args.p1, args.p2, args.p_extra_increment)

    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)

    if not n_vals or not p_vals:
        print("[ERROR] Empty sweep ranges. Check bounds and increments.", file=sys.stderr)
        sys.exit(2)

    # Print p_vals to be tested
    print(f"--> Will test the following {len(p_vals)} values for p: {p_vals}")
    print("-" * 60)

    a, b = args.a, args.b
    exe = args.exe
    mpirun_cmd = args.mpirun
    extra = args.extra_mpirun_args

    results: List[Dict[str, object]] = []
    baseline_time_for_n: Dict[int, float] = {}

    total_runs = len(n_vals) * len(p_vals)
    completed = 0
    progress_bar(completed, total_runs, prefix="Progress")

    # Run sweep
    for n in n_vals:
        for p in p_vals:
            parsed = run_once(mpirun_cmd, extra, exe, a, b, n, p)
            if parsed is not None:
                answer, a_out, b_out, n_out, np_out, t_sec = parsed

                # If program echoes different n/p, prefer echoed values
                if n_out != n or np_out != p:
                    print(f"\n[WARN] Program echoed n={n_out}, p={np_out} but requested n={n}, p={p}. Using echoed values.",
                          file=sys.stderr)
                    n = n_out
                    p = np_out

                if p == 1:
                    baseline_time_for_n[n_out] = t_sec

                results.append({
                    "n": n_out,
                    "p": np_out,
                    "a": a_out,
                    "b": b_out,
                    "answer": answer,
                    "time_seconds": t_sec,
                })

            completed += 1
            # Update progress bar after each attempt (success or fail)
            progress_bar(completed, total_runs, prefix="Progress")

    if not results:
        print("\n[ERROR] No successful runs. Check mpirun/executable paths.", file=sys.stderr)
        sys.exit(1)

    # Compute speedup/efficiency where possible
    for row in results:
        n = int(row["n"])
        p = int(row["p"])
        t = float(row["time_seconds"])
        base = baseline_time_for_n.get(n)
        if base and t > 0:
            speedup = base / t
            efficiency = speedup / p
        else:
            speedup = None
            efficiency = None
        row["speedup"] = speedup if speedup is not None else ""
        row["efficiency"] = efficiency if efficiency is not None else ""

    # Write CSV
    csv_path = args.csv
    fieldnames = ["n", "p", "a", "b", "answer", "time_seconds", "speedup", "efficiency"]
    try:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in sorted(results, key=lambda r: (int(r["n"]), int(r["p"]))):
                w.writerow(row)
    except OSError as e:
        print(f"\n[ERROR] Could not write CSV to {csv_path!r}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nWrote {len(results)} rows to {csv_path}")
    print_table(results)

    # Make plots
    save_plot_time(results, args.timing_png, a, b)
    save_plot_speedup(results, args.speedup_png, a, b)
    save_plot_efficiency(results, args.efficiency_png, a, b)


if __name__ == "__main__":
    main()