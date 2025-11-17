#!/usr/bin/env python3

"""
run_scalability_benchmark.py

Runs a scalability benchmark (fixed problem size, sweep P)
for the parallel (pthreads or MPI) mc_slab implementations.

This script is designed to be run *inside* a large Slurm
interactive session (e.g., `srun -n 128 ...`).
"""

import argparse
import os
import sys
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import time
import glob
import logging
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Plotting Functions ---

def plot_benchmarks(df_summary, plots_dir, dpi=150):
    """
    Generates all benchmark plots from the summary dataframe.
    """
    os.makedirs(plots_dir, exist_ok=True)
    
    # We only have one 'n', so we don't need a complex legend
    df_n = df_summary.sort_values('P')
    n_val = df_n['n'].iloc[0]
    n_label = f'n={n_val/1e6:.1f}M' if n_val >= 1e6 else f'n={n_val/1e3:.0f}K'
    # Style for plt.errorbar (has capsize)
    errorbar_style = {'marker': 'o', 'capsize': 4, 'alpha': 0.8}
    # Style for plt.plot (no capsize)
    plot_style = {'marker': 'o', 'alpha': 0.8}

    # --- 1. Timing vs Threads (Page 6, top-left) ---
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        df_n['P'], df_n['time_mean'], yerr=df_n['time_std'],
        label=n_label, **errorbar_style
    )
    plt.xlabel('Threads/Processes (P)')
    plt.ylabel('Time (s)')
    plt.title(f'Timing vs Threads ({n_label})')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'timing_vs_P.png'), dpi=dpi)
    plt.close()

    # --- 2. Speedup vs Threads (Page 6, top-right) ---
    plt.figure(figsize=(10, 6))
    max_p = df_summary['P'].max()
    plt.plot([1, max_p], [1, max_p], 'k--', label='Ideal', alpha=0.7)
    
    plt.plot(
        df_n['P'], df_n['speedup'],
        label=n_label, **plot_style
    )
    plt.xlabel('Threads/Processes (P)')
    plt.ylabel('Speedup S = T1 / TP')
    plt.title(f'Speedup vs Threads ({n_label})')
    plt.legend(loc="upper left")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'speedup_vs_P.png'), dpi=dpi)
    plt.close()

    # --- 3. Efficiency vs Threads (Page 6, bottom-left) ---
    plt.figure(figsize=(10, 6))
    plt.plot(
        df_n['P'], df_n['efficiency'],
        label=n_label, **plot_style
    )
    plt.xlabel('Threads/Processes (P)')
    plt.ylabel('Efficiency E = S / P')
    plt.title(f'Efficiency vs Threads ({n_label})')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'efficiency_vs_P.png'), dpi=dpi)
    plt.close()
    
    logger.info(f"Generated core plots in {plots_dir}")


def calculate_and_plot_isoefficiency(df_summary, eff_targets, plots_dir, dpi=150):
    """
    Calculates and plots isoefficiency data.
    """
    # This plot only makes sense if you *sweep n*
    if df_summary['n'].nunique() <= 1:
        logger.warning("Only one 'n' value found. Skipping isoefficiency plot.")
        logger.warning("To generate this plot, run this script for multiple '--n' values.")
        return
        
    logger.info(f"Calculating isoefficiency for targets: {eff_targets}")
    os.makedirs(plots_dir, exist_ok=True)
    
    p_values = sorted(df_summary['P'].unique())
    iso_data = []

    for E_target in eff_targets:
        min_n_for_p = []
        for P in p_values:
            if P == 1:
                min_n = df_summary[df_summary['P'] == 1]['n'].min()
                min_n_for_p.append(min_n)
                continue
            df_p = df_summary[df_summary['P'] == P].sort_values('n')
            meets_target = df_p[df_p['efficiency'] >= E_target]
            
            if not meets_target.empty:
                min_n = meets_target['n'].min()
                min_n_for_p.append(min_n)
            else:
                min_n_for_p.append(np.nan)
        
        iso_data.append({'E_target': E_target, 'n_values': min_n_for_p})

    plt.figure(figsize=(10, 6))
    markers = ['o', 's', 'D', '^']
    iso_df_data = {'P': p_values}

    for i, data in enumerate(iso_data):
        E_target = data['E_target']
        n_values = data['n_values']
        valid_p = [p for p, n in zip(p_values, n_values) if not np.isnan(n)]
        valid_n = [n for n in n_values if not np.isnan(n)]
        if not valid_n:
            logger.warning(f"No 'n' values found for E_target={E_target}. Skipping plot line.")
            continue
        plt.plot(valid_p, valid_n, label=f'E ≥ {E_target:.2f}', marker=markers[i % len(markers)])
        iso_df_data[f'min_n_E>={E_target:.2f}'] = n_values

    df_iso = pd.DataFrame(iso_df_data)
    csv_path = os.path.join(plots_dir, 'isoefficiency_data.csv')
    df_iso.to_csv(csv_path, index=False)
    logger.info(f"Wrote isoefficiency data to {csv_path}")

    plt.xlabel('Threads/Processes (P)')
    plt.ylabel('Minimal n (particles) to achieve target E')
    plt.title('Isoefficiency Surface')
    plt.legend(title="Targets")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'isoefficiency_surface.png'), dpi=dpi)
    plt.close()

    logger.info(f"Generated isoefficiency plots in {plots_dir}")

# --- Main Simulation ---

def build_command(args, n, P, exe_path):
    """Builds the subprocess command array based on the mode."""
    
    # Base command: ./exe C Cc H n
    base_cmd = [
        exe_path,
        str(args.C),
        str(args.CC),
        str(args.H),
        str(n)
    ]
    
    # Add optional args
    if args.seed is not None:
        base_cmd.extend(['--seed', str(args.seed)])
    
    # If tracing, create a unique base path for this run
    if args.trace_file:
        trace_base = f"{args.trace_file}_n{n}_P{P}"
        base_cmd.extend(['--trace-file', trace_base, '--trace-every', str(args.trace_every)])

    # Add parallel-specific args
    if args.mode == 'pthreads':
        # ./exe C Cc H n [opts] T
        base_cmd.append(str(P))
        return base_cmd
        
    elif args.mode == 'mpi':
        # mpirun -np P ./exe C Cc H n [opts]
        # This works because we are *already* in a Slurm allocation
        return [args.mpi_executable, '-np', str(P)] + base_cmd
        
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


def run_simulation(cmd):
    """
    Runs a single simulation command and captures its wall time.
    Returns (wall_time, stdout, stderr)
    """
    try:
        t0_wall = time.perf_counter()
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=None # Use main timeout logic
        )
        
        t1_wall = time.perf_counter()
        runtime_wall = t1_wall - t0_wall

        if result.returncode != 0:
            logger.warning("Run failed with code %d. CMD: %s", result.returncode, " ".join(cmd))
            logger.warning("STDERR: %s", result.stderr.strip())
            return None, None, result.stderr

        # The C code prints its internal wall time to stderr.
        try:
            internal_time = float(result.stderr.strip().split('\n')[-1])
            runtime_wall = internal_time
        except (ValueError, IndexError):
            pass 

        return runtime_wall, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        logger.error("Run timed out. CMD: %s", " ".join(cmd))
        return None, None, "Timeout"
    except Exception as e:
        logger.exception("An error occurred running command: %s", " ".join(cmd))
        return None, None, str(e)


def main():
    parser = argparse.ArgumentParser(
        description='Run scalability benchmark (fixed n, sweep P) inside an HPC interactive session.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Executable args
    parser.add_argument('--mode', choices=['pthreads', 'mpi'], required=True, help='Parallelization mode.')
    parser.add_argument('--mpi-executable', default='mpirun', help='Path to mpirun (if mode=mpi)')

    # Simulation physics args
    parser.add_argument('--C', type=float, default=0.5, help='Total interaction coeff')
    parser.add_argument('--CC', type=float, default=0.1, help='Absorbing component (Cc)')
    parser.add_argument('--H', type=float, default=5.0, help='Slab thickness H')
    parser.add_argument('--seed', type=int, default=12345, help='Random number seed')

    # ** FIXED PROBLEM SIZE **
    parser.add_argument('--n', type=int, default=50000000, help='Fixed, large particle count (n)')
    
    # ** SWEEP ARGS (P) **
    parser.add_argument('--P-start', type=int, default=1, help='Min threads/processes P')
    parser.add_argument('--P-step', type=int, default=1, help='Step size for P sweep')
    parser.add_argument('--P-max', type=int, default=8, help='Max threads/processes P')
    
    # Benchmarking control
    parser.add_argument('--trials', type=int, default=3, help='Number of timed trials per (n, P) config')
    parser.add_argument('--warmup', type=int, default=1, help='Number of warmup runs (not timed)')
    
    # Isoefficiency
    parser.add_argument('--eff-targets', default='0.5,0.7,0.8', help='Comma-separated target efficiencies')
    
    # Output args
    parser.add_argument('--results-dir', default=None, help='Directory to save results. (Default: perf_results_MODE_timestamp)')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for output plots')
    
    # Tracing (optional, for debugging)
    parser.add_argument('--trace-file', default=None, help='Base path for trace files (e.g., "trace/run").')
    parser.add_argument('--trace-every', type=int, default=10000, help='Record every m-th iteration (if --trace-file is set).')
    
    # User Request: Cleanup
    parser.add_argument('--no-cleanup', action='store_true', default=False, help='If set, do *not* delete intermediate trace files.')

    args = parser.parse_args()

    # --- 1. Setup ---
    # Determine exe_path from mode
    exe_name = f"mc_slab_{args.mode}"
    exe_path = os.path.join('../c_code', exe_name)
    
    if not os.path.exists(exe_path):
        logger.error(f"Executable not found at {exe_path}. Did you `make all` in ../c_code?")
        sys.exit(1)

    if args.results_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.results_dir = f"perf_scalability_{args.mode}_{timestamp}"
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set up file logging
    fh = logging.FileHandler(os.path.join(args.results_dir, "benchmark.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    logger.info("Starting benchmark run.")
    logger.info(f"Using executable: {exe_path}")
    logger.info("Parameters: %s", vars(args))

    # Build task list
    # Use your fixed problem size
    n_values = [args.n] 
    
    # Build P sweep
    p_values = range(args.P_start, args.P_max + 1, args.P_step)
    
    # ** NEW: Add powers-of-two P values for better scalability curves **
    # You can customize this
    p_sweep_vals = [1, 2, 4, 8, 16, 32, 64, 128]
    p_values_to_run = sorted(list(set(p for p in p_sweep_vals if p >= args.P_start and p <= args.P_max)))
    
    logger.info(f"Running P values: {p_values_to_run}")

    tasks = [(n, p) for n in n_values for p in p_values_to_run]
    
    eff_targets = [float(e) for e in args.eff_targets.split(',')]
    
    # --- 2. Run Simulations ---
    raw_results = []
    total_tasks = len(tasks)
    
    logger.info("Starting %d simulation configurations...", total_tasks)

    for i, (n, P) in enumerate(tasks):
        logger.info("[%d/%d] Running n=%d, P=%d...", i + 1, total_tasks, n, P)
        
        # Warmup runs
        for w in range(args.warmup):
            logger.debug("Warmup %d/%d (n=%d, P=%d)...", w + 1, args.warmup, n, P)
            cmd = build_command(args, n, P, exe_path)
            run_simulation(cmd) # Discard results
            
        # Timed trials
        for t in range(args.trials):
            cmd = build_command(args, n, P, exe_path)
            logger.debug("Trial %d/%d (n=%d, P=%d)...", t + 1, args.trials, n, P)
            
            runtime, stdout_str, stderr_str = run_simulation(cmd)
            
            if runtime is not None:
                raw_results.append({
                    'n': n,
                    'P': P,
                    'trial': t,
                    'time_sec': runtime,
                })
            else:
                logger.error("Failed trial %d for (n=%d, P=%d).", t, n, P)

    if not raw_results:
        logger.error("No successful simulation runs. Exiting.")
        sys.exit(1)

    # Save raw CSV
    df_raw = pd.DataFrame(raw_results)
    raw_csv_path = os.path.join(args.results_dir, "raw_runs.csv")
    df_raw.to_csv(raw_csv_path, index=False)
    logger.info("Wrote raw timing data to %s", raw_csv_path)

    # --- 3. Process Results ---
    logger.info("Processing results...")
    
    # Calculate mean/std time per (n, P)
    df_agg = df_raw.groupby(['n', 'P'])['time_sec'].agg(['mean', 'std']).reset_index()
    df_agg.rename(columns={'mean': 'time_mean', 'std': 'time_std'}, inplace=True)
    
    # Get T1 (serial time) for each n
    df_t1 = df_agg[df_agg['P'] == 1][['n', 'time_mean']].rename(columns={'time_mean': 'T1'})
    
    if df_t1.empty:
        logger.error("No P=1 data was collected. Cannot calculate speedup/efficiency.")
        logger.error("Please ensure --P-start is 1.")
        sys.exit(1)
        
    # Merge T1 back and calculate speedup/efficiency
    df_summary = pd.merge(df_agg, df_t1, on='n', how='left')
    df_summary['speedup'] = df_summary['T1'] / df_summary['time_mean']
    df_summary['efficiency'] = df_summary['speedup'] / df_summary['P']
    
    summary_csv_path = os.path.join(args.results_dir, "summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    logger.info("Wrote summary data to %s", summary_csv_path)

    # --- 4. Plot Results ---
    logger.info("Generating plots...")
    plots_dir = os.path.join(args.results_dir, "plots")
    
    plot_benchmarks(df_summary, plots_dir, args.dpi)
    
    # This will be skipped because we only have one 'n' value, which is correct
    calculate_and_plot_isoefficiency(df_summary, eff_targets, plots_dir, args.dpi)

    # --- 5. Cleanup ---
    if args.trace_file and not args.no_cleanup:
        logger.info("Cleaning up intermediate trace files...")
        trace_dir = os.path.dirname(args.trace_file) or "."
        trace_base_name = os.path.basename(args.trace_file)
        
        pattern = os.path.join(trace_dir, f"{trace_base_name}*_[rt][0-9]*.csv")
        
        cleaned_count = 0
        for f in glob.glob(pattern):
            try:
                os.remove(f)
                cleaned_count += 1
            except Exception as e:
                logger.warning("Failed to delete trace file %s: %s", f, e)
        logger.info("Cleaned up %d trace files.", cleaned_count)

    logger.info("Benchmark complete. Results are in: %s", os.path.abspath(args.results_dir))

if __name__ == "__main__":
    main()