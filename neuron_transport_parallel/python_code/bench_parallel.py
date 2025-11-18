#!/usr/bin/env python3

"""
bench_parallel.py

Runs benchmarks for the parallel (pthreads or MPI) mc_slab implementations.
Sweeps over a range of particle counts (n) and processor/thread counts (P).
Calculates and plots timing, speedup, efficiency, and isoefficiency.

This script implements the requirements from pages 5-6 of MCNT-Parallel.pdf.
It now also compiles the required C executable before running.
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
    n_values = sorted(df_summary['n'].unique())
    
    # Use a color map and markers
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_values)))
    markers = ['o', 'v', 's', 'P', 'X', '*', 'D', '^', '<', '>']
    
    def get_style(i):
        return {'color': colors[i], 'marker': markers[i % len(markers)]}

    # --- 1. Timing vs Threads (Page 6, top-left) ---
    plt.figure(figsize=(10, 6))
    for i, n in enumerate(n_values):
        df_n = df_summary[df_summary['n'] == n].sort_values('P')
        if df_n.empty: continue
        plt.errorbar(
            df_n['P'], df_n['time_mean'], yerr=df_n['time_std'],
            label=f'n={n/1e6:.1f}M' if n >= 1e6 else f'n={n/1e3:.0f}K',
            **get_style(i), capsize=4, alpha=0.8
        )
    plt.xlabel('Threads/Processes (P)')
    plt.ylabel('Time (s)')
    plt.title('Timing vs Threads')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title="Particles (n)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Make room for legend
    plt.savefig(os.path.join(plots_dir, 'timing_vs_P_by_n.png'), dpi=dpi)
    plt.close()

    # --- 2. Speedup vs Threads (Page 6, top-right) ---
    plt.figure(figsize=(10, 6))
    # Add ideal speedup line
    max_p = df_summary['P'].max()
    plt.plot([1, max_p], [1, max_p], 'k--', label='Ideal', alpha=0.7)
    
    for i, n in enumerate(n_values):
        df_n = df_summary[df_summary['n'] == n].sort_values('P')
        if df_n.empty: continue
        plt.plot(
            df_n['P'], df_n['speedup'],
            label=f'n={n/1e6:.1f}M' if n >= 1e6 else f'n={n/1e3:.0f}K',
            **get_style(i), alpha=0.8
        )
    plt.xlabel('Threads/Processes (P)')
    plt.ylabel('Speedup S = T1 / TP')
    plt.title('Speedup vs Threads')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title="Particles (n)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(plots_dir, 'speedup_vs_P_by_n.png'), dpi=dpi)
    plt.close()

    # --- 3. Efficiency vs Threads (Page 6, bottom-left) ---
    plt.figure(figsize=(10, 6))
    for i, n in enumerate(n_values):
        df_n = df_summary[df_summary['n'] == n].sort_values('P')
        if df_n.empty: continue
        plt.plot(
            df_n['P'], df_n['efficiency'],
            label=f'n={n/1e6:.1f}M' if n >= 1e6 else f'n={n/1e3:.0f}K',
            **get_style(i), alpha=0.8
        )
    plt.xlabel('Threads/Processes (P)')
    plt.ylabel('Efficiency E = S / P')
    plt.title('Efficiency vs Threads')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title="Particles (n)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1.1)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(plots_dir, 'efficiency_vs_P_by_n.png'), dpi=dpi)
    plt.close()
    
    logger.info(f"Generated core plots in {plots_dir}")


def calculate_and_plot_isoefficiency(df_summary, eff_targets, plots_dir, dpi=150):
    """
    Calculates and plots isoefficiency data.
    Finds the minimum 'n' required to achieve a target efficiency 'E' for each 'P'.
    """
    logger.info(f"Calculating isoefficiency for targets: {eff_targets}")
    os.makedirs(plots_dir, exist_ok=True)
    
    p_values = sorted(df_summary['P'].unique())
    iso_data = []

    for E_target in eff_targets:
        min_n_for_p = []
        for P in p_values:
            if P == 1:
                # Efficiency is always 1.0 at P=1. 
                # Find the smallest n tested at P=1.
                min_n = df_summary[df_summary['P'] == 1]['n'].min()
                min_n_for_p.append(min_n)
                continue

            # Find all (n, efficiency) pairs for this P
            df_p = df_summary[df_summary['P'] == P].sort_values('n')
            # Find runs that meet the target efficiency
            meets_target = df_p[df_p['efficiency'] >= E_target]
            
            if not meets_target.empty:
                # Success: Find the smallest 'n' that met the target
                min_n = meets_target['n'].min()
                min_n_for_p.append(min_n)
            else:
                # Failure: No 'n' was large enough for this 'P'
                min_n_for_p.append(np.nan) # Use NaN to indicate failure
        
        iso_data.append({'E_target': E_target, 'n_values': min_n_for_p})

    # --- 4. Isoefficiency Surface (Page 6, bottom-right) ---
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', 'D', '^']
    
    # For CSV output
    iso_df_data = {'P': p_values}

    for i, data in enumerate(iso_data):
        E_target = data['E_target']
        n_values = data['n_values']
        
        # Plot only valid (non-NaN) data points
        valid_p = [p for p, n in zip(p_values, n_values) if not np.isnan(n)]
        valid_n = [n for n in n_values if not np.isnan(n)]

        if not valid_n:
            logger.warning(f"No 'n' values found for E_target={E_target}. Skipping plot line.")
            continue
        
        plt.plot(
            valid_p, valid_n,
            label=f'E ≥ {E_target:.2f}',
            marker=markers[i % len(markers)]
        )
        
        # Add to CSV data
        iso_df_data[f'min_n_E>={E_target:.2f}'] = n_values

    # Save CSV data
    df_iso = pd.DataFrame(iso_df_data)
    csv_path = os.path.join(plots_dir, 'isoefficiency_data.csv')
    df_iso.to_csv(csv_path, index=False)
    logger.info(f"Wrote isoefficiency data to {csv_path}")

    # Format the plot
    plt.xlabel('Threads/Processes (P)')
    plt.ylabel('Minimal n (particles) to achieve target E')
    plt.title('Isoefficiency Surface')
    plt.legend(title="Targets")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.yscale('log') # Problem size often grows non-linearly
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'isoefficiency_surface.png'), dpi=dpi)
    plt.close()

    logger.info(f"Generated isoefficiency plots in {plots_dir}")

def plot_efficiency_heatmap(df_summary, plots_dir, dpi=150):
    """
    Generates a heatmap of efficiency (E) vs. P and n.
    This is an alternative, more intuitive view of isoefficiency.
    """
    logger.info("Generating efficiency heatmap...")
    try:
        # Pivot the data to create a 2D grid
        # Index (rows) = n, Columns (cols) = P, Values = efficiency
        df_pivot = df_summary.pivot(index='n', columns='P', values='efficiency')

        # Get sorted axis labels
        n_values = sorted(df_pivot.index)
        p_values = sorted(df_pivot.columns)

        # Get the 2D data array    
        efficiency_data = df_pivot.to_numpy()
        plt.figure(figsize=(max(10, len(p_values)), max(8, len(n_values))))

        # Use imshow to plot the 2D data
        # We use a blue-to-red colormap (coolwarm) where 1.0=red, 0.5=white, 0.0=blue
        # Or 'viridis' (green-to-yellow)
        cmap = 'viridis'
        im = plt.imshow(efficiency_data, cmap=cmap, aspect='auto', interpolation='nearest', vmin=0.0, vmax=1.0)

        # Add a color bar to show what the colors mean
        plt.colorbar(im, label='Efficiency (E = S / P)')

        # Set the ticks and labels based on the P and n values
        plt.xticks(ticks=np.arange(len(p_values)), labels=p_values)
        plt.yticks(ticks=np.arange(len(n_values)), labels=[f'{n/1e6:.1f}M' if n >= 1e6 else f'{n/1e3:.0f}K' for n in n_values])

        plt.xlabel('Threads/Processes (P)')
        plt.ylabel('Particles (n)')
        plt.title('Efficiency Heatmap (E vs. P, n)')
        # Add text annotations for each cell to show the exact value
        for i in range(len(n_values)):
            for j in range(len(p_values)):
                val = efficiency_data[i, j]
                if not np.isnan(val):
                    # Choose text color for contrast
                    # This threshold works well for 'viridis'
                    text_color = 'white' if val < 0.4 else 'black'
                    plt.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=10)
                    
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'efficiency_heatmap.png'), dpi=dpi)
        plt.close()
        logger.info(f"Generated efficiency heatmap in {plots_dir}")
    except Exception as e:
        logger.exception("Failed to generate efficiency heatmap: %s", e)
        # Don't crash the whole script if this optional plot fails
        pass

# --- C Code Compilation Function (Adapted from sweep_mc_slab.py) ---

def compile_c_code(c_code_dir, target_name):
    """
    Compiles a specific C code target using 'make <target_name>'
    in the specified directory.
    Assumes this script is run from 'python_code/'
    """
    # Get absolute path to the C code directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    c_dir_abs = os.path.abspath(os.path.join(script_dir, c_code_dir))
    # The executable path is <c_dir_abs>/<target_name>
    exe_path_abs = os.path.join(c_dir_abs, target_name)
    
    logger = logging.getLogger(__name__) 
    logger.info("Compiling C target '%s' in: %s", target_name, c_dir_abs)
    
    # Run 'make clean' (cleans all targets)
    try:
        proc_clean = subprocess.run(
            ["make", "clean"], 
            cwd=c_dir_abs, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        if proc_clean.returncode != 0:
            logger.warning("'make clean' failed. Continuing anyway...")
            logger.debug("make clean stderr: %s", proc_clean.stderr)
    except Exception as e:
        logger.warning("'make clean' failed: %s. Continuing anyway...", e)

    # Run 'make <target_name>'
    try:
        proc_make = subprocess.run(
            ["make", target_name], # Pass the specific target
            cwd=c_dir_abs, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=True  # Raise error if make fails
        )
        logger.debug("make stdout: %s", proc_make.stdout)
        logger.info("C target '%s' compiled successfully.", target_name)
    except subprocess.CalledProcessError as e:
        logger.error("'make %s' failed! Return code: %s", target_name, e.returncode)
        logger.error("make stdout: %s", getattr(e, 'stdout', ''))
        logger.error("make stderr: %s", getattr(e, 'stderr', ''))
        sys.exit(1) # Exit if compilation fails
    except Exception as e:
        logger.exception("Unexpected error while running 'make %s': %s", target_name, e)
        sys.exit(1)

    # Check if executable exists
    if not os.path.exists(exe_path_abs):
        logger.error("Executable not found after compile: %s", exe_path_abs)
        sys.exit(1)
        
    return exe_path_abs

# --- Main Simulation ---

def build_command(args, n, P):
    """Builds the subprocess command array based on the mode."""
    
    # Base command: ./exe C Cc H n
    base_cmd = [
        args.exe, # This is set in main() after compilation
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
        return [args.mpi_executable, '-np', str(P)] + base_cmd
        
    else:
        # This case is handled in main(), but good to be defensive
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

        # "Above & Beyond": The C code *also* prints its internal wall time
        # to stderr. This is often more accurate than the Python wrapper's time.
        # Let's try to parse that first.
        try:
            internal_time = float(result.stderr.strip().split('\n')[-1])
            runtime_wall = internal_time
        except (ValueError, IndexError):
            # Fallback to python's time if stderr parsing fails
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
        description='Benchmark parallel mc_slab (pthreads or MPI).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Executable args
    # --- MODIFIED: Removed --exe argument ---
    parser.add_argument('--mode', choices=['pthreads', 'mpi'], required=True, help='Parallelization mode.')
    parser.add_argument('--mpi-executable', default='mpirun', help='Path to mpirun (if mode=mpi)')

    # Simulation physics args
    parser.add_argument('--C', type=float, default=0.5, help='Total interaction coeff')
    parser.add_argument('--CC', type=float, default=0.1, help='Absorbing component (Cc)')
    parser.add_argument('--H', type=float, default=5.0, help='Slab thickness H')
    parser.add_argument('--seed', type=int, default=12345, help='Random number seed')

    # Sweep args (n)
    parser.add_argument('--n-start', type=int, default=100000, help='Min particle count n')
    parser.add_argument('--n-max', type=int, default=1000000, help='Max particle count n')
    parser.add_argument('--n-steps', type=int, default=2, help='Number of n values to test (logarithmic scale)')
    
    # Sweep args (P)
    parser.add_argument('--P-start', type=int, default=1, help='Min threads/processes P')
    parser.add_argument('--P-step', type=int, default=1, help='Step size for P sweep')
    parser.add_argument('--P-max', type=int, default=8, help='Max threads/processes P')
    
    # Benchmarking control
    parser.add_argument('--trials', type=int, default=3, help='Number of timed trials per (n, P) config')
    parser.add_argument('--warmup', type=int, default=1, help='Number of warmup runs (not timed)')
    
    # Isoefficiency
    parser.add_argument('--eff-targets', default='0.5,0.7,0.8', help='Comma-separated target efficiencies (e.g., 0.5,0.7,0.8)')
    
    # Output args
    parser.add_argument('--results-dir', default=None, help='Directory to save results. (Default: perf_results_MODE_timestamp)')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for output plots')
    
    # Tracing (optional, for debugging)
    parser.add_argument('--trace-file', default=None, help='Base path for trace files (e.g., "trace/run").')
    parser.add_argument('--trace-every', type=int, default=10000, help='Record every m-th iteration (if --trace-file is set).')
    
    # User Request: Cleanup
    parser.add_argument('--no-cleanup', action='store_true', default=False, help='If set, do *not* delete intermediate trace files.')

    args = parser.parse_args()

    # --- 1. Setup & Compile C Code ---
    
    # Determine the target executable name based on the mode
    if args.mode == 'pthreads':
        exe_name = 'mc_slab_pthreads'
    elif args.mode == 'mpi':
        exe_name = 'mc_slab_mpi'
    else:
        # This case should be handled by argparse 'choices', but good to be safe
        logger.error("Unknown mode: %s", args.mode)
        sys.exit(1)

    # Compile the required C executable
    try:
        c_code_dir_relative = '../c_code'
        compiled_exe_path = compile_c_code(c_code_dir_relative, exe_name)
        
        # Add the compiled exe path to the args object
        args.exe = compiled_exe_path
        
    except Exception as e:
        logger.exception("Could not compile C code: %s", e)
        logger.info("Please ensure 'make' is installed and Makefile is present in %s.", c_code_dir_relative)
        sys.exit(1)

    # --- End of C Code compilation ---

    if args.results_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.results_dir = f"perf_results_{args.mode}_{timestamp}"
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set up file logging
    fh = logging.FileHandler(os.path.join(args.results_dir, "benchmark.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    logger.info("Starting benchmark run.")
    logger.info(f"Using executable: {args.exe}") # Log the auto-detected exe
    logger.info("Parameters: %s", vars(args))

    # Build task list
    if args.n_steps > 1:
        n_values = np.logspace(np.log10(args.n_start), np.log10(args.n_max), args.n_steps, dtype=int)
    else:
        n_values = [args.n_start]
        
    # --- START: MODIFIED SECTION ---
    # Create the user-requested range of P values
    p_values_range = range(args.P_start, args.P_max + 1, args.P_step)
    
    # Use a set for easy manipulation
    p_values_set = set(p_values_range)
    
    # Add P=1 if it's not present, as it's required for T1 baseline
    if 1 not in p_values_set:
        logger.info("--P-start was not 1. Automatically adding P=1 for serial baseline (T1).")
        p_values_set.add(1)
        
    # Sort the final list to ensure runs happen in a logical order
    p_values = sorted(list(p_values_set))
    
    tasks = [(n, p) for n in n_values for p in p_values]
    # --- END: MODIFIED SECTION ---
    
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
            cmd = build_command(args, n, P)
            run_simulation(cmd) # Discard results
            
        # Timed trials
        for t in range(args.trials):
            cmd = build_command(args, n, P)
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
    logger.info("WWrote summary data to %s", summary_csv_path)

    # --- 4. Plot Results ---
    logger.info("Generating plots...")
    plots_dir = os.path.join(args.results_dir, "plots")
    
    plot_benchmarks(df_summary, plots_dir, args.dpi)
    
    calculate_and_plot_isoefficiency(df_summary, eff_targets, plots_dir, args.dpi)
    
    plot_efficiency_heatmap(df_summary, plots_dir, args.dpi)

    # --- 5. Cleanup ---
    if args.trace_file and not args.no_cleanup:
        logger.info("Cleaning up intermediate trace files...")
        # Find all files matching the trace_file base
        trace_dir = os.path.dirname(args.trace_file) or "."
        trace_base_name = os.path.basename(args.trace_file)
        
        # Look for trace_r*.csv or trace_t*.csv
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