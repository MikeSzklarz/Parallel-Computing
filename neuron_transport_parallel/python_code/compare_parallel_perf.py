#!/usr/bin/env python3
"""
compare_parallel_perf.py

[Above and Beyond]

Runs both a Pthreads and an MPI executable for the same set of parameters
and plots their speedup curves on the same graphs for a direct
head-to-head performance comparison.

This script now auto-compiles the required C executables,
includes default values for all arguments, and generates
Timing, Speedup, and Efficiency plots using sweep-based arguments.

Addresses the request on 'MCNT-Parallel.pdf', page 10:
"compare the MPI version to the Pthread version"
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
import logging
import itertools
import stat # Added for file permissions

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Matplotlib style
plt.style.use('ggplot')

# --- C Code Compilation Functions ---

def clean_c_code(c_code_dir):
    """
    Runs 'make clean' once in the specified directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    c_dir_abs = os.path.abspath(os.path.join(script_dir, c_code_dir))
    
    logger.info("Cleaning C code targets in: %s", c_dir_abs)
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
        
    # Ensure the file is executable
    try:
        # Set permissions to rwxr-xr-x (owner can rwx, group/others can rx)
        os.chmod(exe_path_abs, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        logger.info("Set execute permissions on %s", exe_path_abs)
    except Exception as e:
        logger.error("Failed to set execute permissions on %s: %s", exe_path_abs, e)
        sys.exit(1)
            
    return exe_path_abs

# --- Helper Functions ---

def format_n(n):
    """Formats large n values for plot labels."""
    if n >= 1_000_000:
        return f'{n / 1_000_000:.1f}M'.replace('.0M', 'M')
    if n >= 1_000:
        return f'{n // 1_000}K'
    return str(n)

def run_simulation(cmd_args, timeout):
    """
    Runs a single simulation subprocess and returns its wall time.
    """
    try:
        t_start = time.perf_counter()
        # Note: We now capture stderr to parse the internal C-code time
        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True
        )
        t_end = time.perf_counter()
        
        # Try to parse internal time from stderr (more accurate)
        try:
            internal_time = float(result.stderr.strip().split('\n')[-1])
            return internal_time
        except (ValueError, IndexError):
            # Fallback to python's time
            logger.debug("Could not parse internal time from stderr. Falling back to Python timer.")
            return (t_end - t_start)
            
    except subprocess.CalledProcessError as e:
        # This catches errors *from the C program* (non-zero return code)
        logger.warning(f"Run failed for {' '.join(cmd_args)}. Error: {e.stderr.strip()}")
        return None
    except Exception as e:
        # This catches Python errors (e.g., FileNotFoundError)
        logger.warning(f"Run failed for {' '.join(cmd_args)}. Error: {e}")
        return None

def create_run_command(exe, run_type, base_physics_args, seed, n, P):
    """Builds the command list for subprocess."""
    
    # C usage: exe C Cc H n --seed s
    cmd = [exe] + base_physics_args + [str(n), '--seed', str(seed)]
    
    if run_type == 'pthread':
        # pthreads usage: exe C Cc H n --seed s T
        return cmd + [str(P)]
    elif run_type == 'mpi':
        # mpi usage: mpirun -np P exe C Cc H n --seed s
        return ['mpirun', '-np', str(P)] + cmd
    else:
        raise ValueError(f"Invalid run_type: {run_type}")

def run_benchmark_for_exe(exe, run_type, base_physics_args, seed, n_values, p_values, trials, warmup, timeout):
    """Runs the full benchmark for one executable."""
    
    raw_results = []
    # Create a product of n_values and p_values for the job list
    param_combinations = list(itertools.product(n_values, p_values))
    total_jobs = len(param_combinations)
    
    logger.info(f"--- Benchmarking {run_type.upper()} ({exe}) ---")
    
    job_idx = 1
    for n, P in param_combinations:
        logger.info(f"    [Job {job_idx}/{total_jobs}]: n={format_n(n)}, P={P}")
        
        cmd = create_run_command(exe, run_type, base_physics_args, seed, n, P)
        
        # Warmup
        for w in range(warmup):
            logger.debug(f"Warmup {w+1}/{warmup}...")
            run_simulation(cmd, timeout)
            
        # Timed trials
        trial_times = []
        for t in range(trials):
            logger.debug(f"Trial {t+1}/{trials}...")
            run_time = run_simulation(cmd, timeout)
            if run_time is not None:
                trial_times.append(run_time)
        
        if trial_times:
            raw_results.append({
                'n': n,
                'P': P,
                'T_mean': np.mean(trial_times),
                'T_std': np.std(trial_times),
                'run_type': run_type
            })
        else:
            logger.warning(f"All trials failed for n={format_n(n)}, P={P}")
        
        job_idx += 1
        
    return pd.DataFrame(raw_results)

def calculate_speedup_and_efficiency(df):
    """Calculates speedup and efficiency from a mean-time dataframe."""
    
    df_t1 = df[df['P'] == 1][['n', 'T_mean']].rename(
        columns={'T_mean': 'T1'}
    )
    
    if df_t1.empty:
        logger.error(f"No P=1 data found for {df['run_type'].iloc[0]}. Cannot calculate speedup/efficiency.")
        return pd.DataFrame()

    df = pd.merge(df, df_t1, on='n', how='left')
    df['Speedup_S'] = df['T1'] / df['T_mean']
    df['Efficiency_E'] = df['Speedup_S'] / df['P']
    return df


# --- Plotting Functions ---

def plot_comparison_timing(df_p_n, df_m_n, n_val, p_ticks, max_p, results_dir, plot_format):
    """Generates the Timing vs. P comparison plot."""
    plt.figure(figsize=(10, 7))
    
    # Plot Pthreads
    if not df_p_n.empty:
        plt.plot(
            df_p_n['P'], df_p_n['T_mean'],
            label='Pthreads', marker='o', linestyle='-', markersize=8
        )
    
    # Plot MPI
    if not df_m_n.empty:
        plt.plot(
            df_m_n['P'], df_m_n['T_mean'],
            label='MPI', marker='s', linestyle='-', markersize=8
        )
    
    plt.xlabel('Processes / Threads (P)')
    plt.ylabel('Mean Wall Time (s)')
    plt.yscale('log') # Time often scales logarithmically
    plt.title(f'Pthreads vs. MPI Timing (n = {format_n(n_val)})')
    plt.legend(loc='upper right', fontsize='large')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xticks(p_ticks)
    plt.xlim(left=0.5, right=max_p + 0.5)
    
    plot_path = os.path.join(results_dir, f'comparison_timing_n={format_n(n_val)}.{plot_format}')
    try:
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Wrote plot: {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save plot {plot_path}: {e}")

def plot_comparison_speedup(df_p_n, df_m_n, n_val, p_ticks, max_p, results_dir, plot_format):
    """Generates the Speedup vs. P comparison plot."""
    plt.figure(figsize=(10, 7))
    
    # Plot ideal speedup
    plt.plot([1, max_p], [1, max_p], 'k--', label='Ideal Speedup', alpha=0.6)
    
    # Plot Pthreads
    if not df_p_n.empty:
        plt.plot(
            df_p_n['P'], df_p_n['Speedup_S'],
            label='Pthreads', marker='o', linestyle='-', markersize=8
        )
    
    # Plot MPI
    if not df_m_n.empty:
        plt.plot(
            df_m_n['P'], df_m_n['Speedup_S'],
            label='MPI', marker='s', linestyle='-', markersize=8
        )
    
    plt.xlabel('Processes / Threads (P)')
    plt.ylabel('Speedup (S = T1 / TP)')
    plt.title(f'Pthreads vs. MPI Speedup (n = {format_n(n_val)})')
    plt.legend(loc='upper left', fontsize='large')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xticks(p_ticks)
    plt.xlim(left=0.5, right=max_p + 0.5)
    
    plot_path = os.path.join(results_dir, f'comparison_speedup_n={format_n(n_val)}.{plot_format}')
    try:
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Wrote plot: {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save plot {plot_path}: {e}")

def plot_comparison_efficiency(df_p_n, df_m_n, n_val, p_ticks, max_p, results_dir, plot_format):
    """Generates the Efficiency vs. P comparison plot."""
    plt.figure(figsize=(10, 7))
    
    # Plot ideal efficiency
    plt.plot([1, max_p], [1, 1], 'k--', label='Ideal Efficiency', alpha=0.6)
    
    # Plot Pthreads
    if not df_p_n.empty:
        plt.plot(
            df_p_n['P'], df_p_n['Efficiency_E'],
            label='Pthreads', marker='o', linestyle='-', markersize=8
        )
    
    # Plot MPI
    if not df_m_n.empty:
        plt.plot(
            df_m_n['P'], df_m_n['Efficiency_E'],
            label='MPI', marker='s', linestyle='-', markersize=8
        )
    
    plt.xlabel('Processes / Threads (P)')
    plt.ylabel('Efficiency (E = S / P)')
    plt.title(f'Pthreads vs. MPI Efficiency (n = {format_n(n_val)})')
    plt.legend(loc='lower left', fontsize='large')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xticks(p_ticks)
    plt.xlim(left=0.5, right=max_p + 0.5)
    plt.ylim(bottom=0, top=1.1) # Efficiency is 0 to 1
    
    plot_path = os.path.join(results_dir, f'comparison_efficiency_n={format_n(n_val)}.{plot_format}')
    try:
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Wrote plot: {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save plot {plot_path}: {e}")


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description='Compare Pthreads vs MPI performance for MCNT.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Simulation physics args
    parser.add_argument('--C', type=float, default=0.5, help='Total interaction coeff')
    parser.add_argument('--CC', type=float, default=0.1, help='Absorbing component (Cc)')
    parser.add_argument('--H', type=float, default=5.0, help='Slab thickness H')
    parser.add_argument('--seed', type=int, default=12345, help='Random number seed')
    
    # --- NEW: Benchmarking sweep args ---
    parser.add_argument('--n-start', type=int, default=1000000, help='Min particle count n')
    parser.add_argument('--n-max', type=int, default=10000000, help='Max particle count n')
    parser.add_argument('--n-steps', type=int, default=4, help='Number of n values to test (logarithmic scale)')
    
    parser.add_argument('--P-start', type=int, default=1, help='Min threads/processes P')
    parser.add_argument('--P-max', type=int, default=8, help='Max threads/processes P')
    parser.add_argument('--P-step', type=int, default=1, help='Step size for P sweep')
    
    # Benchmarking control args
    parser.add_argument('--trials', type=int, default=3, help='Number of timed trials per (n,P) pair')
    parser.add_argument('--warmup', type=int, default=1, help='Number of warmup runs per (n,P) pair')
    parser.add_argument('--timeout', type=float, default=300.0, help='Timeout per simulation run (seconds)')
    
    # Output args
    parser.add_argument('--results-dir', default=None, help='Directory to save results, CSVs, and plots. (Default: comparison_results_TIMESTAMP)')
    parser.add_argument('--plot-format', type=str, default='png', help='Format for plots (png, pdf, svg)')

    args = parser.parse_args()

    # --- 1. Setup ---
    if args.results_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.results_dir = f"comparison_results_{timestamp}"

    os.makedirs(args.results_dir, exist_ok=True)
    
    # Add file handler to log to results dir
    fh = logging.FileHandler(os.path.join(args.results_dir, 'comparison.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    logger.info("Starting Pthreads vs MPI comparison script...")

    # --- 1b. Compile C Code ---
    try:
        c_code_dir_relative = '../c_code'
        logger.info("Compiling C executables...")
        
        # Run clean() ONCE before any building
        clean_c_code(c_code_dir_relative)
        
        pthread_exe_path = compile_c_code(
            c_code_dir_relative, 'mc_slab_pthreads'
        )
        mpi_exe_path = compile_c_code(
            c_code_dir_relative, 'mc_slab_mpi'
        )
        logger.info("C executables compiled successfully.")
        
    except Exception as e:
        logger.exception(f"Could not compile C code: {e}")
        logger.info(f"Please ensure 'make' is installed and Makefile is present in {c_code_dir_relative}.")
        sys.exit(1)
    
    # --- 1c. Build n_values and p_values from args ---
    if args.n_steps > 1:
        n_values = np.logspace(np.log10(args.n_start), np.log10(args.n_max), args.n_steps, dtype=int)
        n_values = sorted(list(set(n_values))) # Remove duplicates
    else:
        n_values = [args.n_start]
    
    p_values = list(range(args.P_start, args.P_max + 1, args.P_step))
    if 1 not in p_values:
        p_values.insert(0, 1) # Add 1 if not present, required for T1
    p_values = sorted(list(set(p_values))) # Remove duplicates
    
    logger.info(f"Parameters: {vars(args)}")
    logger.info(f"Generated n values: {n_values}")
    logger.info(f"Generated P values: {p_values}")
    
    # Base physics args (positional)
    base_physics_args = [str(args.C), str(args.CC), str(args.H)]
    
    # --- 2. Run Benchmarks ---
    df_pthread_raw = run_benchmark_for_exe(
        pthread_exe_path, 'pthread', base_physics_args, args.seed, # Pass seed
        n_values, p_values, args.trials, args.warmup, args.timeout
    )
    
    df_mpi_raw = run_benchmark_for_exe(
        mpi_exe_path, 'mpi', base_physics_args, args.seed, # Pass seed
        n_values, p_values, args.trials, args.warmup, args.timeout
    )
    
    if df_pthread_raw.empty and df_mpi_raw.empty:
        logger.error("Both benchmarks failed to produce any results. Exiting.")
        sys.exit(1)
    elif df_pthread_raw.empty:
         logger.warning("Pthreads benchmark failed to produce results. Plotting MPI only.")
    elif df_mpi_raw.empty:
         logger.warning("MPI benchmark failed to produce results. Plotting Pthreads only.")


    # --- 3. Analyze Data ---
    logger.info("Calculating speedups and efficiency...")
    df_pthread = pd.DataFrame()
    df_mpi = pd.DataFrame()
    
    if not df_pthread_raw.empty:
        df_pthread = calculate_speedup_and_efficiency(df_pthread_raw)
        if not df_pthread.empty:
            df_pthread.to_csv(os.path.join(args.results_dir, 'summary_pthread.csv'), index=False)

    if not df_mpi_raw.empty:
        df_mpi = calculate_speedup_and_efficiency(df_mpi_raw)
        if not df_mpi.empty:
            df_mpi.to_csv(os.path.join(args.results_dir, 'summary_mpi.csv'), index=False)
    
    if df_pthread.empty and df_mpi.empty:
        logger.error("Speedup/Efficiency calculation failed for all runs. Exiting.")
        sys.exit(1)
        
    # --- 4. Generate Comparison Plots ---
    logger.info("Generating comparison plots...")
    max_p = max(p_values)
    p_ticks = p_values # Use the generated p_values list

    for n in n_values:
        df_p_n = df_pthread[df_pthread['n'] == n].sort_values('P') if not df_pthread.empty else pd.DataFrame()
        df_m_n = df_mpi[df_mpi['n'] == n].sort_values('P') if not df_mpi.empty else pd.DataFrame()
        
        if df_p_n.empty and df_m_n.empty:
            logger.warning(f"Skipping plot for n={format_n(n)}: no data.")
            continue
        
        # Call all three plotting functions
        plot_comparison_timing(
            df_p_n, df_m_n, n, p_ticks, max_p, args.results_dir, args.plot_format
        )
        
        plot_comparison_speedup(
            df_p_n, df_m_n, n, p_ticks, max_p, args.results_dir, args.plot_format
        )
        
        plot_comparison_efficiency(
            df_p_n, df_m_n, n, p_ticks, max_p, args.results_dir, args.plot_format
        )

    logger.info(f"--- Comparison complete. Results are in: {os.path.abspath(args.results_dir)} ---")


if __name__ == "__main__":
    main()