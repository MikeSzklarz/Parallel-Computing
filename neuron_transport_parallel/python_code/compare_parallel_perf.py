#!/usr/bin/env python3
"""
compare_parallel_perf.py

[Above and Beyond]

Runs both a Pthreads and an MPI executable for the same set of parameters
and plots their speedup curves on the same graphs for a direct
head-to-head performance comparison.

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

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Matplotlib style
plt.style.use('ggplot')

# --- Helper Functions ---

def parse_n_values(n_str):
    """Parses a comma-separated string of n values, allowing 'M' (millions)."""
    vals = []
    for part in n_str.split(','):
        part = part.strip().upper()
        if part.endswith('M'):
            val = int(float(part[:-1]) * 1_000_000)
        elif part.endswith('K'):
            val = int(float(part[:-1]) * 1_000)
        else:
            val = int(part)
        vals.append(val)
    return sorted(list(set(vals)))

def parse_p_values(p_str):
    """Parses a comma-separated string of P values."""
    return sorted(list(set(int(p.strip()) for p in p_str.split(','))))

def format_n(n):
    """Formats large n values for plot labels."""
    if n >= 1_000_000:
        return f'{n // 1_000_000}M'
    if n >= 1_000:
        return f'{n // 1_000}K'
    return str(n)

def run_simulation(cmd_args, timeout):
    """
    Runs a single simulation subprocess and returns its wall time.
    """
    try:
        t_start = time.perf_counter()
        subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True
        )
        t_end = time.perf_counter()
        return (t_end - t_start)
    except Exception as e:
        logger.warning(f"Run failed for {' '.join(cmd_args)}. Error: {e}")
        return None

def create_run_command(exe, run_type, base_sim_args, n, P):
    """Builds the command list for subprocess."""
    # Base command: [exe, C, Cc, H, n, --seed, s]
    cmd = [exe] + base_sim_args + [str(n), '--seed', str(base_sim_args[-1])]
    
    if run_type == 'pthread':
        # ./mc_slab_pthreads C Cc H n --seed s T
        return cmd + [str(P)]
    elif run_type == 'mpi':
        # mpirun -np P ./mc_slab_mpi C Cc H n --seed s
        return ['mpirun', '-np', str(P)] + cmd
    else:
        raise ValueError(f"Invalid run_type: {run_type}")

def run_benchmark_for_exe(exe, run_type, base_sim_args, n_values, p_values, trials, warmup, timeout):
    """Runs the full benchmark for one executable."""
    
    raw_results = []
    param_combinations = list(itertools.product(n_values, p_values))
    total_jobs = len(param_combinations)
    
    logger.info(f"--- Benchmarking {run_type.upper()} ({exe}) ---")
    
    job_idx = 1
    for n, P in param_combinations:
        logger.info(f"  [Job {job_idx}/{total_jobs}]: n={format_n(n)}, P={P}")
        cmd = create_run_command(exe, run_type, base_sim_args, n, P)
        
        # Warmup
        for _ in range(warmup):
            run_simulation(cmd, timeout)
            
        # Timed trials
        trial_times = []
        for t in range(trials):
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
        
        job_idx += 1
        
    return pd.DataFrame(raw_results)

def calculate_speedup(df, p_values):
    """Calculates speedup from a mean-time dataframe."""
    
    df_t1 = df[df['P'] == 1][['n', 'T_mean']].rename(
        columns={'T_mean': 'T1'}
    )
    
    if df_t1.empty:
        logger.error(f"No P=1 data found for {df['run_type'].iloc[0]}. Cannot calculate speedup.")
        min_p = min(p_values)
        if min_p > 1:
            logger.warning(f"Using P={min_p} as baseline T1. Speedup will be relative.")
            df_t1 = df[df['P'] == min_p][['n', 'T_mean']].rename(
                columns={'T_mean': 'T1'}
            )
            if df_t1.empty:
                 return pd.DataFrame()
        else:
            return pd.DataFrame()

    df = pd.merge(df, df_t1, on='n', how='left')
    df['Speedup_S'] = df['T1'] / df['T_mean']
    return df

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description='Compare Pthreads vs MPI performance for MCNT.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Executable args
    parser.add_argument('--pthread-exe', required=True, help='Path to Pthreads executable')
    parser.add_argument('--mpi-exe', required=True, help='Path to MPI executable')
    
    # Simulation physics args
    parser.add_argument('--C', required=True, type=float, help='Total interaction coeff')
    parser.add_argument('--CC', required=True, type=float, help='Absorbing component (Cc)')
    parser.add_argument('--H', required=True, type=float, help='Slab thickness H')
    parser.add_argument('--seed', type=int, default=12345, help='Random number seed')
    
    # Benchmarking sweep args
    parser.add_argument('--n-values', required=True, type=parse_n_values, help="Comma-separated list of n values (e.g., '1M,10M')")
    parser.add_argument('--P-values', required=True, type=parse_p_values, help="Comma-separated list of P values (e.g., '1,2,4,8,12,16')")
    
    # Benchmarking control args
    parser.add_argument('--trials', type=int, default=3, help='Number of timed trials per (n,P) pair')
    parser.add_argument('--warmup', type=int, default=1, help='Number of warmup runs per (n,P) pair')
    parser.add_argument('--timeout', type=float, default=300.0, help='Timeout per simulation run (seconds)')
    
    # Output args
    parser.add_argument('--results-dir', default='comparison_results', help='Directory to save results, CSVs, and plots')
    parser.add_argument('--plot-format', type=str, default='png', help='Format for plots (png, pdf, svg)')

    args = parser.parse_args()

    # --- 1. Setup ---
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Add file handler to log to results dir
    fh = logging.FileHandler(os.path.join(args.results_dir, 'comparison.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    logger.info("Starting Pthreads vs MPI comparison script...")
    logger.info(f"Parameters: {vars(args)}")
    
    # Base sim args: [C, Cc, H, --seed, s]
    base_sim_args = [str(args.C), str(args.CC), str(args.H), str(args.seed)]
    
    # --- 2. Run Benchmarks ---
    df_pthread_raw = run_benchmark_for_exe(
        args.pthread_exe, 'pthread', base_sim_args,
        args.n_values, args.P_values, args.trials, args.warmup, args.timeout
    )
    
    df_mpi_raw = run_benchmark_for_exe(
        args.mpi_exe, 'mpi', base_sim_args,
        args.n_values, args.P_values, args.trials, args.warmup, args.timeout
    )
    
    if df_pthread_raw.empty or df_mpi_raw.empty:
        logger.error("One or both benchmarks failed to produce results. Exiting.")
        sys.exit(1)

    # --- 3. Analyze Data ---
    logger.info("Calculating speedups...")
    df_pthread = calculate_speedup(df_pthread_raw, args.P_values)
    df_mpi = calculate_speedup(df_mpi_raw, args.P_values)
    
    if df_pthread.empty or df_mpi.empty:
        logger.error("Speedup calculation failed for one or both runs. Exiting.")
        sys.exit(1)
        
    # Save CSVs
    df_pthread.to_csv(os.path.join(args.results_dir, 'summary_pthread.csv'), index=False)
    df_mpi.to_csv(os.path.join(args.results_dir, 'summary_mpi.csv'), index=False)
    
    # --- 4. Generate Comparison Plots ---
    logger.info("Generating comparison plots...")
    max_p = max(args.P_values)

    for n in args.n_values:
        df_p_n = df_pthread[df_pthread['n'] == n].sort_values('P')
        df_m_n = df_mpi[df_mpi['n'] == n].sort_values('P')
        
        if df_p_n.empty or df_m_n.empty:
            logger.warning(f"Skipping plot for n={format_n(n)}: missing data.")
            continue
            
        plt.figure(figsize=(10, 7))
        
        # Plot ideal speedup
        plt.plot([1, max_p], [1, max_p], 'k--', label='Ideal Speedup')
        
        # Plot Pthreads
        plt.plot(
            df_p_n['P'], df_p_n['Speedup_S'],
            label='Pthreads', marker='o', linestyle='-'
        )
        
        # Plot MPI
        plt.plot(
            df_m_n['P'], df_m_n['Speedup_S'],
            label='MPI', marker='s', linestyle='-'
        )
        
        plt.xlabel('Processes / Threads (P)')
        plt.ylabel('Speedup (S = T1 / TP)')
        plt.title(f'Pthreads vs. MPI Speedup (n = {format_n(n)})')
        plt.legend(loc='upper left')
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.xticks(args.P_values)
        
        plot_path = os.path.join(args.results_dir, f'comparison_speedup_n={format_n(n)}.{args.plot_format}')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Wrote plot: {plot_path}")

    logger.info("--- Comparison complete. ---")


if __name__ == "__main__":
    main()