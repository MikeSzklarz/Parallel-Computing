#!/usr/bin/env python3

"""
sweep_mc_slab.py

Sweeps the slab thickness (H) for the 'mc_slab' C program,
compiles the C code, runs simulations, and plots the results.

This script is designed to be run from the 'python_code/' directory.
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

# Configure default console logging early so the script prints INFO/DEBUG
# Omit the logger name to avoid showing '__main__' in messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Plotting Functions (from pages 37-38) ---

def plot_fractions(csv_path, plot_path, title_str, dpi):
    """
    Generates the 'fractions_vs_H.png' plot.
    """
    try:
        df = pd.read_csv(csv_path)
        plt.figure(figsize=(10, 6))
        
        plt.plot(df['H'], df['reflected'], label='Reflected (r/n)', marker='.')
        plt.plot(df['H'], df['absorbed'], label='Absorbed (b/n)', marker='.')
        plt.plot(df['H'], df['transmitted'], label='Transmitted (t/n)', marker='.')
        
        plt.xlabel('H (slab thickness)')
        plt.ylabel('Fraction')
        plt.title(title_str)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.savefig(plot_path, dpi=dpi)
        plt.close()
        logging.getLogger(__name__).info("final plot         : %s", plot_path)
    except Exception as e:
        logging.getLogger(__name__).exception("Failed to generate fractions plot: %s", e)

def plot_convergence(trace_csv, plot_path, h, C, Cc, N, dpi):
    """
    Generates the 'convergence.png' plot for a specific H.
    """
    try:
        df = pd.read_csv(trace_csv)
        plt.figure(figsize=(10, 6))

        # Plot shaded area for min/max to show variance (optional but nice)
        # For this, we'd need multiple runs. For a single run, just plot the line.
        
        plt.plot(df['k'], df['reflected'], label='Reflected', alpha=0.8)
        plt.plot(df['k'], df['absorbed'], label='Absorbed', alpha=0.8)
        plt.plot(df['k'], df['transmitted'], label='Transmitted', alpha=0.8)
        
        plt.xlabel('Samples k')
        plt.ylabel('Running fraction')
        plt.title(f'Convergence (H={h}, C={C}, Cc={Cc}, N={N})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.savefig(plot_path, dpi=dpi)
        plt.close()
    except Exception as e:
        logging.getLogger(__name__).exception("Failed to generate convergence plot: %s", e)

def plot_runtime_and_scaling(df, plots_dir, dpi=150):
    """Generate runtime-related plots in plots_dir from dataframe df.
    Expects df to contain columns: H, N, runtime_wall_sec, runtime_cpu_sec
    """
    logger = logging.getLogger(__name__)
    os.makedirs(plots_dir, exist_ok=True)

    try:
        # Runtime vs H
        plt.figure(figsize=(8,5))
        if 'runtime_wall_sec' in df.columns:
            plt.plot(df['H'], df['runtime_wall_sec'], marker='o', label='wall')
        if 'runtime_cpu_sec' in df.columns:
            plt.plot(df['H'], df['runtime_cpu_sec'], marker='x', label='cpu')
        plt.xlabel('H')
        plt.ylabel('Runtime (s)')
        plt.title('Runtime vs H')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        out = os.path.join(plots_dir, 'runtime_vs_H.png')
        plt.savefig(out, dpi=dpi)
        plt.close()
        logger.info('Wrote runtime plot: %s', out)

        # Runtime per particle vs H
        if 'runtime_wall_sec' in df.columns and 'N' in df.columns:
            plt.figure(figsize=(8,5))
            per_particle = df['runtime_wall_sec'] / df['N']
            plt.plot(df['H'], per_particle, marker='o')
            plt.xlabel('H')
            plt.ylabel('Runtime per particle (s)')
            plt.title('Runtime per particle vs H')
            plt.grid(True, linestyle='--', alpha=0.6)
            out2 = os.path.join(plots_dir, 'runtime_per_particle_vs_H.png')
            plt.savefig(out2, dpi=dpi)
            plt.close()
            logger.info('Wrote runtime-per-particle plot: %s', out2)

        # Runtime vs N (scaling) if multiple N values present
        if 'N' in df.columns and df['N'].nunique() > 1:
            plt.figure(figsize=(8,5))
            # group by N and take mean runtime
            grouped = df.groupby('N').agg({'runtime_wall_sec':'mean'}).reset_index()
            plt.loglog(grouped['N'], grouped['runtime_wall_sec'], marker='o')
            # fit slope
            m, b = np.polyfit(np.log(grouped['N']), np.log(grouped['runtime_wall_sec']), 1)
            plt.plot(grouped['N'], np.exp(b)*grouped['N']**m, '--', label=f'slope={m:.2f}')
            plt.xlabel('N (particles)')
            plt.ylabel('Runtime (s)')
            plt.title('Runtime vs N (scaling)')
            plt.legend()
            plt.grid(True, which='both', ls='--', alpha=0.6)
            out3 = os.path.join(plots_dir, 'runtime_vs_N.png')
            plt.savefig(out3, dpi=dpi)
            plt.close()
            logger.info('Wrote runtime-vs-N plot: %s', out3)
        else:
            logger.info('Skipping runtime vs N plot (only one N value present)')

    except Exception as e:
        logger.exception('Failed to create runtime/scaling plots: %s', e)

def aggregate_convergence_across_runs(results_dir, plots_dir, dpi=150):
    """Look for per-H trace files under each H_*/data and, if multiple runs exist,
    compute mean/std across runs and plot convergence bands into plots_dir.
    """
    logger = logging.getLogger(__name__)
    os.makedirs(plots_dir, exist_ok=True)
    try:
        h_dirs = sorted(glob.glob(os.path.join(results_dir, 'H_*')))
        for h_dir in h_dirs:
            # trace files are stored directly under H_<h> directories
            trace_files = sorted(glob.glob(os.path.join(h_dir, 'trace*.csv')))
            if len(trace_files) < 2:
                continue
            # read and merge by 'k'
            dfs = [pd.read_csv(p) for p in trace_files]
            merged = pd.concat(dfs, keys=range(len(dfs)), names=['run', 'row']).reset_index(level=0)
            # compute mean/std per k
            stats = merged.groupby('k').agg({'reflected':['mean','std'], 'absorbed':['mean','std'], 'transmitted':['mean','std']})
            ks = stats.index.values
            plt.figure(figsize=(10,6))
            for col, color in [('reflected','C0'), ('absorbed','C1'), ('transmitted','C2')]:
                mean = stats[(col,'mean')]
                std = stats[(col,'std')]
                plt.plot(ks, mean, label=f'{col} mean', color=color)
                plt.fill_between(ks, mean-std, mean+std, alpha=0.25, color=color)
            plt.xlabel('Samples k')
            plt.ylabel('Running fraction')
            h_val = os.path.basename(h_dir).split('H_')[-1]
            plt.title(f'Aggregate convergence (H={h_val})')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            out = os.path.join(plots_dir, f'convergence_aggregate_H_{h_val}.png')
            plt.savefig(out, dpi=dpi)
            plt.close()
            logger.info('Wrote aggregate convergence plot: %s', out)
    except Exception as e:
        logger.exception('Failed to create aggregate convergence plots: %s', e)

# --- C Code Compilation Function ---

def compile_c_code(c_code_dir, exe_name):
    """
    Compiles the C code using 'make' in the specified directory.
    """
    # Get absolute path to the C code directory
    # Assumes this script is in 'python_code' and C code is in 'c_code'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    c_dir_abs = os.path.abspath(os.path.join(script_dir, c_code_dir))
    exe_path_abs = os.path.join(c_dir_abs, exe_name)
    
    logger = logging.getLogger(__name__)
    logger.info("Compiling C code in: %s", c_dir_abs)
    
    # Run 'make clean'
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

    # Run 'make'
    try:
        proc_make = subprocess.run(
            ["make"], 
            cwd=c_dir_abs, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=True  # Raise error if make fails
        )
        logger.debug("make stdout: %s", proc_make.stdout)
        logger.info("C code compiled successfully.")
    except subprocess.CalledProcessError as e:
        logger.error("'make' failed! Return code: %s", e.returncode)
        logger.error("make stdout: %s", getattr(e, 'stdout', ''))
        logger.error("make stderr: %s", getattr(e, 'stderr', ''))
        sys.exit(1) # Exit if compilation fails
    except Exception as e:
        logger.exception("Unexpected error while running 'make': %s", e)
        sys.exit(1)

    # Check if executable exists
    if not os.path.exists(exe_path_abs):
        logger.error("Executable not found after compile: %s", exe_path_abs)
        sys.exit(1)
        
    return exe_path_abs

# --- Main Simulation ---

def main():
    parser = argparse.ArgumentParser(
        description='Sweep H for mc_slab and plot results.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Use arguments from page 24
    parser.add_argument('--exe', default='../c_code/mc_slab', help='Path to mc_slab executable.')
    parser.add_argument('--C', required=True, type=float, help='Total interaction coeff')
    parser.add_argument('--CC', required=True, type=float, help='Absorbing component (Cc)')
    parser.add_argument('--H-min', required=True, type=float, help='Min slab thickness H')
    parser.add_argument('--H-max', required=True, type=float, help='Max slab thickness H')
    parser.add_argument('--H-step', required=True, type=float, help='Step size for H sweep')
    parser.add_argument('--N', required=True, type=int, help='Number of particles (n)')
    parser.add_argument('--seed', type=int, default=42, help='Random number seed')
    parser.add_argument('--timeout', type=float, default=30.0, help='Timeout per simulation run (seconds)')
    parser.add_argument('--trace', action='store_true', help='Enable per-iteration tracing to CSV.')
    parser.add_argument('--trace-every', type=int, default=100, help='Record every m-th iteration.')
    parser.add_argument('--make-convergence-plots', action='store_true', help='When tracing, also render convergence plots per H.')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for output plots')
    parser.add_argument('--title', default=None, help='Custom title for fractions plot')
    
    args = parser.parse_args()

    # Log initial parameters (high verbosity)
    logger = logging.getLogger(__name__)
    logger.info("Starting sweep_mc_slab.py")
    logger.info("Initial parameters: %s", vars(args))

    # --- 1. Compile C Code ---
    # We compile first, but we'll use the args.exe path for the run
    # This allows user to override, but ensures it's built at least once.
    try:
        # Determine the C code directory relative to the --exe path
        c_exe_path = os.path.abspath(args.exe)
        c_code_dir = os.path.dirname(c_exe_path)
        exe_name = os.path.basename(c_exe_path)
        compiled_exe_path = compile_c_code(c_code_dir, exe_name)
        
        # Use the compiled path for the rest of the script
        run_exe_path = compiled_exe_path
        
    except Exception as e:
        logger.exception("Could not compile C code: %s", e)
        logger.info("Please ensure 'make' is installed and Makefile is present.")
        sys.exit(1)

    # --- 2. Setup Results Directory ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Attach file handler to log all experiment activity to experiment.log
    try:
        fh_path = os.path.join(results_dir, "experiment.log")
        fh = logging.FileHandler(fh_path)
        fh.setLevel(logging.DEBUG)
        # Omit logger name in file formatter as well
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(fh)
        logger.info("Experiment log file created at: %s", fh_path)
    except Exception:
        logger.exception("Failed to create experiment.log in results directory")
    # create top-level data and plots folders
    top_data_dir = os.path.join(results_dir, 'data')
    top_plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(top_data_dir, exist_ok=True)
    os.makedirs(top_plots_dir, exist_ok=True)
    
    # --- 3. Run H Sweep ---
    h_values = np.arange(args.H_min, args.H_max + args.H_step, args.H_step)
    results = []

    logger.info("Running sweep for %d H-values...", len(h_values))

    for h in h_values:
        h = round(h, 6) # Avoid floating point issues

        # Create subdirectory for this H-value (keep outputs directly under H_<h>)
        h_dir = os.path.join(results_dir, f"H_{h}")
        os.makedirs(h_dir, exist_ok=True)

        cmd = [
            run_exe_path,
            str(args.C),
            str(args.CC),
            str(h),
            str(args.N),
            '--seed', str(args.seed)
        ]

        if args.trace:
            # place trace files directly under H-specific folder
            trace_csv_path = os.path.join(h_dir, "trace.csv")
            cmd.extend(['--trace-file', trace_csv_path, '--trace-every', str(args.trace_every)])

        try:
            # time the subprocess run (wall + process CPU)
            t0_wall = time.perf_counter()
            t0_cpu = time.process_time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=True
            )
            t1_wall = time.perf_counter()
            t1_cpu = time.process_time()
            runtime_wall = t1_wall - t0_wall
            runtime_cpu = t1_cpu - t0_cpu

            # Parse stdout (e.g., "0.30490000 0.45840000 0.23670000")
            parts = result.stdout.strip().split()
            if len(parts) != 3:
                raise ValueError("Unexpected output: %s" % result.stdout.strip())

            r_frac = float(parts[0])
            b_frac = float(parts[1])
            t_frac = float(parts[2])

            results.append({
                'H': h,
                'N': args.N,
                'C': args.C,
                'CC': args.CC,
                'seed': args.seed,
                'reflected': r_frac,
                'absorbed': b_frac,
                'transmitted': t_frac,
                'runtime_wall_sec': runtime_wall,
                'runtime_cpu_sec': runtime_cpu,
                'timestamp': datetime.datetime.now().isoformat()
            })
            logger.info("H=%.4f -> reflected=%.8f absorbed=%.8f transmitted=%.8f (wall=%.4fs cpu=%.4fs)", h, r_frac, b_frac, t_frac, runtime_wall, runtime_cpu)

            # Make convergence plot for this H if requested
            if args.trace and args.make_convergence_plots:
                conv_plot_path = os.path.join(h_dir, "convergence.png")
                plot_convergence(
                    trace_csv_path, conv_plot_path,
                    h, args.C, args.CC, args.N, args.dpi
                )
        except subprocess.TimeoutExpired:
            logger.error("H = %s timed out after %ss.", h, args.timeout)
        except subprocess.CalledProcessError as e:
            logger.error("H = %s failed with return code %s", h, e.returncode)
            logger.error("stdout: %s", getattr(e, 'stdout', ''))
            logger.error("stderr: %s", getattr(e, 'stderr', ''))
        except Exception as e:
            logger.exception("H = %s failed: %s", h, e)

    # --- 4. Save Final Results and Plot ---
    if not results:
        logger.error("No results were collected. Exiting.")
        sys.exit(1)

    # Save main CSV into data folder
    df_final = pd.DataFrame(results)
    csv_out_path = os.path.join(top_data_dir, "fractions_vs_H.csv")
    df_final.to_csv(csv_out_path, index=False)
    logger.info("final fractions CSV: %s", csv_out_path)

    # Save final plot into plots folder
    plot_out_path = os.path.join(top_plots_dir, "fractions_vs_H.png")
    if args.title:
        title = args.title
    else:
        title = f'Fractions vs H (C={args.C}, Cc={args.CC}, N={args.N})'
    plot_fractions(csv_out_path, plot_out_path, title, args.dpi)

    # Additional runtime/scaling plots
    plot_runtime_and_scaling(df_final, top_plots_dir, dpi=args.dpi)

    # Aggregate convergence across repeats, if multiple trace files per H exist
    aggregate_convergence_across_runs(results_dir, top_plots_dir, dpi=args.dpi)

    logger.info("results dir        : %s", os.path.abspath(results_dir))
    if args.trace and args.make_convergence_plots:
        logger.info("per-H convergence plots are under each H_* folders (convergence.png files).")

if __name__ == "__main__":
    main()