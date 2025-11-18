#!/usr/bin/env python3

"""
check_consistency.py

Runs consistency checks comparing the serial mc_slab implementation against
both parallel (pthreads and MPI) implementations.

This script implements and expands on the requirements from pages 7-8 of
MCNT-Parallel.pdf.

It now auto-compiles all C-code dependencies and provides
detailed statistics (mean, std) for all simulation outputs.
"""

import argparse
import os
import sys
import subprocess
import pandas as pd
import numpy as np
import datetime
import logging
import glob
import stat # For file permissions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    c_dir_abs = os.path.abspath(os.path.join(script_dir, c_code_dir))
    exe_path_abs = os.path.join(c_dir_abs, target_name)
    
    logger = logging.getLogger(__name__) 
    logger.info("Compiling C target '%s' in: %s", target_name, c_dir_abs)
    
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

    if not os.path.exists(exe_path_abs):
        logger.error("Executable not found after compile: %s", exe_path_abs)
        sys.exit(1)
        
    try:
        os.chmod(exe_path_abs, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        logger.info("Set execute permissions on %s", exe_path_abs)
    except Exception as e:
        logger.error("Failed to set execute permissions on %s: %s", exe_path_abs, e)
        sys.exit(1)
            
    return exe_path_abs

# --- Simulation Functions ---

def build_command(args, exe_path, mode, n, P):
    """Builds the subprocess command array for serial, pthreads, or mpi."""
    
    # Base command: ./exe C Cc H n
    base_cmd = [
        str(args.C),
        str(args.CC),
        str(args.H),
        str(n),
        '--seed', str(args.seed) # Use same seed
    ]

    # If tracing, create a unique base path for this run
    trace_base = None
    if args.trace_file:
        trace_base = f"{args.trace_file}_n{n}_P{P}_{mode}"
        
    if mode == 'serial':
        cmd = [exe_path] + base_cmd
        if trace_base:
            # Note: Serial trace doesn't get a rank/thread ID
            cmd.extend(['--trace-file', f"{trace_base}.csv", '--trace-every', str(args.trace_every)])
        return cmd

    elif mode == 'pthreads':
        # ./exe C Cc H n [opts] T
        cmd = [exe_path] + base_cmd
        if trace_base:
            cmd.extend(['--trace-file', trace_base, '--trace-every', str(args.trace_every)])
        cmd.append(str(P))
        return cmd
        
    elif mode == 'mpi':
        # mpirun -np P ./exe C Cc H n [opts]
        cmd = [args.mpi_executable, '-np', str(P), exe_path] + base_cmd
        if trace_base:
            cmd.extend(['--trace-file', trace_base, '--trace-every', str(args.trace_every)])
        return cmd
        
    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_and_parse(cmd):
    """
    Runs a simulation command and parses the "r b t" stdout.
    Returns (r, b, t) tuple or (None, None, None) on failure.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120.0, # 2 min timeout
            check=True
        )
        
        parts = result.stdout.strip().split()
        if len(parts) != 3:
            logger.warning("Unexpected stdout: '%s' from CMD: %s", result.stdout.strip(), " ".join(cmd))
            return None, None, None
            
        r = float(parts[0])
        b = float(parts[1])
        t = float(parts[2])
        return r, b, t

    except subprocess.TimeoutExpired:
        logger.error("Run timed out. CMD: %s", " ".join(cmd))
        return None, None, None
    except subprocess.CalledProcessError as e:
        logger.error("Run failed (code %d). CMD: %s", e.returncode, " ".join(cmd))
        logger.error("STDERR: %s", e.stderr.strip())
        return None, None, None
    except Exception as e:
        logger.exception("An error occurred running command: %s", " ".join(cmd))
        return None, None, None

def main():
    parser = argparse.ArgumentParser(
        description='Check consistency between serial, pthreads, and mpi mc_slab.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Executable args
    parser.add_argument('--mpi-executable', default='mpirun', help='Path to mpirun')

    # Simulation physics args
    parser.add_argument('--C', type=float, default=1.0, help='Total interaction coeff')
    parser.add_argument('--CC', type=float, default=0.3, help='Absorbing component (Cc)')
    parser.add_argument('--H', type=float, default=10.0, help='Slab thickness H')
    parser.add_argument('--seed', type=int, default=12345, help='Random number seed')

    # Sweep args (n)
    parser.add_argument('--n-start', type=int, default=100000, help='Min particle count n')
    parser.add_argument('--n-max', type=int, default=100000000, help='Max particle count n')
    parser.add_argument('--n-steps', type=int, default=5, help='Number of n values to test (logarithmic scale)')
    
    # Sweep args (P)
    parser.add_argument('--P-start', type=int, default=1, help='Min threads/processes P')
    parser.add_argument('--P-max', type=int, default=8, help='Max threads/processes P')
    parser.add_argument('--P-step', type=int, default=1, help='Step size for P sweep')
    
    # Benchmarking control
    parser.add_argument('--trials', type=int, default=3, help='Number of trials per (n, P) config')
    parser.add_argument('--abs-threshold', type=float, default=0.001, help='Absolute difference threshold for PASS/FAIL')

    # Output args
    parser.add_argument('--results-dir', default=None, help='Directory to save results. (Default: consistency_results_all_timestamp)')
    
    # Tracing (optional)
    parser.add_argument('--trace-file', default=None, help='Base path for trace files (e.g., "trace/check").')
    parser.add_argument('--trace-every', type=int, default=10000, help='Record every m-th iteration.')
    
    # User Request: Cleanup
    parser.add_argument('--no-cleanup', action='store_true', default=False, help='If set, do *not* delete intermediate trace files.')
    
    args = parser.parse_args()

    # --- 1. Setup ---
    if args.results_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.results_dir = f"consistency_results_all_{timestamp}"
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set up file logging
    fh = logging.FileHandler(os.path.join(args.results_dir, "consistency_check.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    logger.info("Starting comprehensive consistency check.")
    
    # --- 1b. Compile C Code ---
    try:
        c_code_dir_relative = '../c_code'
        logger.info("Compiling C executables...")
        
        clean_c_code(c_code_dir_relative)
        
        serial_exe_path = compile_c_code(
            c_code_dir_relative, 'mc_slab'
        )
        pthreads_exe_path = compile_c_code(
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
    
    logger.info("Parameters: %s", vars(args))

    # Build task list
    if args.n_steps > 1:
        n_values = np.logspace(np.log10(args.n_start), np.log10(args.n_max), args.n_steps, dtype=int)
    else:
        n_values = [args.n_start]
        
    p_values = list(range(args.P_start, args.P_max + 1, args.P_step))
    if 1 not in p_values:
        p_values.insert(0, 1) # Add 1 if not present
    p_values = sorted(list(set(p_values))) # Remove duplicates
    
    logger.info(f"Generated n values: {n_values}")
    logger.info(f"Generated P values: {p_values}")

    tasks = [(n, p) for n in n_values for p in p_values]
    
    # --- 2. Run Simulations ---
    raw_results = []
    total_tasks = len(tasks)
    
    logger.info("Starting %d simulation configurations, %d trials each...", total_tasks, args.trials)

    for i, (n, P) in enumerate(tasks):
        logger.info("[%d/%d] Running n=%d, P=%d...", i + 1, total_tasks, n, P)
        
        for t in range(args.trials):
            # Run Serial
            serial_cmd = build_command(args, serial_exe_path, 'serial', n, P)
            r_s, b_s, t_s = run_and_parse(serial_cmd)
            
            # Run Pthreads
            pthreads_cmd = build_command(args, pthreads_exe_path, 'pthreads', n, P)
            r_p, b_p, t_p = run_and_parse(pthreads_cmd)

            # Run MPI
            mpi_cmd = build_command(args, mpi_exe_path, 'mpi', n, P)
            r_m, b_m, t_m = run_and_parse(mpi_cmd)
            
            # Check for any failures
            if r_s is None or r_p is None or r_m is None:
                logger.error("Failed run for (n=%d, P=%d, t=%d). Skipping trial.", n, P, t)
                continue
                
            # Compare Serial vs Pthreads
            dr_sp = abs(r_s - r_p)
            db_sp = abs(b_s - b_p)
            dt_sp = abs(t_s - t_p)
            max_abs_diff_sp = max(dr_sp, db_sp, dt_sp)
            passed_sp = max_abs_diff_sp <= args.abs_threshold
            
            # Compare Serial vs MPI
            dr_sm = abs(r_s - r_m)
            db_sm = abs(b_s - b_m)
            dt_sm = abs(t_s - t_m)
            max_abs_diff_sm = max(dr_sm, db_sm, dt_sm)
            passed_sm = max_abs_diff_sm <= args.abs_threshold
            
            raw_results.append({
                'n': n,
                'P': P,
                'trial': t,
                'r_serial': r_s, 'b_serial': b_s, 't_serial': t_s,
                'r_pthreads': r_p, 'b_pthreads': b_p, 't_pthreads': t_p,
                'r_mpi': r_m, 'b_mpi': b_m, 't_mpi': t_m,
                'dr_sp': dr_sp, 'db_sp': db_sp, 'dt_sp': dt_sp,
                'max_abs_diff_sp': max_abs_diff_sp, 'passed_sp': passed_sp,
                'dr_sm': dr_sm, 'db_sm': db_sm, 'dt_sm': dt_sm,
                'max_abs_diff_sm': max_abs_diff_sm, 'passed_sm': passed_sm
            })
            
            logger.debug(f"n={n} P={P} t={t} | S_vs_P_diff={max_abs_diff_sp:.8f} ({"PASS" if passed_sp else "FAIL"}) "
                         f"| S_vs_M_diff={max_abs_diff_sm:.8f} ({"PASS" if passed_sm else "FAIL"})")

    if not raw_results:
        logger.error("No successful simulation pairs. Exiting.")
        sys.exit(1)

    # Save raw CSV
    df_raw = pd.DataFrame(raw_results)
    raw_csv_path = os.path.join(args.results_dir, "raw_runs.csv")
    df_raw.to_csv(raw_csv_path, index=False)
    logger.info("Wrote raw consistency data to %s", raw_csv_path)

    # --- 3. Process and Summarize Results ---
    logger.info("Processing results...")
    
    # Calculate mean diffs per (n, P)
    df_summary = df_raw.groupby(['n', 'P']).agg(
        # Trial info
        trials=('trial', 'count'),
        
        # --- NEW: Serial results stats ---
        r_serial_mean=('r_serial', 'mean'), r_serial_std=('r_serial', 'std'),
        b_serial_mean=('b_serial', 'mean'), b_serial_std=('b_serial', 'std'),
        t_serial_mean=('t_serial', 'mean'), t_serial_std=('t_serial', 'std'),

        # --- NEW: Pthreads results stats ---
        r_pthreads_mean=('r_pthreads', 'mean'), r_pthreads_std=('r_pthreads', 'std'),
        b_pthreads_mean=('b_pthreads', 'mean'), b_pthreads_std=('b_pthreads', 'std'),
        t_pthreads_mean=('t_pthreads', 'mean'), t_pthreads_std=('t_pthreads', 'std'),
        
        # --- NEW: MPI results stats ---
        r_mpi_mean=('r_mpi', 'mean'), r_mpi_std=('r_mpi', 'std'),
        b_mpi_mean=('b_mpi', 'mean'), b_mpi_std=('b_mpi', 'std'),
        t_mpi_mean=('t_mpi', 'mean'), t_mpi_std=('t_mpi', 'std'),

        # Serial vs Pthreads consistency
        max_abs_diff_sp_mean=('max_abs_diff_sp', 'mean'),
        max_abs_diff_sp_max=('max_abs_diff_sp', 'max'),
        pass_count_sp=('passed_sp', 'sum'),
        
        # Serial vs MPI consistency
        max_abs_diff_sm_mean=('max_abs_diff_sm', 'mean'),
        max_abs_diff_sm_max=('max_abs_diff_sm', 'max'),
        pass_count_sm=('passed_sm', 'sum')
    ).reset_index()
    
    df_summary['pass_rate_sp'] = df_summary['pass_count_sp'] / df_summary['trials']
    df_summary['all_passed_sp'] = df_summary['pass_rate_sp'] == 1.0
    
    df_summary['pass_rate_sm'] = df_summary['pass_count_sm'] / df_summary['trials']
    df_summary['all_passed_sm'] = df_summary['pass_rate_sm'] == 1.0
    
    summary_csv_path = os.path.join(args.results_dir, "summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    logger.info("Wrote summary data to %s", summary_csv_path)
    # --- NEW: Log message confirming new stats ---
    logger.info("Summary includes mean/std for all simulation results (r,b,t) and consistency checks.")


    # --- 4. Print Final Report to Console ---
    total_configs = len(df_summary)
    
    total_pass_sp = df_summary['all_passed_sp'].sum()
    total_fail_sp = total_configs - total_pass_sp
    
    total_pass_sm = df_summary['all_passed_sm'].sum()
    total_fail_sm = total_configs - total_pass_sm
    
    print("\n" + "="*40)
    print("=== Comprehensive Consistency Summary ===")
    print(f"Threshold: abs_diff <= {args.abs_threshold}")
    print(f"Configs tested (n,P): {total_configs}")
    print("-" * 20)
    print("  Serial vs Pthreads")
    print(f"    PASS (all trials): {total_pass_sp}")
    print(f"    FAIL (any trial):  {total_fail_sp}")
    print("-" * 20)
    print("  Serial vs MPI")
    print(f"    PASS (all trials): {total_pass_sm}")
    print(f"    FAIL (any trial):  {total_fail_sm}")
    print("="*40)
    
    if total_fail_sp > 0:
        logger.warning("Failing (Serial vs Pthreads) configurations (worst first):")
        failed_configs = df_summary[~df_summary['all_passed_sp']].sort_values('max_abs_diff_sp_max', ascending=False)
        for _, row in failed_configs.iterrows():
            logger.warning(f"  n=%-10d P=%-3d | max_diff=%.8f | passed {row['pass_count_sp']}/{row['trials']}", 
                           int(row['n']), int(row['P']), row['max_abs_diff_sp_max'])

    if total_fail_sm > 0:
        logger.warning("Failing (Serial vs MPI) configurations (worst first):")
        failed_configs = df_summary[~df_summary['all_passed_sm']].sort_values('max_abs_diff_sm_max', ascending=False)
        for _, row in failed_configs.iterrows():
            logger.warning(f"  n=%-10d P=%-3d | max_diff=%.8f | passed {row['pass_count_sm']}/{row['trials']}", 
                           int(row['n']), int(row['P']), row['max_abs_diff_sm_max'])

    # --- 5. Cleanup ---
    if args.trace_file and not args.no_cleanup:
        logger.info("Cleaning up intermediate trace files...")
        trace_dir = os.path.dirname(args.trace_file) or "."
        trace_base_name = os.path.basename(args.trace_file)
        
        pattern = os.path.join(trace_dir, f"{trace_base_name}*_n*_P*.*csv")
        
        cleaned_count = 0
        for f in glob.glob(pattern):
            try:
                os.remove(f)
                cleaned_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete trace file %s: %s", f, e)
        logger.info("Cleaned up %d trace files.", cleaned_count)

    logger.info("Consistency check complete. Results are in: %s", os.path.abspath(args.results_dir))

if __name__ == "__main__":
    main()