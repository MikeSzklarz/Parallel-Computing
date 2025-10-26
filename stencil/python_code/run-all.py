#!/usr/bin/env python3
"""
run-all.py

Master script to build, run, and visualize the stencil project.
This script coordinates all C and Python steps.
"""

import os
import subprocess
import sys
import argparse
import time
import re  # <-- ADDED: For parsing
import numpy as np  # <-- ADDED: For stats


def run_command(cmd_list, cwd=".", shell=False):
    """
    Runs a shell command, prints its output in real-time,
    and returns the return code, stdout, and stderr.
    """
    print(f"--- Running: {' '.join(cmd_list)} (in {cwd}) ---")

    try:
        # Use Popen to capture stdout/stderr
        proc = subprocess.Popen(
            cmd_list,
            cwd=cwd,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout_lines = []
        stderr_lines = []

        # Read stdout line by line
        for line in proc.stdout:
            print(line, end="")
            stdout_lines.append(line)

        # Read stderr line by line
        for line in proc.stderr:
            print(line, end="", file=sys.stderr)
            stderr_lines.append(line)

        proc.wait()

        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)

        if proc.returncode != 0:
            print(f"Error executing command: {' '.join(cmd_list)}", file=sys.stderr)
            print(f"Return Code: {proc.returncode}", file=sys.stderr)
            print(f"Stderr: \n{stderr}", file=sys.stderr)
        else:
            print("--- Command successful ---")

        return proc.returncode, stdout, stderr

    except Exception as e:
        print(f"Failed to run command: {e}", file=sys.stderr)
        return -1, "", str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Build, run, and visualize the 2D stencil project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rows", type=int, default=100, help="Number of rows for the grid."
    )
    parser.add_argument(
        "--cols", type=int, default=100, help="Number of columns for the grid."
    )
    parser.add_argument(
        "--iters", type=int, default=500, help="Number of stencil iterations."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of computation trials to run for averaging.",
    )  

    args = parser.parse_args()

    rows = args.rows
    cols = args.cols
    iters = args.iters
    trials = args.trials  

    # --- Paths ---
    # Assumes this script is in 'python_code' and C code is in 'c_code'
    c_code_dir = "../c_code"
    data_dir = "../data"

    # Create unique file names
    base_name = f"{rows}x{cols}x{iters}"
    initial_file = f"{data_dir}/initial.{rows}x{cols}.dat"
    final_file = f"{data_dir}/final.{base_name}.dat"
    raw_stack_file = f"{data_dir}/all.{base_name}.raw"
    movie_file = f"{data_dir}/stencil.{base_name}.mp4"

    # Image files (will be created in the CWD, which is python_code)
    img_initial_file = f"{data_dir}/initial.{rows}x{cols}.png"
    img_final_file = f"{data_dir}/final.{base_name}.png"
    img_3d_file = f"{data_dir}/final.{base_name}.3d.png"

    # Start timer
    total_start_time = time.time()

    # --- 1. Create Data Directory ---
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Creating data directory: {data_dir}\n")

    # --- 2. Build C code ---
    run_command(["make", "clean"], cwd=c_code_dir)
    run_command(["make"], cwd=c_code_dir)

    # --- 3. Create initial grid ---
    make_cmd = [f"{c_code_dir}/make-2d", str(rows), str(cols), initial_file]
    run_command(make_cmd, cwd=".")
    print(f"Successfully created: {initial_file}\n")

    # --- 4. Run stencil simulation ---
    stencil_cmd = [
        f"{c_code_dir}/stencil-2d",
        str(iters),
        initial_file,
        final_file,
        raw_stack_file,
    ]

    # --- MODIFIED: Run multiple trials ---
    computation_times = []
    time_parser = re.compile(r"^COMP_TIME:\s*([\d\.]+)", re.M)

    print(f"--- Running {trials} stencil trials... ---")
    for i in range(trials):
        print(f"\n--- Running Trial {i+1}/{trials} ---")
        return_code, stdout, stderr = run_command(stencil_cmd, cwd=".")

        if return_code != 0:
            print(f"Error during trial {i+1}, aborting.")
            sys.exit(1)

        match = time_parser.search(stdout)
        if match:
            time_val = float(match.group(1))
            computation_times.append(time_val)
            print(f"Trial {i+1} time: {time_val:.6f} s")
        else:
            print(f"Error: Could not parse computation time for trial {i+1}.")

    if not computation_times:
        print("Error: No computation times were recorded.")
        sys.exit(1)

    # Calculate stats
    avg_time = np.mean(computation_times)
    min_time = np.min(computation_times)
    max_time = np.max(computation_times)

    print("\n--- Stencil Performance Summary ---")
    print(f"  Trials:     {trials}")
    print(f"  Avg Time:   {avg_time:.6f} s")
    print(f"  Min Time:   {min_time:.6f} s (Best)")
    print(f"  Max Time:   {max_time:.6f} s (Worst)")
    print(f"  All Times:  {[float(f'{t:.6f}') for t in computation_times]}")
    # --- End of MODIFIED section ---

    print(f"\nSuccessfully created: {final_file}")
    print(f"Successfully created: {raw_stack_file}\n")

    # --- 5. Create 2D image of initial state ---
    img_initial_cmd = [
        "python3",
        "./display-image.py",
        "--in",
        initial_file,
        "--out",
        img_initial_file,
    ]
    run_command(img_initial_cmd, cwd=".")

    # --- 6. Create 2D image of final state ---
    img_final_cmd = [
        "python3",
        "./display-image.py",
        "--in",
        final_file,
        "--out",
        img_final_file,
    ]
    run_command(img_final_cmd, cwd=".")

    # --- 7. Create 3D plot of final state ---
    plot3d_cmd = ["python3", "./plot-3d.py", "--in", final_file, "--out", img_3d_file]
    run_command(plot3d_cmd, cwd=".")

    # --- 8. Create movie ---
    movie_cmd = [
        "python3",
        "./make-movie.py",
        "--in",
        raw_stack_file,
        "--out",
        movie_file,
    ]
    run_command(movie_cmd, cwd=".")
    print(f"Successfully created: {movie_file}\n")

    # --- 9. Final Summary ---
    total_time = time.time() - total_start_time
    print(f"--- All steps completed successfully! ---")
    print(f"Grid Size:        {rows}x{cols}")
    print(f"Iterations:       {iters}")
    print(f"Avg Comp. Time:   {avg_time:.6f} s ({trials} trials)")  # <-- MODIFIED
    print(f"Total Run Time:   {total_time:.2f} seconds")
    print("\n--- Output Files ---")
    print(f"Initial grid:     {initial_file}")
    print(f"Final grid:       {final_file}")
    print(f"Raw stack file:   {raw_stack_file}")
    print(f"Initial PNG:      {img_initial_file}")
    print(f"Final PNG:        {img_final_file}")
    print(f"Final 3D Plot:    {img_3d_file}")
    print(f"Output movie:     {movie_file}")


if __name__ == "__main__":
    main()
