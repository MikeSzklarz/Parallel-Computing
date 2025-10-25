#!/usr/bin/env python3

"""
run-all.py

A helper script to build and run the entire C stencil project
from the root 'stencil/' directory.

Assumes the following structure:
stencil/
├── c_code/
│   ├── make-2d
│   └── (all C files)
├── python_code/
│   └── run-all.py
└── data/ (will be created)

Usage (from 'stencil/' root):
python3 python_code/run-all.py [rows] [cols] [iterations]
"""

import os
import sys
import subprocess

def run_command(cmd_list, working_dir=None):
    """
    Helper to run a command and check for errors.
    Can specify a 'working_dir' to run the command in.
    """
    print(f"\n--- Running: {' '.join(cmd_list)} ---")
    if working_dir:
        print(f"--- in directory: {working_dir} ---")
        
    try:
        # Using text=True for automatic encoding/decoding
        # Pass the 'cwd' (current working directory) argument
        result = subprocess.run(cmd_list, 
                                check=True, 
                                text=True, 
                                capture_output=True, 
                                cwd=working_dir)
        # Print stdout for make, which is useful
        if "make" in cmd_list[0]:
            print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"\n*** ERROR: Command failed with exit code {e.returncode} ***")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n*** ERROR: Command not found: {cmd_list[0]} ***")
        print("Please ensure the command is in your system's PATH.")
        sys.exit(1)

def main():
    # --- 1. Define Paths ---
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Get the parent directory (project root, e.g., 'stencil/')
    project_root = os.path.dirname(script_dir)
    
    # All paths are now relative to the project root
    c_code_dir = os.path.join(project_root, "c_code")
    data_dir = os.path.join(project_root, "data")

    # C executables
    make_2d_exec = os.path.join(c_code_dir, "make-2d")
    stencil_2d_exec = os.path.join(c_code_dir, "stencil-2d")
    
    # 2. Set parameters
    rows = "100"
    cols = "100"
    iters = "500"

    if len(sys.argv) == 4:
        rows = sys.argv[1]
        cols = sys.argv[2]
        iters = sys.argv[3]
    elif len(sys.argv) != 1:
        print(f"Usage: python3 {sys.argv[0]} [rows cols iterations]")
        print(f"If no arguments are given, defaults to 100 100 500.")
        sys.exit(1)

    print(f"Starting stencil run with grid={rows}x{cols}, iterations={iters}")
    
    # 3. Create data directory
    os.makedirs(data_dir, exist_ok=True)
    print(f"Using data directory: {data_dir}")

    # 4. Define filenames (now inside data_dir)
    initial_file = os.path.join(data_dir, f"initial.{rows}x{cols}.dat")
    final_file = os.path.join(data_dir, f"final.{rows}x{cols}x{iters}.dat")
    all_iter_file = os.path.join(data_dir, f"all.{rows}x{cols}x{iters}.raw")

    # 5. Run the commands
    # Run make 'inside' the c_code directory
    print(f"Building C code in {c_code_dir}...")
    
    # Pass 'working_dir=c_code_dir' to run 'make' in the correct folder
    run_command(["make", "clean"], working_dir=c_code_dir)
    run_command(["make"], working_dir=c_code_dir)
    
    # These commands are fine, as they use absolute paths to the executables
    run_command([make_2d_exec, rows, cols, initial_file])
    
    run_command([stencil_2d_exec,
                 iters,
                 initial_file,
                 final_file,
                 all_iter_file])
    
    print("\n--- All steps completed successfully! ---")
    print(f"  Initial file: {initial_file}")
    print(f"  Final file:   {final_file}")
    print(f"  Stack file:   {all_iter_file}")
    print("\nYou can now run make-movie.py on the .raw file.")

if __name__ == "__main__":
    main()

