#!/usr/bin/env python3

"""
display_image.py

Reads a single-frame binary data file (like initial.dat or final.dat)
and saves it as a PNG image.

Usage (from 'stencil/' root):
python3 python_code/display_image.py <input_file.dat>
(Outputs <input_file_base>.png in the root 'stencil/' directory)

Example:
python3 python_code/display_image.py data/initial.100x100.dat
(Outputs initial.100x100.png)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def read_single_frame(filename):
    """
    Reads a binary file containing metadata (rows, cols) and one
    frame of double-precision data.
    """
    with open(filename, 'rb') as f:
        # Read metadata (2 ints)
        metadata = np.fromfile(f, dtype=np.int32, count=2)
        rows, cols = metadata[0], metadata[1]
        
        # Read data
        data = np.fromfile(f, dtype=np.float64, count=rows * cols)
        
        if data.size != rows * cols:
            print(f"Error: File size does not match metadata. Expected {rows*cols} doubles, got {data.size}")
            return None
            
        return data.reshape((rows, cols))

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(f"Usage: python3 {sys.argv[0]} <input_file.dat> [output_file.png]")
        sys.exit(1)

    input_file = sys.argv[1]
    
    if len(sys.argv) == 3:
        output_file = sys.argv[2]
    else:
        # Default output name:
        # Get the base filename (e.g., "initial.100x100.dat")
        base_name_with_ext = os.path.basename(input_file)
        # Get the name without extension (e.g., "initial.100x100")
        base_name = os.path.splitext(base_name_with_ext)[0]
        # Save PNG in the *current working directory* (project root)
        output_file = f"{base_name}.png"

    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    # Read the data
    grid = read_single_frame(input_file)
    if grid is None:
        sys.exit(1)

    print(f"Read {grid.shape[0]}x{grid.shape[1]} grid from {input_file}")

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.imshow(grid, cmap='coolwarm', vmin=0.0, vmax=1.0)
    
    plt.colorbar(label='Value')
    plt.title(f"Visualization of {input_file}")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    
    # Save the file
    plt.savefig(output_file)
    print(f"Saved image to {output_file}")

if __name__ == "__main__":
    main()
