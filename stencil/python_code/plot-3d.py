#!/usr/bin/env python3
"""
plot-3d.py

Reads a single 2D binary data file and generates a 3D surface plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import sys

def read_data(input_file):
    """
    Reads a binary grid file (int32 rows, int32 cols, float64* data)
    and returns a 2D numpy array.
    """
    try:
        with open(input_file, 'rb') as f:
            # Read metadata (2 ints)
            metadata = np.fromfile(f, dtype=np.int32, count=2)
            rows, cols = metadata[0], metadata[1]
            
            # Read the rest of the data
            data = np.fromfile(f, dtype=np.float64, count=rows * cols)
            
            if data.size != rows * cols:
                print(f"Error: File {input_file} is corrupted or truncated.")
                return None
                
            return data.reshape((rows, cols))
            
    except FileNotFoundError:
        print(f"Error: File not found at {input_file}")
        return None
    except Exception as e:
        print(f"An error occurred reading {input_file}: {e}")
        return None

def create_3d_plot(data, output_file):
    """
    Generates and saves a 3D surface plot from the 2D data.
    """
    try:
        rows, cols = data.shape
        
        # Create X, Y coordinate grids
        x = np.arange(0, cols, 1)
        y = np.arange(0, rows, 1)
        X, Y = np.meshgrid(x, y)
        
        # Create the 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        # Using 'jet' to match the movie colormap
        ax.plot_surface(X, Y, data, cmap='jet', rstride=1, cstride=1, antialiased=False)
        
        ax.set_title("3D Surface Plot of Final State")
        ax.set_xlabel("Column (X-axis)")
        ax.set_ylabel("Row (Y-axis)")
        ax.set_zlabel("Value")
        
        # Set Z-axis limits to match the data range [0, 1]
        ax.set_zlim(0.0, 1.0)
        
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Successfully saved 3D plot to {output_file}")
        
    except Exception as e:
        print(f"An error occurred during plotting: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Create a 3D surface plot from a stencil .dat file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--in",
        dest="input_file",
        required=True,
        help="Path to the input .dat file (e.g., final.dat)."
    )
    parser.add_argument(
        "--out",
        dest="output_file",
        required=True,
        help="Path for the output .png plot file."
    )
    
    args = parser.parse_args()

    data = read_data(args.input_file)
    
    if data is not None:
        create_3d_plot(data, args.output_file)

if __name__ == "__main__":
    main()
