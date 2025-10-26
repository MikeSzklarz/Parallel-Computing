#!/usr/bin/env python3
"""
display-image.py

Reads a single 2D binary data file and saves it as a 2D PNG image.
Uses argparse for command-line arguments.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys


def read_data(input_file):
    """
    Reads a binary grid file (int32 rows, int32 cols, float64* data)
    and returns a 2D numpy array.
    """
    try:
        with open(input_file, "rb") as f:
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


def create_2d_image(data, output_file):
    """
    Generates and saves a 2D plot from the data.
    """
    try:
        rows, cols = data.shape

        plt.figure(figsize=(10, 8))
        # Use 'jet' colormap to be consistent with the 3D plot and movie
        # vmin/vmax lock the color bar from 0.0 to 1.0
        plt.imshow(data, cmap="jet", vmin=0.0, vmax=1.0)

        plt.colorbar(label="Value")
        plt.title(f"2D Visualization of {os.path.basename(output_file)}")
        plt.xlabel("Columns")
        plt.ylabel("Rows")

        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Successfully saved visualization to {output_file}")

    except Exception as e:
        print(f"An error occurred during plotting: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a 2D image from a stencil .dat file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--in",
        dest="input_file",
        required=True,
        help="Path to the input .dat file (e.g., initial.dat).",
    )
    parser.add_argument(
        "--out",
        dest="output_file",
        required=True,
        help="Path for the output .png image file.",
    )

    args = parser.parse_args()

    data = read_data(args.input_file)

    if data is not None:
        create_2d_image(data, args.output_file)


if __name__ == "__main__":
    main()
