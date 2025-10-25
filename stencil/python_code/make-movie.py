#!/usr/bin/env python3

"""
make-movie.py

Reads a multi-frame binary data stack (like all.raw) and creates
an MP4 movie from the frames.

Matches the file format:
- int rows, int cols (metadata, written ONCE)
- N+1 frames of (rows * cols) doubles

Example:
python3 python_code/make-movie.py --in data/all.100x100x500.dat --out data/movie.mp4
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import os
import sys
from tqdm import tqdm

# We need a non-GUI backend for matplotlib to work in scripts
# and to avoid memory leaks when creating many plots.
import matplotlib
matplotlib.use('Agg')

def read_stack_file(filename):
    """
    Reads the 'all iterations' stack file.
    Returns (rows, cols, data_stack)
    """
    with open(filename, 'rb') as f:
        # Read metadata (2 ints = 8 bytes)
        metadata = np.fromfile(f, dtype=np.int32, count=2)
        rows, cols = metadata[0], metadata[1]
        
        # Read the rest of the file
        all_data = np.fromfile(f, dtype=np.float64)
        
        frame_size = rows * cols
        if all_data.size % frame_size != 0:
            print(f"Error: Total data size ({all_data.size}) is not a multiple of frame size ({frame_size})")
            return None, None, None
            
        num_frames = all_data.size // frame_size
        
        # Reshape into (num_frames, rows, cols)
        data_stack = all_data.reshape((num_frames, rows, cols))
        return rows, cols, data_stack

def create_frame_image(frame_data, frame_number, total_frames, vmin=None, vmax=None):
    """
    Creates a single frame image using Matplotlib.
    Returns a numpy array (image) in BGR format for OpenCV.
    """
    
    # --- Start: Default theme with 'coolwarm' cmap ---
    
    # Create figure with default white background
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display the grid, changing colormap to 'coolwarm'
    # use provided vmin/vmax (defaults to data range if None)
    im = ax.imshow(frame_data, cmap='coolwarm', vmin=vmin, vmax=vmax)
    
    # Add color bar (default styling)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Value', fontsize=14)
    
    # Add title and frame number (default black text)
    ax.set_title(f"Stencil Heat Diffusion", fontsize=16)
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    
    # Add frame number text (default styling)
    ax.text(0.02, 0.95, f'Frame: {frame_number} / {total_frames - 1}',
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # --- End: Default theme ---

    # Convert matplotlib canvas to a numpy array
    fig.canvas.draw()
    
    # 1. Get the ARGB buffer (4 channels)
    img_data_argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    
    # 2. Reshape to (height, width, 4)
    img_data_argb = img_data_argb.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    
    # 3. Convert from ARGB to BGR using NumPy slicing
    #    Matplotlib's buffer is [A, R, G, B] (indices 0, 1, 2, 3)
    #    OpenCV needs [B, G, R] (indices 3, 2, 1)
    img_data_bgr = img_data_argb[:, :, [3, 2, 1]]
    
    # Close the figure to free memory
    plt.close(fig)
    
    # Return the BGR image
    return img_data_bgr

def main():
    parser = argparse.ArgumentParser(description="Create a movie from a stencil data stack.")
    parser.add_argument('--in', dest='input_file', required=True,
                        help="Input stack file (e.g., all.100x100x500.dat)")
    parser.add_argument('--out', dest='output_file', required=True,
                        help="Output movie file (e.g., movie.mp4)")
    parser.add_argument('--fps', type=int, default=30,
                        help="Frames per second for the output movie (default: 30)")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    # 1. Read the data
    print(f"Reading data from {args.input_file}...")
    rows, cols, stack = read_stack_file(args.input_file)
    if stack is None:
        sys.exit(1)
        
    num_frames = stack.shape[0]
    print(f"Read {num_frames} frames of {rows}x{cols} data.")

    # 2. Prepare the video writer
    # Get frame dimensions by creating one frame
    # compute global min/max so colormap uses full dynamic range across all frames
    global_vmin = float(stack.min())
    global_vmax = float(stack.max())
    
    test_frame = create_frame_image(stack[0], 0, num_frames, vmin=global_vmin, vmax=global_vmax)
    height, width, layers = test_frame.shape
    
    # Use 'mp4v' codec for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(args.output_file, fourcc, args.fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for {args.output_file}")
        sys.exit(1)

    print(f"Generating {num_frames} frames for {args.output_file}...")
    
    # 3. Create and write each frame
    # Use tqdm for a progress bar
    for i in tqdm(range(num_frames), desc="Rendering movie"):
        frame_img = create_frame_image(stack[i], i, num_frames, vmin=global_vmin, vmax=global_vmax)
        
        # The frame_img is already in BGR format, so we can write it directly
        video_writer.write(frame_img)

    # 4. Clean up
    video_writer.release()
    print(f"\nMovie successfully saved to {args.output_file}")

if __name__ == "__main__":
    main()

