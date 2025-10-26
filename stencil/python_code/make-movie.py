#!/usr/bin/env python3

"""
make-movie.py

Reads a raw "all iterations" binary stack file and converts it to an
MP4 movie using OpenCV for high performance.

The binary file is expected to have the format:
- int: rows
- int: cols
- double[rows*cols]: frame 0 data
- double[rows*cols]: frame 1 data
- ...
- double[rows*cols]: frame N data

Usage:
python ./make-movie.py --in <input_stack_file> --out <output_movie_file>

Example:
python ./make-movie.py --in ../data/all.100x100x500.raw --out ../data/stencil.mp4
"""

import numpy as np
import cv2  # Requires opencv-python
import os
import sys
import argparse

def create_movie(input_file, output_file, fps=30):
    """
    Reads the raw stack file and writes an MP4 movie.
    """
    try:
        # Open the file and read metadata
        with open(input_file, 'rb') as f:
            # Read metadata (2 ints)
            metadata = np.fromfile(f, dtype=np.int32, count=2)
            rows, cols = metadata[0], metadata[1]
            
            print(f"Reading {rows}x{cols} frames from {input_file}...")
            
            # Get file size to calculate number of frames
            f.seek(0, os.SEEK_END)
            total_bytes = f.tell()
            metadata_bytes = 2 * np.dtype(np.int32).itemsize
            data_bytes = total_bytes - metadata_bytes
            
            frame_size_bytes = rows * cols * np.dtype(np.float64).itemsize
            
            if data_bytes % frame_size_bytes != 0:
                print("Error: File data size is not an even multiple of frame size.")
                print(f"Data bytes: {data_bytes}, Frame bytes: {frame_size_bytes}")
                return

            num_frames = data_bytes // frame_size_bytes
            print(f"Found {num_frames} frames.")

            if num_frames == 0:
                print("Error: No frames found in file.")
                return

            # --- Setup Video Writer (OpenCV) ---
            # Define a fixed output size for the video
            output_dim = (800, 800) # (width, height)
            
            # Define the codec and create VideoWriter object
            # 'mp4v' is a good codec for .mp4 files.
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Note: OpenCV dimensions are (width, height) -> (cols, rows)
            # We now use our fixed output_dim
            video_writer = cv2.VideoWriter(output_file, fourcc, fps, output_dim)

            if not video_writer.isOpened():
                print(f"Error: Could not open video writer for {output_file}")
                return

            # Rewind to the start of the data
            f.seek(metadata_bytes)
            
            # --- Process frames one by one ---
            for i in range(num_frames):
                # Read one frame's worth of data
                frame_data = np.fromfile(f, dtype=np.float64, count=rows * cols)
                
                if frame_data.size != rows * cols:
                    print(f"Warning: Truncated file? Stopped at frame {i}")
                    break
                    
                frame = frame_data.reshape((rows, cols))
                
                # --- Convert data to 8-bit image ---
                # 1. Normalize from [0.0, 1.0] to [0, 255]
                # We clip to ensure values are in range (just in case)
                frame_normalized = np.clip(frame, 0.0, 1.0)
                frame_uint8 = (frame_normalized * 255).astype(np.uint8)
                
                # 2. Apply a colormap
                #    Changed to COLORMAP_JET as COOLWARM is not in all cv2 versions
                frame_color = cv2.applyColorMap(frame_uint8, cv2.COLORMAP_JET)
                
                # 3. --- NEW: Resize the frame ---
                # Scale from (rows, cols) up to our (output_dim)
                # We use INTER_NEAREST to keep the sharp "pixel" look
                frame_resized = cv2.resize(frame_color, output_dim, interpolation=cv2.INTER_NEAREST)
                
                # 4. Add frame number
                # --- Text parameters adjusted for larger 800x800 frame ---
                font_scale = 1.0
                thickness = 2
                position = (10, 30) # 10px from left, 30px from top
                
                cv2.putText(
                    frame_resized, # Draw text on the resized frame
                    f"frame {i}",
                    position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),  # Color (white)
                    thickness
                )
                
                # 5. Write the resized frame
                video_writer.write(frame_resized)
                
                if (i + 1) % fps == 0:
                    print(f"Processed frame {i + 1}/{num_frames}")

            # Release the video writer
            video_writer.release()
            print(f"\nSuccessfully created movie: {output_file}")

    except FileNotFoundError:
        print(f"Error: File not found at {input_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert a 2D stencil raw stack file to an MP4 movie.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--in",
        dest="input_file",
        required=True,
        help="Path to the input raw stack file."
    )
    parser.add_argument(
        "--out",
        dest="output_file",
        required=True,
        help="Path for the output .mp4 movie file."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the output movie."
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
        
    create_movie(args.input_file, args.output_file, args.fps)

if __name__ == "__main__":
    main()