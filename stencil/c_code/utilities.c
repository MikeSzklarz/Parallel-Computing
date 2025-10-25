/*
 * utilities.c
 *
 * Source file for utility functions shared across the stencil project.
 * - 2D array allocation/deallocation
 * - File I/O
 *
 * File Format:
 * - A file consists of two 'int' values (rows, cols) followed
 * by (rows * cols) 'double' values in row-major order.
 *
 * "All Iterations" File Format:
 * - This file consists of two 'int' values (rows, cols) followed
 * by (N+1) * (rows * cols) 'double' values.
 * - The metadata is written *only once* at the beginning.
 * - The `append_data_to_stream` function is used to write *only*
 * the raw data for each frame.
 */

#include "utilities.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * Allocates a 2D array of doubles.
 * Uses a single contiguous block of memory for cache efficiency
 * and easy row-major file I/O.
 */
double** allocate_2d_array(int rows, int cols) {
    if (rows <= 0 || cols <= 0) return NULL;

    // Allocate memory for the row pointers
    double** arr = (double**)malloc(rows * sizeof(double*));
    if (arr == NULL) {
        perror("Error: malloc failed for row pointers");
        return NULL;
    }

    // Allocate a single contiguous block for all the data
    double* data_block = (double*)malloc(rows * cols * sizeof(double));
    if (data_block == NULL) {
        perror("Error: malloc failed for data block");
        free(arr);
        return NULL;
    }

    // Set each row pointer to the correct position in the data block
    for (int i = 0; i < rows; i++) {
        arr[i] = data_block + (i * cols);
    }

    return arr;
}

/**
 * Frees a 2D array allocated by allocate_2d_array.
 */
void free_2d_array(double** arr) {
    if (arr != NULL) {
        // Free the contiguous data block (which is arr[0])
        if (arr[0] != NULL) {
            free(arr[0]);
        }
        // Free the row pointers
        free(arr);
    }
}

/**
 * Copies the contents of one 2D array to another.
 * Assumes both arrays are the same size.
 */
void copy_2d_array(double** src, double** dest, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dest[i][j] = src[i][j];
        }
    }
}

/**
 * Writes a 2D grid to a binary file.
 * Format: int rows, int cols, double[rows*cols] data
 */
void write_data_to_file(const char* filename, double** grid, int rows, int cols) {
    FILE* f = fopen(filename, "wb");
    if (f == NULL) {
        perror("Error opening file for writing");
        return;
    }

    // 1. Write metadata (rows, cols)
    if (fwrite(&rows, sizeof(int), 1, f) != 1 ||
        fwrite(&cols, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Error writing metadata to %s\n", filename);
        fclose(f);
        return;
    }

    // 2. Write data block (grid[0] is the start of the contiguous block)
    if (fwrite(grid[0], sizeof(double), rows * cols, f) != (size_t)(rows * cols)) {
        fprintf(stderr, "Error writing data to %s\n", filename);
    }

    fclose(f);
}

/**
 * Appends *only* the raw 2D grid data to an already open file stream.
 * This function does NOT write metadata.
 * It is used for writing frames to the 'all_iterations' file.
 */
void append_data_to_stream(FILE* stream, double** grid, int rows, int cols) {
    if (stream == NULL) return;

    // Write data block (grid[0] is the start of the contiguous block)
    if (fwrite(grid[0], sizeof(double), rows * cols, stream) != (size_t)(rows * cols)) {
        fprintf(stderr, "Error appending data to stream\n");
    }
}


/**
 * Reads a 2D grid from a binary file.
 * Allocates memory for the grid. The caller is responsible for freeing it.
 * Format: int rows, int cols, double[rows*cols] data
 */
double** read_data_from_file(const char* filename, int* out_rows, int* out_cols) {
    FILE* f = fopen(filename, "rb");
    if (f == NULL) {
        perror("Error opening file for reading");
        return NULL;
    }

    // 1. Read metadata
    int rows, cols;
    if (fread(&rows, sizeof(int), 1, f) != 1 ||
        fread(&cols, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Error reading metadata from %s\n", filename);
        fclose(f);
        return NULL;
    }

    // 2. Allocate grid
    double** grid = allocate_2d_array(rows, cols);
    if (grid == NULL) {
        fprintf(stderr, "Failed to allocate grid for reading file\n");
        fclose(f);
        return NULL;
    }

    // 3. Read data block
    if (fread(grid[0], sizeof(double), rows * cols, f) != (size_t)(rows * cols)) {
        fprintf(stderr, "Error reading data from %s (file might be truncated)\n", filename);
        free_2d_array(grid);
        fclose(f);
        return NULL;
    }

    fclose(f);
    *out_rows = rows;
    *out_cols = cols;
    return grid;
}

/**
 * Prints the grid to the console in a formatted way.
 */
void print_grid(double** grid, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f  ", grid[i][j]);
        }
        printf("\n");
    }
}

