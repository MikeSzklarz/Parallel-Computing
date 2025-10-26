/*
 * stencil-2d.c
 *
 * Performs a 9-point stencil operation on a 2D data file over
 * multiple iterations.
 *
 * This version uses a double-buffer approach with pointer swapping
 * for high performance, eliminating array copies.
 *
 * Usage:
 * ./stencil-2d <num iterations> <input_file> <output_file> [all_iterations_file]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> // For memset
#include "utilities.h"
#include "timer.h" // For timing

int main(int argc, char* argv[]) {
    // --- 1. Print usage and parse arguments ---
    if (argc < 4 || argc > 5) {
        fprintf(stderr, "usage: ./stencil-2d <num iterations> <input_file> <output_file> [all_iterations_file]\n");
        return 1;
    }

    int k_iters = atoi(argv[1]);
    const char* input_filename = argv[2];
    const char* output_filename = argv[3];
    const char* all_iter_filename = (argc == 5) ? argv[4] : NULL;

    if (k_iters < 0) {
        fprintf(stderr, "Error: iterations must be a non-negative integer.\n");
        return 1;
    }

    // --- 2. Read initial data and allocate grids ---
    int rows, cols;
    // 'read_grid' will be our persistent "current" state
    double** read_grid = read_data_from_file(input_filename, &rows, &cols);
    if (read_grid == NULL) {
        fprintf(stderr, "Error reading from %s\n", input_filename);
        return 1;
    }

    // 'write_grid' will be our "next" state
    double** write_grid = allocate_2d_array(rows, cols);
    if (write_grid == NULL) {
        fprintf(stderr, "Error allocating write_grid\n");
        free_2d_array(read_grid);
        return 1;
    }

    // Copy initial state to write_grid (to preserve boundaries)
    // We only need to do this ONCE.
    copy_2d_array(read_grid, write_grid, rows, cols);


    // --- 3. Open movie file stream if provided ---
    FILE* all_iter_stream = NULL;
    if (all_iter_filename != NULL) {
        all_iter_stream = fopen(all_iter_filename, "wb");
        if (all_iter_stream == NULL) {
            perror("Error opening all_iterations_file for writing");
            free_2d_array(read_grid);
            free_2d_array(write_grid);
            return 1;
        }

        // Write the metadata (rows, cols) ONCE at the beginning
        if (fwrite(&rows, sizeof(int), 1, all_iter_stream) != 1 ||
            fwrite(&cols, sizeof(int), 1, all_iter_stream) != 1) {
            fprintf(stderr, "Error writing metadata to %s\n", all_iter_filename);
            fclose(all_iter_stream);
            free_2d_array(read_grid);
            free_2d_array(write_grid);
            return 1;
        }
    }
    
    // --- 4. Write initial state (Iteration 0) to movie file ---
    if (all_iter_stream != NULL) {
        append_data_to_stream(all_iter_stream, read_grid, rows, cols);
    }

    // --- 5. Define stencil weights ---
    const double w_total = 9.0;
    double** temp_ptr = NULL; // For swapping

    // --- 6. Perform stencil iterations ---
    printf("Starting %d stencil iterations on %dx%d grid...\n", k_iters, rows, cols);

    // --- Add Timing ---
    double start, finish, elapsed;
    GET_TIME(start);

    for (int k = 0; k < k_iters; k++) {
        // Compute new values for all INTERIOR points
        // Read from 'read_grid', write to 'write_grid'
        for (int i = 1; i < rows - 1; i++) {
            for (int j = 1; j < cols - 1; j++) {
                
                // Get values from the 'read_grid',
                // following the exact order from your instructions.
                double sum = 0.0;
                
                sum += read_grid[i - 1][j - 1]; // NW
                sum += read_grid[i - 1][j];     // N
                sum += read_grid[i - 1][j + 1]; // NE
                sum += read_grid[i][j + 1];     // E
                sum += read_grid[i + 1][j + 1]; // SE
                sum += read_grid[i + 1][j];     // S
                sum += read_grid[i + 1][j - 1]; // SW
                sum += read_grid[i][j - 1];     // W
                sum += read_grid[i][j];         // C (Center)

                // Apply the simple average
                write_grid[i][j] = sum / w_total;
            }
        }
        
        // Write this iteration's *result* to the movie file
        if (all_iter_stream != NULL) {
            // write_grid now holds the new state
            append_data_to_stream(all_iter_stream, write_grid, rows, cols);
        }

        // --- POINTER SWAP ---
        // The 'write_grid' is now the "current" state for the next iteration
        // and 'read_grid' is the "next" (empty) buffer.
        temp_ptr = read_grid;
        read_grid = write_grid;
        write_grid = temp_ptr;
    }
    
    // --- Stop Timing ---
    GET_TIME(finish);
    elapsed = finish - start;

    printf("Iterations complete.\n");
    printf("Computation Time: %f seconds\n", elapsed); // Human-readable
    printf("COMP_TIME: %f\n", elapsed); // Machine-readable for parser

    // --- 7. Write final state to output file ---
    // 'read_grid' holds the pointer to the *last* computed state
    // (due to the final swap).
    write_data_to_file(output_filename, read_grid, rows, cols);
    printf("Final state written to %s\n", output_filename);

    // --- 8. Clean up ---
    if (all_iter_stream != NULL) {
        fclose(all_iter_stream);
        printf("All %d iterations (N+1 states) written to %s\n", k_iters, all_iter_filename);
    }
    
    free_2d_array(read_grid);
    free_2d_array(write_grid);

    return 0;
}

