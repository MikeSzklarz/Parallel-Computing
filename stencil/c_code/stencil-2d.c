/*
 * stencil-2d.c
 *
 * Performs a 9-point stencil operation on a 2D data file over
 * multiple iterations.
 *
 * Stencil Weights (Simple Average, as per user's instructions):
 * [1  1  1]
 * [1  1  1] * (1/9)
 * [1  1  1]
 *
 * Only interior points (not on the boundary) are updated.
 *
 * Usage:
 * ./stencil-2d <num iterations> <input_file> <output_file> [all_iterations_file]
 */

#include <stdio.h>
#include <stdlib.h>
#include "utilities.h"
#include "timer.h" // <-- ADDED: Include the timer header

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
    double** current_grid = read_data_from_file(input_filename, &rows, &cols);
    if (current_grid == NULL) {
        fprintf(stderr, "Error reading from %s\n", input_filename);
        return 1;
    }

    double** next_grid = allocate_2d_array(rows, cols);
    if (next_grid == NULL) {
        fprintf(stderr, "Error allocating next_grid\n");
        free_2d_array(current_grid);
        return 1;
    }

    // Copy initial state to next_grid (to preserve boundaries)
    copy_2d_array(current_grid, next_grid, rows, cols);


    // --- 3. Open movie file stream if provided ---
    FILE* all_iter_stream = NULL;
    if (all_iter_filename != NULL) {
        all_iter_stream = fopen(all_iter_filename, "wb");
        if (all_iter_stream == NULL) {
            perror("Error opening all_iterations_file for writing");
            free_2d_array(current_grid);
            free_2d_array(next_grid);
            return 1;
        }

        // Write the metadata (rows, cols) ONCE at the beginning
        // of the "all iterations" file.
        if (fwrite(&rows, sizeof(int), 1, all_iter_stream) != 1 ||
            fwrite(&cols, sizeof(int), 1, all_iter_stream) != 1) {
            fprintf(stderr, "Error writing metadata to %s\n", all_iter_filename);
            fclose(all_iter_stream);
            free_2d_array(current_grid);
            free_2d_array(next_grid);
            return 1;
        }
    }
    
    // --- 4. Write initial state (Iteration 0) to movie file ---
    if (all_iter_stream != NULL) {
        // This function *only* writes the raw data, not metadata
        append_data_to_stream(all_iter_stream, current_grid, rows, cols);
    }

    // --- 5. Define stencil weights ---
    // Simple 9-point average (sum of all 9 cells / 9.0)
    const double w_total = 9.0;
    
    // --- ADDED: Declare timer variables ---
    double start, finish, elapsed;

    // --- 6. Perform stencil iterations ---
    printf("Starting %d stencil iterations on %dx%d grid...\n", k_iters, rows, cols);

    GET_TIME(start); // <-- ADDED: Start timer

    for (int k = 0; k < k_iters; k++) {
        // Compute new values for all INTERIOR points
        for (int i = 1; i < rows - 1; i++) {
            for (int j = 1; j < cols - 1; j++) {
                
                // Get values from the 'current' grid,
                // following the exact order from your instructions.
                double sum = 0.0;
                
                sum += current_grid[i - 1][j - 1]; // NW
                sum += current_grid[i - 1][j];     // N
                sum += current_grid[i - 1][j + 1]; // NE
                sum += current_grid[i][j + 1];     // E
                sum += current_grid[i + 1][j + 1]; // SE
                sum += current_grid[i + 1][j];     // S
                sum += current_grid[i + 1][j - 1]; // SW
                sum += current_grid[i][j - 1];     // W
                sum += current_grid[i][j];         // C (Center)

                // Apply the simple average
                next_grid[i][j] = sum / w_total;
            }
        }

        // Copy 'next' grid back to 'current' grid for the next iteration
        copy_2d_array(next_grid, current_grid, rows, cols);
        
        // Write this iteration's *result* to the movie file
        if (all_iter_stream != NULL) {
            append_data_to_stream(all_iter_stream, current_grid, rows, cols);
        }
    }

    GET_TIME(finish); // <-- ADDED: Stop timer
    elapsed = finish - start; // <-- ADDED: Calculate elapsed time

    printf("Iterations complete.\n");
    printf("Computation Time: %f seconds\n", elapsed); // <-- Human-readable
    printf("COMP_TIME: %f\n", elapsed); // <-- Machine-readable for parser

    // --- 7. Write final state to output file ---
    // This function (write_data_to_file) is unchanged
    // and correctly writes (metadata + 1 frame)
    write_data_to_file(output_filename, current_grid, rows, cols);
    printf("Final state written to %s\n", output_filename);

    // --- 8. Clean up ---
    if (all_iter_stream != NULL) {
        fclose(all_iter_stream);
        printf("All %d iterations (N+1 states) written to %s\n", k_iters, all_iter_filename);
    }
    
    free_2d_array(current_grid);
    free_2d_array(next_grid);

    return 0;
}


