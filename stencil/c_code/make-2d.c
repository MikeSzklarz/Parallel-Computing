/*
 * make-2d.c
 *
 * Creates a 2D data file with specified dimensions and boundary conditions.
 *
 * Boundary Conditions:
 * - Left wall (col 0): 1.0
 * - Right wall (col N-1): 1.0
 * - Top wall (row 0, interior): 0.0
 * - Bottom wall (row M-1, interior): 0.0
 * - All other interior points: 0.0
 *
 * Corners will be 1.0 as the left/right wall conditions take precedence.
 *
 * Usage:
 * ./make-2d <rows> <cols> <output_file>
 *
 * Example:
 * ./make-2d 100 100 initial_grid.dat
 */

#include <stdio.h>
#include <stdlib.h>
#include "utilities.h"

int main(int argc, char* argv[]) {
    // --- 1. Print usage and parse arguments ---
    if (argc != 4) {
        fprintf(stderr, "usage: ./make-2d <rows> <cols> <output_file>\n");
        return 1;
    }

    int rows = atoi(argv[1]);
    int cols = atoi(argv[2]);
    const char* filename = argv[3];

    if (rows <= 2 || cols <= 2) {
        fprintf(stderr, "Error: rows and cols must be greater than 2.\n");
        return 1;
    }

    // --- 2. Allocate memory ---
    double** data = allocate_2d_array(rows, cols);
    if (data == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for 2D array.\n");
        return 1;
    }

    // --- 3. Initialize data (all to 0.0) ---
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = 0.0;
        }
    }

    // --- 4. Set boundary conditions ---
    
    // Left and Right walls (1.0)
    // This will set the corners to 1.0
    for (int i = 0; i < rows; i++) {
        data[i][0] = 1.0;
        data[i][cols - 1] = 1.0;
    }

    // Top and Bottom walls (0.0) - INTERIOR ONLY
    // We only need to set 0.0 for j from 1 to cols-2
    // But since we initialized everything to 0.0, this step is
    // technically already done. We include it for clarity.
    for (int j = 1; j < cols - 1; j++) {
        data[0][j] = 0.0;
        data[rows - 1][j] = 0.0;
    }

    // --- 5. Write data to file ---
    write_data_to_file(filename, data, rows, cols);

    // --- 6. Clean up ---
    free_2d_array(data);

    printf("Successfully created initial data file: %s (%d x %d)\n", filename, rows, cols);

    return 0;
}
