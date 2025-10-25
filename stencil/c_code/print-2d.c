/*
 * print-2d.c
 *
 * Reads a 2D data file (in binary format) and prints its
 * contents to the screen in a human-readable format.
 *
 * Usage:
 * ./print-2d <input data file>
 *
 * Example:
 * ./print-2d initial_grid.dat
 */

#include <stdio.h>
#include <stdlib.h>
#include "utilities.h"

int main(int argc, char* argv[]) {
    // --- 1. Print usage and parse arguments ---
    if (argc != 2) {
        fprintf(stderr, "usage: ./print-2d <input data file>\n");
        return 1;
    }
    
    const char* filename = argv[1];

    // --- 2. Read data from file ---
    int rows, cols;
    double** data = read_data_from_file(filename, &rows, &cols);

    if (data == NULL) {
        fprintf(stderr, "Error: Failed to read data from %s\n", filename);
        return 1;
    }

    // --- 3. Print data to screen ---
    printf("reading in file: %s (%d x %d)\n", filename, rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Print with 2 decimal places, followed by a space
            // Add an extra space for padding
            printf("%.2f  ", data[i][j]); 
        }
        printf("\n"); // Newline at the end of each row
    }

    // --- 4. Clean up ---
    free_2d_array(data);

    return 0;
}
