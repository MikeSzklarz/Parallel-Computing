/*
 * utilities.h
 *
 * Header file for shared utility functions in the 2D stencil project.
 * Declares functions for 2D array memory management and file I/O.
 */

#ifndef UTILITIES_H
#define UTILITIES_H

#include <stdio.h> // For FILE*

/**
 * @brief Allocates a contiguous 2D array (matrix) of doubles.
 *
 * Memory is allocated as one contiguous block for the data, and an
 * array of pointers to the start of each row.
 *
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 * @return A pointer to an array of pointers (double**), representing the 2D array.
 * Returns NULL on allocation failure.
 */
double** allocate_2d_array(int rows, int cols);

/**
 * @brief Frees a contiguous 2D array allocated by allocate_2d_array.
 *
 * @param data The 2D array to free.
 * @param rows The number of rows (needed if you want to be extra safe,
 * but with our allocation scheme, only data and data[0] are needed).
 */
void free_2d_array(double** data);

/**
 * @brief Copies the contents of one 2D array to another.
 *
 * Assumes both src and dest are of the same dimensions (rows, cols)
 * and have been allocated.
 *
 * @param src The source array.
 * @param dest The destination array.
 * @param rows The number of rows.
 * @param cols The number of columns.
 */
void copy_2d_array(double** src, double** dest, int rows, int cols);

/**
 * @brief Reads matrix dimensions and data from a binary file.
 *
 * The file is expected to store two ints (rows, cols) followed by
 * (rows * cols) doubles.
 * This function allocates memory for the matrix.
 *
 * @param filename The name of the input file.
 * @param rows Pointer to an int to store the number of rows.
 * @param cols Pointer to an int to store the number of columns.
 * @return A new 2D array (double**) containing the data from the file.
 * Returns NULL on file read or allocation failure.
 */
double** read_data_from_file(const char* filename, int* rows, int* cols);

/**
 * @brief Writes matrix dimensions and data to a binary file.
 *
 * The file will store two ints (rows, cols) followed by
 * (rows * cols) doubles. This function overwrites any existing file.
 *
 * @param filename The name of the output file.
 * @param data The 2D array to write.
 * @param rows The number of rows.
 * @param cols The number of columns.
 */
void write_data_to_file(const char* filename, double** data, int rows, int cols);

/**
 * @brief Appends matrix dimensions and data to an open binary file stream.
 *
 * Used for writing multiple iterations to a single "movie" file.
 * The stream is NOT closed by this function.
 *
 * @param stream An open FILE* stream (e.g., from fopen(..., "wb")).
 * @param data The 2D array to write.
 * @param rows The number of rows.
 * @param cols The number of columns.
 */
void append_data_to_stream(FILE* stream, double** data, int rows, int cols);

#endif // UTILITIES_H
