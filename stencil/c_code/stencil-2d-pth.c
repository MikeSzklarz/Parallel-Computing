// stencil-2d-pth.c
// Parallel 9-point 2D stencil with Pthreads (row decomposition + double buffering)
//
// Usage:
//   ./stencil-2d-pth <iters> <in.raw> <out.raw> [-s <stack.raw>] -t <threads>
//
// Minimal fixes implemented:
//  - Start timing AFTER threads are created (one-time launch barrier).
//  - Cap active threads to min(requested, rows-2) to avoid barrier-only threads.
//  - Keep I/O out of timed loop (no per-iteration fflush; fclose flushes once).
//  - Match utilities.h API: read_data_from_file, write_data_to_file (void),
//    allocate_2d_array, free_2d_array, append_data_to_stream.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>

#include "utilities.h"  // read_data_from_file, write_data_to_file, allocate_2d_array, free_2d_array, append_data_to_stream
#include "timer.h"      // GET_TIME(t) -> monotonic-ish timing

typedef struct {
    int rows, cols, iters;
    int nthreads;                 // active worker count

    // Per-iteration barrier (workers only) and a one-time launch barrier (workers + main)
    pthread_barrier_t *bar;       // count = nthreads
    pthread_barrier_t *launch;    // count = nthreads + 1 (includes main)

    // Double buffers; thread 0 swaps these pointers after each iteration
    double ***pread;              // &A
    double ***pwrite;             // &B

    // Optional stack output
    FILE *stack_fp;
    int   write_stack;
} Shared;

typedef struct {
    Shared *S;
    int tid;                      // 0..nthreads-1
    int start_row;                // inclusive (interior index)
    int end_row;                  // exclusive
} Worker;

// Match serial arithmetic and order exactly to ensure byte-for-byte match.
static inline double stencil9(double **restrict A, int i, int j)
{
    double sum = 0.0;
    sum += A[i-1][j-1]; // NW
    sum += A[i-1][j];   // N
    sum += A[i-1][j+1]; // NE
    sum += A[i][j+1];   // E
    sum += A[i+1][j+1]; // SE
    sum += A[i+1][j];   // S
    sum += A[i+1][j-1]; // SW
    sum += A[i][j-1];   // W
    sum += A[i][j];     // C
    return sum / 9.0;
}

// Partition interior rows [1 .. rows-2] across P threads as evenly as possible.
static void compute_band(int rows, int P, int tid, int *start_row, int *end_row)
{
    const int interior = (rows >= 2) ? (rows - 2) : 0;
    if (interior <= 0) { *start_row = 1; *end_row = 1; return; }

    const int base = interior / P;
    const int rem  = interior % P;

    const int my_count = base + (tid < rem ? 1 : 0);
    const int prior    = tid * base + (tid < rem ? tid : rem);

    *start_row = 1 + prior;          // first interior row is index 1
    *end_row   = *start_row + my_count;
}

static void* worker_main(void *arg)
{
    Worker *w = (Worker*)arg;
    Shared *S = w->S;

    const int rows  = S->rows;
    const int cols  = S->cols;
    const int iters = S->iters;

    const int r0 = w->start_row;
    const int r1 = w->end_row;   // exclusive

    // Wait until main has created all threads and started the timer.
    pthread_barrier_wait(S->launch);

    for (int t = 0; t < iters; ++t) {
        double **read_grid  = *S->pread;
        double **write_grid = *S->pwrite;

        // Compute my band into write_grid (boundaries not updated)
        for (int i = r0; i < r1; ++i) {
            for (int j = 1; j < cols - 1; ++j) {
                write_grid[i][j] = stencil9(read_grid, i, j);
            }
        }

        // Barrier 1: all threads finished computing this iteration
        pthread_barrier_wait(S->bar);

        // Thread 0 swaps A/B and (optionally) appends the new current frame to stack
        if (w->tid == 0) {
            double **tmp = *S->pread;
            *S->pread  = *S->pwrite;
            *S->pwrite = tmp;

            if (S->write_stack && S->stack_fp) {
                append_data_to_stream(S->stack_fp, *S->pread, rows, cols);
                // no fflush here; fclose at the end will flush once
            }
        }

        // Barrier 2: ensure everyone sees new A/B before next iteration
        pthread_barrier_wait(S->bar);
    }

    return NULL;
}

static void usage(const char *prog)
{
    fprintf(stderr, "Usage: %s <iters> <in.raw> <out.raw> [-s <stack.raw>] -t <threads>\n", prog);
}

int main(int argc, char **argv)
{
    if (argc < 4) { usage(argv[0]); return 1; }

    int iters = atoi(argv[1]);
    const char *in_path  = argv[2];
    const char *out_path = argv[3];

    const char *stack_path = NULL;
    int threads_req = 1;

    for (int i = 4; i < argc; ++i) {
        if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            stack_path = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threads_req = atoi(argv[++i]);
        }
    }
    if (iters < 0 || threads_req < 1) { usage(argv[0]); return 1; }

    // Read input grid
    int rows = 0, cols = 0;
    double **A = read_data_from_file(in_path, &rows, &cols);
    if (A == NULL) {
        fprintf(stderr, "Failed to read input '%s'\n", in_path);
        return 1;
    }

    // Allocate second buffer and copy boundaries once to keep them identical to A
    double **B = allocate_2d_array(rows, cols);
    if (!B) {
        fprintf(stderr, "allocate_2d_array failed for %dx%d\n", rows, cols);
        free_2d_array(A);
        return 1;
    }
    // Copy whole grid once to seed B (ensures borders correct even before any writes)
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            B[i][j] = A[i][j];

    // Optional stack: write initial frame (t=0) and remember to append after each swap
    FILE *sf = NULL;
    if (stack_path && stack_path[0] != '\0') {
        sf = fopen(stack_path, "wb");
        if (!sf) {
            fprintf(stderr, "Failed to open stack file '%s': %s\n", stack_path, strerror(errno));
            free_2d_array(B); free_2d_array(A);
            return 1;
        }
        // If your serial writer includes metadata, emit it here in the same format.
        // Many of your utilities expect just frames; if metadata is needed, add it consistently.
        append_data_to_stream(sf, A, rows, cols); // t=0 frame
    }

    // Determine active worker count (avoid barrier-only threads)
    const int interior_rows = (rows >= 2) ? (rows - 2) : 0;
    int active = threads_req;
    if (interior_rows <= 0)      active = 1;
    else if (active > interior_rows) active = interior_rows;

    // Barriers: per-iter among workers; one-time launch among workers+main
    pthread_barrier_t bar, launch;
    pthread_barrier_init(&bar,    NULL, active);
    pthread_barrier_init(&launch, NULL, active + 1);

    // Shared state
    Shared S;
    S.rows = rows; S.cols = cols; S.iters = iters; S.nthreads = active;
    S.bar = &bar; S.launch = &launch;
    S.stack_fp = sf; S.write_stack = (sf != NULL);

    double **read_grid  = A;
    double **write_grid = B;
    S.pread  = &read_grid;
    S.pwrite = &write_grid;

    // Threads + partition
    pthread_t *tids = (pthread_t*)malloc(sizeof(pthread_t) * active);
    Worker    *ws   = (Worker*)   malloc(sizeof(Worker)    * active);
    if (!tids || !ws) {
        fprintf(stderr, "malloc failed for thread structs\n");
        if (sf) fclose(sf);
        free(tids); free(ws);
        free_2d_array(B); free_2d_array(A);
        return 1;
    }
    for (int t = 0; t < active; ++t) {
        ws[t].S = &S; ws[t].tid = t;
        compute_band(rows, active, t, &ws[t].start_row, &ws[t].end_row);
    }

    // Create workers (they block on launch barrier)
    for (int t = 0; t < active; ++t) {
        if (pthread_create(&tids[t], NULL, worker_main, &ws[t]) != 0) {
            fprintf(stderr, "pthread_create failed: %s\n", strerror(errno));
            // join any started threads
            for (int j = 0; j < t; ++j) pthread_join(tids[j], NULL);
            if (sf) fclose(sf);
            pthread_barrier_destroy(&bar);
            pthread_barrier_destroy(&launch);
            free(ws); free(tids);
            free_2d_array(B); free_2d_array(A);
            return 1;
        }
    }

    // Timed region begins AFTER creation; release workers together.
    double t0, t1; GET_TIME(t0);
    pthread_barrier_wait(&launch);

    for (int t = 0; t < active; ++t) pthread_join(tids[t], NULL);
    GET_TIME(t1);
    printf("COMP_TIME: %.6f\n", (t1 - t0));

    // Cleanup threading primitives and stack
    pthread_barrier_destroy(&bar);
    pthread_barrier_destroy(&launch);
    if (sf) fclose(sf);

    // Final grid is in *S.pread (read_grid)
    write_data_to_file(out_path, *S.pread, rows, cols);

    // Free resources
    free(ws); free(tids);
    free_2d_array(B);
    free_2d_array(A);
    return 0;
}
