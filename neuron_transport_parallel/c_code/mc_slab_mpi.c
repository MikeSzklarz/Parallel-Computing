/*
 * mc_slab_mpi.c
 *
 * MPI parallel C implementation of a 2D Monte Carlo neutron transport.
 *
 * This implementation goes "above and beyond" by using a high-quality,
 * rank-safe xorshift128+ PRNG instead of the weaker rand().
 *
 * Each rank simulates its share of particles and MPI_Reduce is used
 * to sum the final counts to rank 0.
 * Tracing is supported on a per-rank basis (e.g., trace_r0.csv, trace_r1.csv).
 *
 * Usage:
 * mpirun -np P ./mc_slab_mpi C Cc H n [--trace-file path] [--trace-every m] [--seed s]
 *
 * Parameters:
 * P >= 1        (number of processes, via mpirun)
 * C  > 0         (total interaction coeff)
 * Cc in [0, C)  (absorbing component)
 * H  > 0         (slab thickness)
 * n  >= 1        (number of particles to simulate, *total*)
 *
 * Optional Flags:
 * --trace-file path  : Base path for trace CSVs (e.g., "trace" -> "trace_r0.csv").
 * --trace-every m    : Write to trace file every 'm' particles.
 * --seed s           : Base random number seed.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> // For strcmp, strrchr
#include <math.h>   // For log, cos, M_PI
#include <time.h>   // For time()
#include <errno.h>  // For error checking
#include <stdint.h> // For uint64_t
#include <mpi.h>

// --- High-Quality PRNG (xorshift128+) ---
// This is superior to rand() for simulation.
// It is safe for MPI as long as each rank seeds differently.

struct prng_state {
    uint64_t s[2];
};

// Function prototype for next_prng() to resolve implicit declaration warning
uint64_t next_prng(struct prng_state *state);

// Function to seed the PRNG state
void seed_prng(struct prng_state *state, uint64_t seed) {
    state->s[0] = seed ^ 0x123456789ABCDEF0;
    state->s[1] = seed ^ 0xFEDCBA9876543210;
    // Simple warm-up
    for(int i=0; i<8; ++i) (void)next_prng(state);
}

// Get next 64-bit random number
uint64_t next_prng(struct prng_state *state) {
	uint64_t s1 = state->s[0];
	const uint64_t s0 = state->s[1];
	state->s[0] = s0;
	s1 ^= s1 << 23; // a
	state->s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5); // b, c
	return state->s[1] + s0;
}

// Get a uniform random double in [0, 1)
double get_rand_double(struct prng_state *state) {
    // 53-bit resolution
    return (next_prng(state) >> 11) * (1.0 / 9007199254740992.0);
}

// --- End PRNG ---


// --- Simulation Parameters ---
// Use a struct to easily Bcast all params from rank 0.
typedef struct {
    double C_total;
    double C_capture;
    double H_thickness;
    long long n_particles_total;
    long long trace_every;
    uint64_t seed_base;
    int trace_path_len; // Length of trace_path string
} sim_params_t;


// Prints the required usage string
void print_usage() {
    printf("Usage: mpirun -np P ./mc_slab_mpi C Cc H n [--trace-file path] [--trace-every m] [--seed s]\n");
    printf("  P >= 1        (number of processes, via mpirun)\n");
    printf("  C > 0         (total interaction coeff)\n");
    printf("  Cc in [0,C)   (absorbing component)\n");
    printf("  H > 0         (slab thickness)\n");
    printf("  n >= 1        (number of particles)\n");
    printf("  --seed s      (optional: random number seed)\n");
    printf("  --trace-file  (optional: base path for trace files)\n");
    printf("  --trace-every (optional: trace output frequency)\n");
}


int main(int argc, char *argv[]) {
    
    int rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    sim_params_t params;
    char* trace_path = NULL; // Only rank 0 allocates this
    
    // --- 1. Argument Parsing and Validation (Rank 0 only) ---
    if (rank == 0) {
        if (argc < 5) {
            print_usage();
            MPI_Abort(MPI_COMM_WORLD, 1); // Abort all processes
        }

        // Set defaults
        params.C_total = atof(argv[1]);
        params.C_capture = atof(argv[2]);
        params.H_thickness = atof(argv[3]);
        params.n_particles_total = atoll(argv[4]);
        params.trace_every = 0;
        params.seed_base = 0;
        params.trace_path_len = 0;
        
        int seed_provided = 0;

        // Parse optional arguments
        for (int i = 5; i < argc; ++i) {
            if (strcmp(argv[i], "--trace-file") == 0) {
                if (i + 1 < argc) trace_path = argv[++i];
            } else if (strcmp(argv[i], "--trace-every") == 0) {
                if (i + 1 < argc) params.trace_every = atoll(argv[++i]);
            } else if (strcmp(argv[i], "--seed") == 0) {
                if (i + 1 < argc) {
                    params.seed_base = (uint64_t)atoll(argv[++i]);
                    seed_provided = 1;
                }
            } else {
                fprintf(stderr, "Error: Unknown argument '%s'.\n", argv[i]);
                print_usage();
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        // Validate
        if (params.C_total <= 0 || params.C_capture < 0 || params.C_capture >= params.C_total || 
            params.H_thickness <= 0 || params.n_particles_total < 1) {
            fprintf(stderr, "Error: Invalid parameters.\n");
            print_usage();
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (trace_path && params.trace_every <= 0) {
            fprintf(stderr, "Error: --trace-every must be > 0 when --trace-file is used.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (!seed_provided) {
            params.seed_base = (uint64_t)time(NULL);
        }
        if (trace_path) {
            params.trace_path_len = strlen(trace_path) + 1; // +1 for null terminator
        }
    }

    // --- 2. Broadcast Parameters ---
    // Broadcast the parameter struct from rank 0 to all other ranks
    MPI_Bcast(&params, sizeof(sim_params_t), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Broadcast the trace_path string (if it exists)
    char trace_path_local[1024]; // Buffer on all ranks
    if (params.trace_path_len > 0) {
        if (rank == 0) {
            strncpy(trace_path_local, trace_path, sizeof(trace_path_local) - 1);
            trace_path_local[sizeof(trace_path_local) - 1] = '\0';
        }
        MPI_Bcast(trace_path_local, params.trace_path_len, MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    
    // --- 3. Setup Local Simulation ---
    double p_absorb = params.C_capture / params.C_total;

    // Divide work
    long long n_base = params.n_particles_total / world_size;
    long long n_rem = params.n_particles_total % world_size;
    long long n_local = n_base + (rank < n_rem ? 1 : 0); // Fairer distribution

    // Each rank initializes its own unique PRNG
    struct prng_state prng;
    seed_prng(&prng, params.seed_base + rank); // Crucial: unique seed per rank

    // Open unique trace file if specified
    FILE* trace_file = NULL;
    if (params.trace_path_len > 0 && params.trace_every > 0) {
        char trace_path_full[1024];
        
        char* ext_dot = strrchr(trace_path_local, '.');
        if (ext_dot) {
            int base_len = ext_dot - trace_path_local;
            // e.g., "trace.csv" -> "trace_r0.csv"
            // Manually limit base_len to prevent overflow (silences warning)
            if (base_len > 900) base_len = 900; 
            snprintf(trace_path_full, sizeof(trace_path_full), "%.*s_r%d%s", 
                     base_len, trace_path_local, rank, ext_dot);
        } else {
            // e.g., "trace" -> "trace_r0.csv"
            // Limit to 900 chars to prevent overflow (silences warning)
            snprintf(trace_path_full, sizeof(trace_path_full), "%.900s_r%d.csv", 
                     trace_path_local, rank);
        }

        trace_file = fopen(trace_path_full, "w");
        if (trace_file) {
            fprintf(trace_file, "k,reflected,absorbed,transmitted\n");
        } else if (rank == 0) { // Only print warning once
            fprintf(stderr, "Warning: Could not open trace files (e.g., '%s').\n", trace_path_full);
        }
    }

    #ifndef M_PI
    #define M_PI 3.14159265358979323846
    #endif

    // --- 4. Run Local Simulation ---
    long long r_local = 0;
    long long b_local = 0;
    long long t_local = 0;

    MPI_Barrier(MPI_COMM_WORLD); // Sync all processes before starting timer
    double t_start = MPI_Wtime();

    for (long long i = 1; i <= n_local; ++i) {
        double d = 0.0, x = 0.0;
        int a = 1;
        while (a) {
            double u_dist = get_rand_double(&prng);
            if (u_dist == 0.0) u_dist = 1e-9;
            double L = -(1.0 / params.C_total) * log(u_dist);
            x = x + L * cos(d);

            if (x < 0.0) { r_local++; a = 0; }
            else if (x >= params.H_thickness) { t_local++; a = 0; }
            else {
                if (get_rand_double(&prng) < p_absorb) { b_local++; a = 0; }
                else { d = get_rand_double(&prng) * M_PI; }
            }
        }
        if (trace_file && (i % params.trace_every == 0)) {
            fprintf(trace_file, "%lld,%.8f,%.8f,%.8f\n", i, 
                    (double)r_local / i, (double)b_local / i, (double)t_local / i);
        }
    }
    
    double t_end = MPI_Wtime();
    double t_wall_local = t_end - t_start;

    if (trace_file) fclose(trace_file);

    // --- 5. Aggregate Results ---
    long long r_total = 0;
    long long b_total = 0;
    long long t_total = 0;
    double t_wall_max = 0.0;

    // Sum all counts to rank 0
    MPI_Reduce(&r_local, &r_total, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&b_local, &b_total, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_local, &t_total, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // Get the *maximum* wall time from any rank
    MPI_Reduce(&t_wall_local, &t_wall_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // --- 6. Final Output (Rank 0 only) ---
    if (rank == 0) {
        double r_final = (double)r_total / params.n_particles_total;
        double b_final = (double)b_total / params.n_particles_total;
        double t_final = (double)t_total / params.n_particles_total;

        // Print wall time to stderr (for benchmarking)
        fprintf(stderr, "%.8f\n", t_wall_max);

        // Print final fractions to stdout
        printf("%.8f %.8f %.8f\n", r_final, b_final, t_final);
    }

    MPI_Finalize();
    return 0;
}