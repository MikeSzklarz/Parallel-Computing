/*
 * mc_slab_pthreads.c
 *
 * Pthreads parallel C implementation of a 2D Monte Carlo neutron transport.
 *
 * This implementation goes "above and beyond" by using a high-quality,
 * thread-safe xorshift128+ PRNG instead of the weaker rand_r().
 *
 * Each thread simulates its share of particles and aggregates results.
 * Tracing is supported on a per-thread basis (e.g., trace_t0.csv, trace_t1.csv).
 *
 * Usage:
 * ./mc_slab_pthreads C Cc H n [--trace-file path] [--trace-every m] [--seed s] T
 *
 * Parameters:
 * C  > 0         (total interaction coeff)
 * Cc in [0, C)  (absorbing component)
 * H  > 0         (slab thickness)
 * n  >= 1        (number of particles to simulate)
 * T  >= 1        (number of threads; REQUIRED and must be the LAST argument)
 *
 * Optional Flags:
 * --trace-file path  : Base path for trace CSVs (e.g., "trace" -> "trace_t0.csv").
 * --trace-every m    : Write to trace file every 'm' particles.
 * --seed s           : Base random number seed.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> // For strcmp, strrchr
#include <math.h>   // For log, cos, M_PI
#include <time.h>   // For time() and clock_gettime
#include <errno.h>  // For error checking
#include <pthread.h>
#include <stdint.h> // For uint64_t

// --- High-Quality PRNG (xorshift128+) ---
// This is superior to rand() or rand_r() for simulation.

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


// --- Threading Structures ---

typedef struct {
    // Inputs
    int thread_id;
    long long n_particles_local;
    double C_total;
    double C_capture;
    double H_thickness;
    double p_absorb;
    uint64_t seed;
    char* trace_base_path;
    long long trace_every;

    // Outputs
    long long r_count;
    long long b_count;
    long long t_count;
} thread_data_t;


// Prints the required usage string
void print_usage() {
    printf("Usage: ./mc_slab_pthreads C Cc H n [--trace-file path] [--trace-every m] [--seed s] T\n");
    printf("  C > 0         (total interaction coeff)\n");
    printf("  Cc in [0,C)   (absorbing component)\n");
    printf("  H > 0         (slab thickness)\n");
    printf("  n >= 1        (number of particles)\n");
    printf("  T >= 1        (number of threads; REQUIRED last arg)\n");
    printf("  --seed s      (optional: random number seed)\n");
    printf("  --trace-file  (optional: base path for trace files)\n");
    printf("  --trace-every (optional: trace output frequency)\n");
}

// Get wall time
double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// The core simulation logic, run by each thread
void* simulate_particles_thread(void* arg) {
    thread_data_t *data = (thread_data_t*)arg;

    // --- 1. Initialize Thread-Local State ---
    data->r_count = 0;
    data->b_count = 0;
    data->t_count = 0;

    // Initialize this thread's PRNG state
    struct prng_state prng;
    seed_prng(&prng, data->seed); // Seed is already unique (base + thread_id)

    // Open unique trace file if specified
    FILE* trace_file = NULL;
    if (data->trace_base_path && data->trace_every > 0) {
        char trace_path_full[1024];
        
        // Check if path has an extension
        char* ext_dot = strrchr(data->trace_base_path, '.');
        if (ext_dot) {
            // Insert thread ID before extension, e.g., "trace.csv" -> "trace_t0.csv"
            int base_len = ext_dot - data->trace_base_path;
            snprintf(trace_path_full, sizeof(trace_path_full), "%.*s_t%d%s", 
                     base_len, data->trace_base_path, data->thread_id, ext_dot);
        } else {
            // Just append, e.g., "trace" -> "trace_t0.csv"
            snprintf(trace_path_full, sizeof(trace_path_full), "%s_t%d.csv", 
                     data->trace_base_path, data->thread_id);
        }

        trace_file = fopen(trace_path_full, "w");
        if (trace_file) {
            fprintf(trace_file, "k,reflected,absorbed,transmitted\n");
        } else {
            fprintf(stderr, "Warning: Thread %d could not open trace file '%s'.\n", 
                    data->thread_id, trace_path_full);
        }
    }

    // Ensure M_PI is defined
    #ifndef M_PI
    #define M_PI 3.14159265358979323846
    #endif

    // --- 2. Run Simulation ---
    for (long long i = 1; i <= data->n_particles_local; ++i) {
        double d = 0.0;     // Initial direction (0 radians)
        double x = 0.0;     // Initial position (left edge of slab)
        int a = 1;          // "alive"

        while (a) {
            double u_dist = get_rand_double(&prng);
            if (u_dist == 0.0) u_dist = 1e-9; // Avoid log(0)
            
            double L = -(1.0 / data->C_total) * log(u_dist);
            x = x + L * cos(d);

            if (x < 0.0) {
                data->r_count++;
                a = 0;
            } else if (x >= data->H_thickness) {
                data->t_count++;
                a = 0;
            } else {
                double u_event = get_rand_double(&prng);
                if (u_event < data->p_absorb) {
                    data->b_count++;
                    a = 0;
                } else {
                    double u_dir = get_rand_double(&prng);
                    d = u_dir * M_PI;
                }
            }
        } // end while(a)

        // --- 3. Tracing (Thread-Local) ---
        if (trace_file && (i % data->trace_every == 0)) {
            // Note: This traces fractions based on *this thread's* counts
            double r_frac = (double)data->r_count / i;
            double b_frac = (double)data->b_count / i;
            double t_frac = (double)data->t_count / i;
            fprintf(trace_file, "%lld,%.8f,%.8f,%.8f\n", i, r_frac, b_frac, t_frac);
        }

    } // end for(n_particles_local)

    if (trace_file) {
        fclose(trace_file);
    }

    return NULL;
}


int main(int argc, char *argv[]) {
    // --- 1. Argument Parsing and Validation ---

    // Last arg MUST be T
    if (argc < 6) { // 5 required + T
        print_usage();
        return 1;
    }

    // Required arguments
    double C_total = atof(argv[1]);
    double C_capture = atof(argv[2]);
    double H_thickness = atof(argv[3]);
    long long n_particles = atoll(argv[4]);
    
    // Optional trace arguments
    char* trace_path = NULL;
    long long trace_every = 0;
    
    // Optional seed argument
    uint64_t seed_val = 0;
    int seed_provided = 0;

    // Last arg is T, parse it first
    int T_threads = atoi(argv[argc - 1]);
    
    // Parse optional arguments (stop before T)
    for (int i = 5; i < argc - 1; ++i) {
        if (strcmp(argv[i], "--trace-file") == 0) {
            if (i + 1 < argc - 1) trace_path = argv[++i];
        } else if (strcmp(argv[i], "--trace-every") == 0) {
            if (i + 1 < argc - 1) trace_every = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc - 1) {
                seed_val = (uint64_t)atoll(argv[++i]);
                seed_provided = 1;
            }
        } else {
            fprintf(stderr, "Error: Unknown argument '%s'.\n", argv[i]);
            print_usage();
            return 1;
        }
    }

    // Validate parameters
    if (C_total <= 0 || C_capture < 0 || C_capture >= C_total || 
        H_thickness <= 0 || n_particles < 1 || T_threads < 1) {
        fprintf(stderr, "Error: Invalid parameters.\n");
        print_usage();
        return 1;
    }
    if (trace_path && trace_every <= 0) {
        fprintf(stderr, "Error: --trace-every must be > 0 when --trace-file is used.\n");
        return 1;
    }

    // --- 2. Setup Simulation ---
    if (!seed_provided) {
        seed_val = (uint64_t)time(NULL);
    }
    
    double p_absorb = C_capture / C_total;

    // --- 3. Thread Creation & Execution ---
    double t_start = get_time_sec();

    pthread_t* threads = malloc(T_threads * sizeof(pthread_t));
    thread_data_t* thread_data = malloc(T_threads * sizeof(thread_data_t));
    if (!threads || !thread_data) {
        fprintf(stderr, "Error: Could not allocate memory for threads.\n");
        return 1;
    }

    // Divide work
    long long n_base = n_particles / T_threads;
    long long n_rem = n_particles % T_threads;

    for (int i = 0; i < T_threads; ++i) {
        thread_data[i].thread_id = i;
        thread_data[i].n_particles_local = n_base + (i < n_rem ? 1 : 0); // Fairer distribution
        thread_data[i].C_total = C_total;
        thread_data[i].C_capture = C_capture;
        thread_data[i].H_thickness = H_thickness;
        thread_data[i].p_absorb = p_absorb;
        thread_data[i].seed = seed_val + i; // Crucial: unique seed per thread
        thread_data[i].trace_base_path = trace_path;
        thread_data[i].trace_every = trace_every;

        int rc = pthread_create(&threads[i], NULL, simulate_particles_thread, &thread_data[i]);
        if (rc) {
            fprintf(stderr, "Error: pthread_create() failed with code %d\n", rc);
            return 1;
        }
    }

    // --- 4. Join Threads and Aggregate Results ---
    long long r_total = 0;
    long long b_total = 0;
    long long t_total = 0;

    for (int i = 0; i < T_threads; ++i) {
        pthread_join(threads[i], NULL);
        // Aggregate counts from each thread
        r_total += thread_data[i].r_count;
        b_total += thread_data[i].b_count;
        t_total += thread_data[i].t_count;
    }
    
    double t_end = get_time_sec();
    double t_wall = t_end - t_start;

    // --- 5. Final Output ---
    free(threads);
    free(thread_data);

    // Calculate final fractions
    double r_final = (double)r_total / n_particles;
    double b_final = (double)b_total / n_particles;
    double t_final = (double)t_total / n_particles;

    // Print wall time to stderr (for benchmarking)
    fprintf(stderr, "%.8f\n", t_wall);

    // Print final fractions to stdout
    printf("%.8f %.8f %.8f\n", r_final, b_final, t_final);

    return 0;
}