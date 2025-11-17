/*
 * mc_slab.c
 *
 * Serial C implementation of a 2D Monte Carlo neutron transport simulation
 * for a homogeneous slab.
 *
 * Based on the model described in the "Monte Carlo Neutron Transport"
 * assignment for CSCI 473 - Parallel Systems, Fall 2025.
 *
 * Usage:
 * ./mc_slab C Cc H n [--trace-file path] [--trace-every m]
 *
 * Parameters:
 * C  > 0         (total interaction coefficient)
 * Cc in [0, C)  (absorbing component)
 * H  > 0         (slab thickness)
 * n  >= 1        (number of particles to simulate)
 *
 * Optional Flags:
 * --trace-file path  : Path to a CSV file to write running fractions.
 * --trace-every m    : Write to trace file every 'm' particles.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> // For strcmp
#include <math.h>   // For log, cos, M_PI
#include <time.h>   // For time() to seed rand()
#include <errno.h>  // For error checking

// Helper function to get a uniform random number in [0, 1)
double get_rand() {
    return (double)rand() / (double)RAND_MAX;
}

// Prints the required usage string from the assignment PDF
void print_usage() {
    printf("Usage: ./mc_slab C Cc H n [--trace-file path] [--trace-every m] [--seed s]\n");
    printf("  C > 0         (total interaction coeff)\n");
    printf("  Cc in [0,C)   (absorbing component)\n");
    printf("  H > 0         (slab thickness)\n");
    printf("  n >= 1        (number of particles)\n");
    printf("  --seed s      (optional: random number seed)\n");
}

int main(int argc, char *argv[]) {
    // --- 1. Argument Parsing and Validation ---

    if (argc < 5) {
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
    FILE* trace_file = NULL;

    // Optional seed argument
    long long seed_val = 0;
    int seed_provided = 0;

    // Parse optional arguments
    for (int i = 5; i < argc; ++i) {
        if (strcmp(argv[i], "--trace-file") == 0) {
            if (i + 1 < argc) {
                trace_path = argv[++i];
            } else {
                printf("Error: --trace-file requires a path.\n");
                print_usage();
                return 1;
            }
        } else if (strcmp(argv[i], "--trace-every") == 0) {
            if (i + 1 < argc) {
                trace_every = atoll(argv[++i]);
            } else {
                printf("Error: --trace-every requires a number.\n");
                print_usage();
                return 1;
            }
        } else if (strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed_val = atoll(argv[++i]);
                seed_provided = 1;
            } else {
                printf("Error: --seed requires a number.\n");
                print_usage();
                return 1;
            }
        } else {
            printf("Error: Unknown argument '%s'.\n", argv[i]);
            print_usage();
            return 1;
        }
    }

    // Validate parameters (as shown on page 17)
    if (C_total <= 0) {
        printf("Error: C must be > 0.\n");
        print_usage();
        return 1;
    }
    if (C_capture < 0 || C_capture >= C_total) {
        printf("Error: Cc must be in [0, C).\n");
        print_usage();
        return 1;
    }
    if (H_thickness <= 0) {
        printf("Error: H must be > 0.\n");
        print_usage();
        return 1;
    }
    if (n_particles < 1) {
        printf("Error: n must be >= 1.\n");
        print_usage();
        return 1;
    }
    if (trace_path && trace_every <= 0) {
        printf("Error: --trace-every must be > 0 when --trace-file is used.\n");
        print_usage();
        return 1;
    }

    // --- 2. Setup Simulation ---

    // Seed the random number generator
    if (seed_provided) {
        srand(seed_val);
    } else {
        srand(time(NULL));
    }

    // Open trace file if specified
    if (trace_path) {
        trace_file = fopen(trace_path, "w");
        if (!trace_file) {
            fprintf(stderr, "Error: Could not open trace file '%s'.\n", trace_path);
            // Continue without tracing
        } else {
            // Write header for the convergence plot CSV (see page 38)
            fprintf(trace_file, "k,reflected,absorbed,transmitted\n");
        }
    }

    // Event counters (reflected, absorbed, transmitted)
    long long r_count = 0;
    long long b_count = 0; // 'b' for absorbed (blocked)
    long long t_count = 0;

    // Pre-calculate absorption probability
    double p_absorb = C_capture / C_total;
    
    // Ensure M_PI is defined. If not, define it.
    #ifndef M_PI
    #define M_PI 3.14159265358979323846
    #endif

    // --- 3. Run Simulation ---
    // Follows pseudocode from page 16, but uses fresh random numbers
    // for each decision, as is correct for Monte Carlo methods.

    for (long long i = 1; i <= n_particles; ++i) {
        double d = 0.0;     // Initial direction (0 radians)
        double x = 0.0;     // Initial position (left edge of slab)
        int a = 1;          // 'a' for "alive" (true)

        while (a) {
            // Calculate distance to next interaction
            // L = -(1/C) * ln(u)
            double u_dist = get_rand();
            // Avoid log(0) which is -inf
            if (u_dist == 0.0) u_dist = 1e-9; 
            
            double L = -(1.0 / C_total) * log(u_dist);

            // Update particle position
            // x = x + L * cos(d)
            x = x + L * cos(d);

            // Check particle status
            if (x < 0.0) {
                // Reflected
                r_count++;
                a = 0; // Particle is no longer "alive"
            } else if (x >= H_thickness) {
                // Transmitted
                t_count++;
                a = 0; // Particle is no longer "alive"
            } else {
                // Interaction within the slab
                double u_event = get_rand();
                
                if (u_event < p_absorb) {
                    // Absorbed
                    b_count++;
                    a = 0; // Particle is no longer "alive"
                } else {
                    // Scattered
                    // New direction d = u * pi
                    double u_dir = get_rand();
                    d = u_dir * M_PI;
                }
            }
        } // end while(a)

        // --- 4. Tracing ---
        if (trace_file && (i % trace_every == 0)) {
            double r_frac = (double)r_count / i;
            double b_frac = (double)b_count / i;
            double t_frac = (double)t_count / i;
            fprintf(trace_file, "%lld,%.8f,%.8f,%.8f\n", i, r_frac, b_frac, t_frac);
        }

    } // end for(n_particles)

    // --- 5. Final Output ---

    // Close the trace file if it was opened
    if (trace_file) {
        fclose(trace_file);
    }

    // Calculate final fractions
    double r_final = (double)r_count / n_particles;
    double b_final = (double)b_count / n_particles;
    double t_final = (double)t_count / n_particles;

    // Print final fractions to stdout in the format specified on page 19
    // (reflected absorbed transmitted)
    printf("%.8f %.8f %.8f\n", r_final, b_final, t_final);

    return 0;
}