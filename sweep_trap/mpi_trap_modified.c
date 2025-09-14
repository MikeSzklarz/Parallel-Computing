/* mpi_trap4.c: Parallel trapezoidal rule with cmdline args and minimal timing output
 * Compile: mpicc -O2 -Wall -o mpi_trap4 mpi_trap4.c
 * Run:     mpiexec -n <np> ./mpi_trap4 <a> <b> <n>
 * Output (rank 0 only): "<answer> <a> <b> <n> <np> <total_time_seconds>"
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* Build a derived datatype for (a,b,n) so we can broadcast them together */
static void Build_mpi_type(double* a_p, double* b_p, int* n_p,
                           MPI_Datatype* input_mpi_t_p) {
  int bl[3] = {1,1,1};
  MPI_Datatype types[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_INT};
  MPI_Aint disp[3] = {0}, a_addr, b_addr, n_addr;

  MPI_Get_address(a_p, &a_addr);
  MPI_Get_address(b_p, &b_addr);
  MPI_Get_address(n_p, &n_addr);
  disp[0] = 0;
  disp[1] = b_addr - a_addr;
  disp[2] = n_addr - a_addr;

  MPI_Type_create_struct(3, bl, disp, types, input_mpi_t_p);
  MPI_Type_commit(input_mpi_t_p);
}

/* f(x) to integrate */
static double f(double x) { return x*x; }

/* Serial trapezoid on a sub-interval */
static double Trap(double left, double right, int traps, double h) {
  double est = (f(left) + f(right)) * 0.5;
  for (int i = 1; i <= traps-1; i++) {
    est += f(left + i*h);
  }
  return est * h;
}

int main(int argc, char* argv[]) {
  int my_rank, comm_sz;
  double a = 0.0, b = 1.0;
  int n = 0;
  int ok = 1;                 /* broadcast flag: 1=proceed, 0=graceful exit */
  double t0, t1;

  MPI_Init(&argc, &argv);
  t0 = MPI_Wtime();
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

  /* Rank 0 parses inputs; others wait for broadcast */
  if (my_rank == 0) {
    if (argc == 1) {
      /* Graceful exit if no args */
      fprintf(stderr, "Usage: %s <a> <b> <n>\n", argv[0]);
      ok = 0;
    } else if (argc != 4) {
      fprintf(stderr, "Error: expected 3 arguments: <a> <b> <n>\n");
      ok = 0;
    } else {
      char* endp = NULL;
      a = strtod(argv[1], &endp);
      if (endp == argv[1] || *endp != '\0') ok = 0;

      endp = NULL;
      b = strtod(argv[2], &endp);
      if (endp == argv[2] || *endp != '\0') ok = 0;

      endp = NULL;
      long n_long = strtol(argv[3], &endp, 10);
      if (endp == argv[3] || *endp != '\0' || n_long <= 0) ok = 0;
      n = (int)n_long;
      if ((long)n != n_long) ok = 0;

      if (ok && !(b > a)) ok = 0;
    }
  }

  /* Broadcast ok flag so everyone agrees to proceed or exit */
  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (!ok) {
    MPI_Finalize();
    return 0;  /* graceful */
  }

  /* Broadcast the validated (a,b,n) using a derived datatype */
  MPI_Datatype input_t;
  Build_mpi_type(&a, &b, &n, &input_t);
  MPI_Bcast(&a, 1, input_t, 0, MPI_COMM_WORLD);
  MPI_Type_free(&input_t);

  /* Global step size */
  double h = (b - a) / (double)n;

  /* Balanced distribution with remainder:
     First r ranks get (q+1) traps, others get q traps. */
  int q = n / comm_sz;
  int r = n % comm_sz;
  int local_n = q + (my_rank < r ? 1 : 0);

  /* Starting trap index for this rank */
  int start_idx = my_rank * q + (my_rank < r ? my_rank : r);
  double local_a = a + start_idx * h;
  double local_b = local_a + local_n * h;

  double local_int = Trap(local_a, local_b, local_n, h);

  double total_int = 0.0;
  MPI_Reduce(&local_int, &total_int, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  t1 = MPI_Wtime();

  /* Rank 0 prints a single space-separated line:
     <answer> <a> <b> <n> <np> <total_time_seconds> */
  if (my_rank == 0) {
    printf("%.15e %.17g %.17g %d %d %.6f\n",
           total_int, a, b, n, comm_sz, (t1 - t0));
    fflush(stdout);
  }

  MPI_Finalize();
  return 0;
}

