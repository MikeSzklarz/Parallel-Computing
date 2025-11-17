# Makefile for MCNT simulation
# Builds serial, pthreads, and MPI versions.

# Compiler and flags
CC = gcc
MPICC = mpicc
# Use C99 standard, level 2 optimization, all warnings, extra warnings,
# and pedantic mode to be strictly compliant.
CFLAGS = -std=c99 -O2 -Wall -Wextra -pedantic
# Link with the math library (-lm) for log() and cos()
LDLIBS = -lm
# Pthreads specific flags
PTHREAD_CFLAGS = $(CFLAGS) -pthread
PTHREAD_LDLIBS = $(LDLIBS) -pthread

# Targets
SERIAL_TARGET = mc_slab
PTHREAD_TARGET = mc_slab_pthreads
MPI_TARGET = mc_slab_mpi

# Default target: 'all'
all: $(SERIAL_TARGET) $(PTHREAD_TARGET) $(MPI_TARGET)

# --- Build Rules ---

# Serial version
$(SERIAL_TARGET): mc_slab.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

# Pthreads version
$(PTHREAD_TARGET): mc_slab_pthreads.c
	$(CC) $(PTHREAD_CFLAGS) $< -o $@ $(PTHREAD_LDLIBS)

# MPI version
$(MPI_TARGET): mc_slab_mpi.c
	$(MPICC) $(CFLAGS) $< -o $@ $(LDLIBS)

# --- Cleanup ---

clean:
	rm -f $(SERIAL_TARGET) $(PTHREAD_TARGET) $(MPI_TARGET) *.o

.PHONY: all clean