# Makefile for the 2D Stencil Project

# --- Variables ---
CC = gcc
# Use -O3 for optimized release builds, or -g for debugging
# CFLAGS = -g -Wall -Wstrict-prototypes -std=gnu99
CFLAGS = -O3 -Wall -Wstrict-prototypes -std=gnu99 -march=native
LFLAGS = -lm

# List of all program executables to build
PROGS = print-2d make-2d stencil-2d stencil-2d-pth

# --- Targets ---

# The default target, 'all', builds all programs
all: $(PROGS)

# --- Linking Programs ---
# Each program depends on its own object file and the shared utilities object file
print-2d: utilities.o print-2d.o
	$(CC) $(CFLAGS) $(LFLAGS) -o print-2d utilities.o print-2d.o

make-2d: utilities.o make-2d.o
	$(CC) $(CFLAGS) $(LFLAGS) -o make-2d utilities.o make-2d.o

stencil-2d: utilities.o stencil-2d.o
	$(CC) $(CFLAGS) $(LFLAGS) -o stencil-2d utilities.o stencil-2d.o

stencil-2d-pth: utilities.o stencil-2d-pth.o
	$(CC) $(CFLAGS) $(LFLAGS) -o stencil-2d-pth utilities.o stencil-2d-pth.o -pthread

# --- Compiling Object Files ---
# Each .c file depends on its .c source and any .h files it includes

utilities.o: utilities.c utilities.h
	$(CC) $(CFLAGS) -c utilities.c

print-2d.o: print-2d.c utilities.h
	$(CC) $(CFLAGS) -c print-2d.c

make-2d.o: make-2d.c utilities.h
	$(CC) $(CFLAGS) -c make-2d.c

stencil-2d.o: stencil-2d.c utilities.h
	$(CC) $(CFLAGS) -c stencil-2d.c

# --- THIS IS THE CORRECTED RULE ---
stencil-2d-pth.o: stencil-2d-pth.c utilities.h
	$(CC) $(CFLAGS) -c stencil-2d-pth.c -o $@

# --- Clean Target ---
# This is the target that was missing.
# It removes all compiled object files (*.o), core dump files,
# and the program executables.
clean:
	rm -f *.o core* $(PROGS)