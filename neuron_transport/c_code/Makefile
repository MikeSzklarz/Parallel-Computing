# Makefile for mc_slab
# Replicates the compile command from page 17 of the assignment PDF.

# Compiler and flags
CC = gcc
# Use C99 standard, level 2 optimization, all warnings, extra warnings,
# and pedantic mode to be strictly compliant.
CFLAGS = -std=c99 -O2 -Wall -Wextra -pedantic
# Link with the math library (-lm) for log() and cos()
LDLIBS = -lm

# Target executable
TARGET = mc_slab

# Default target: 'all'
all: $(TARGET)

# Rule to build the target executable
$(TARGET): mc_slab.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

# Target to clean up build files
clean:
	rm -f $(TARGET)

.PHONY: all clean