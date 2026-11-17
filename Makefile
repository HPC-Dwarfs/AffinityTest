# Builds the three variants of MPI_pin_test.
#   make                                  # mpicc, GCC-style -fopenmp
#   make MPICC=mpiicx OPENMP=-qopenmp     # Intel oneAPI
MPICC  ?= mpicc
CFLAGS ?= -O2 -Wall -Wextra
OPENMP ?= -fopenmp

all: pin_omp pin_pt pin_mpi

pin_omp: main.c
	$(MPICC) $(CFLAGS) $(OPENMP) -o $@ $<

pin_pt: main.c
	$(MPICC) $(CFLAGS) -DPTHREADS -pthread -o $@ $<

pin_mpi: main.c
	$(MPICC) $(CFLAGS) -o $@ $<

clean:
	rm -f pin_omp pin_pt pin_mpi

.PHONY: all clean
