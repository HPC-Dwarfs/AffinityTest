# AffinityTest

A small MPI program that shows where the ranks of a (hybrid) MPI job and their
threads actually run. For every MPI rank, and for every thread of that rank, it
prints the host, process and thread IDs, the CPU the thread is running on, the
NUMA node of that CPU, and the full CPU affinity mask. With OpenMP it also
prints the OpenMP place of each thread.

Use it to check that the launcher (`mpirun`, `srun`, ...) and the OpenMP
runtime pin your job the way you think they do, before you spend time on
performance measurements.

The program supports three variants, selected at compile time:

| Variant        | Threads per rank                     | Who pins the threads             |
| -------------- | ------------------------------------ | -------------------------------- |
| MPI + OpenMP   | OpenMP team (`OMP_NUM_THREADS`)      | MPI launcher + OpenMP runtime    |
| MPI + pthreads | `PTHREAD_THREADS` (default 4)        | MPI launcher only (threads inherit the rank mask) |
| Pure MPI       | none                                 | MPI launcher                     |

## Requirements

- Linux (uses `sched_getaffinity`, `sched_getcpu` and the `getcpu` syscall).
  The build stops with an error on any other OS.
- An MPI library with an `mpicc` wrapper (MPI-3 or newer, for
  `MPI_Comm_split_type`).
- Optional: `pstree` (package `psmisc`) for the `-ps` option.

## Building

```sh
mpicc -O2 -fopenmp            -o pin_omp main.c   # MPI + OpenMP
mpicc -O2 -DPTHREADS -pthread -o pin_pt  main.c   # MPI + pthreads
mpicc -O2                     -o pin_mpi main.c   # pure MPI
```

OpenMP and `-DPTHREADS` cannot be combined. With the Intel or LLVM
compilers, use their OpenMP flag (`-qopenmp`, `-fopenmp`) as usual.

## Running

```sh
# MPI + OpenMP
OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=close mpirun -np 2 ./pin_omp

# MPI + pthreads
PTHREAD_THREADS=4 mpirun -np 2 ./pin_pt

# pure MPI
mpirun -np 8 ./pin_mpi

# Slurm example
export OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=close
srun -N 2 --ntasks-per-node=2 --cpus-per-task=4 --cpu-bind=cores ./pin_omp
```

Make sure the environment variables reach all ranks (for example with
`mpirun -x OMP_NUM_THREADS ...` in Open MPI, or `-genv` in MPICH and Intel
MPI). The program only prints rank 0's OpenMP settings.

### Options and environment variables

| Name              | Meaning |
| ----------------- | ------- |
| `-ps`             | Also print the process tree of each node (see [Process tree](#process-tree--ps)). Only rank 0's command line is checked. |
| `PTHREAD_THREADS` | Number of threads per rank in the pthreads variant, including the main thread. Range 1 to 4096, default 4. |
| `OMP_NUM_THREADS`, `OMP_PLACES`, `OMP_PROC_BIND` | Standard OpenMP controls. Rank 0's values are echoed in the output header. |

## How it works

1. **MPI setup.** Each rank calls `MPI_Init_thread`. It requests
   `MPI_THREAD_FUNNELED` in the threaded variants and `MPI_THREAD_SINGLE` in
   the pure MPI variant. `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)` groups
   the ranks of each node into a node communicator. This gives every rank a
   *local rank* within its node and lets rank 0 count the nodes.
2. **Process sample.** Before any threads start, each rank records one line for
   itself (the `process` line).
3. **Thread samples.** The rank starts its threads: an OpenMP parallel region,
   or `PTHREAD_THREADS - 1` extra pthreads plus the main thread. All threads
   wait at a barrier so the whole team is running, then each thread records one
   line about itself. Because all threads are alive at the same time, the
   CPUs they report are where they run *together*, not one after another.
4. **Process tree (optional).** With `-ps`, thread 0 of every rank waits at a
   node-wide barrier while its other threads are still alive. Local rank 0
   then runs `pstree -p` on its parent process, so the tree shows every rank
   of the node with all its threads.
5. **Output.** All lines are gathered on rank 0 with `MPI_Gatherv`. Ranks may
   have different thread counts. Rank 0 prints the lines in rank and thread
   order, followed by the process trees, one per node.

Each line is a snapshot of the system calls the thread itself makes:

- **Affinity mask:** `sched_getaffinity(0, ...)` for the calling thread.
- **Current CPU:** `sched_getcpu()`.
- **NUMA node:** the `node` result of the `getcpu` syscall, that is, the NUMA
  domain of the current CPU.
- **OpenMP place:** `omp_get_place_num()` / `omp_get_num_places()`.

## Interpreting the output

### Header

```
MPI_pin_test: 2 rank(s) on 1 node(s), thread support MPI_THREAD_FUNNELED
OpenMP: OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=close (rank 0 env)
```

- The number of ranks and the number of distinct nodes (shared memory
  domains, as seen by MPI).
- The thread support level the MPI library provided. If it is lower than the
  requested level, the line ends with `(WARNING: requested ...)`.
- OpenMP variant only: the OpenMP environment of rank 0, `(unset)` if a
  variable is not set.

### Per-rank and per-thread lines

In the threaded variants, each rank is a block separated by a blank line: one
`process` line followed by one line per thread. In the pure MPI variant there
is just one `process` line per rank.

```
rank   0 (local  0) thread 1/4    host node01  pid 20711  tid 20715  cpu   1  numa  0  place 1/4  mask[1] {1}
```

| Field           | Meaning |
| --------------- | ------- |
| `rank 0`        | Rank in `MPI_COMM_WORLD`. |
| `(local 0)`     | Rank within its node. |
| `process`       | The rank itself, sampled before the threads start. |
| `thread 1/4`    | Thread number and team size. Thread 0 is the main thread (the one that called `MPI_Init_thread`). |
| `host node01`   | Name returned by `MPI_Get_processor_name`. |
| `pid` / `tid`   | Linux process ID and thread ID. Thread 0's `tid` equals the `pid`. The other `tid`s are the IDs that `top -H`, `ps -L` and `pstree -p` show. |
| `cpu 1`         | Logical CPU (hardware thread) the thread was running on when it was sampled. |
| `numa 0`        | NUMA node of that CPU. |
| `place 1/4`     | OpenMP only: the thread's place number and the total number of places. `-1` means the thread is not bound to a place (no `OMP_PROC_BIND`/`OMP_PLACES` in effect). |
| `mask[1] {1}`   | Affinity mask: the number of logical CPUs the thread is allowed to run on, followed by that set as a compact range list such as `{0-3,64-67}`. |

**The mask is what matters; `cpu` is only a snapshot.** If the mask holds more
than one CPU, the OS scheduler may move the thread anywhere inside the mask at
any time, and `cpu` will vary from run to run. Only a mask of a single CPU (or
a single core, see SMT below) means the thread is actually pinned.

**About the `process` line.** It shows the affinity of the main thread before
the threaded region starts, which is normally the mask the MPI launcher gave
the rank. Some OpenMP runtimes already bind the main thread to its first place
when they initialize. In that case this line shows the narrower mask of
thread 0 instead.

### What good pinning looks like

Two ranks, four OpenMP threads each, on a 2-socket node with 8 cores per
socket (CPUs 0-7 on NUMA node 0, CPUs 8-15 on NUMA node 1, no SMT). The
launcher binds each rank to 4 cores on its own socket, and
`OMP_PLACES=cores OMP_PROC_BIND=close` pins one thread per core
(illustrative output):

```
rank   0 (local  0) process       host node01  pid 20711  tid 20711  cpu   0  numa  0  mask[4] {0-3}
rank   0 (local  0) thread 0/4    host node01  pid 20711  tid 20711  cpu   0  numa  0  place 0/4  mask[1] {0}
rank   0 (local  0) thread 1/4    host node01  pid 20711  tid 20715  cpu   1  numa  0  place 1/4  mask[1] {1}
rank   0 (local  0) thread 2/4    host node01  pid 20711  tid 20716  cpu   2  numa  0  place 2/4  mask[1] {2}
rank   0 (local  0) thread 3/4    host node01  pid 20711  tid 20717  cpu   3  numa  0  place 3/4  mask[1] {3}

rank   1 (local  1) process       host node01  pid 20712  tid 20712  cpu   8  numa  1  mask[4] {8-11}
rank   1 (local  1) thread 0/4    host node01  pid 20712  tid 20712  cpu   8  numa  1  place 0/4  mask[1] {8}
rank   1 (local  1) thread 1/4    host node01  pid 20712  tid 20718  cpu   9  numa  1  place 1/4  mask[1] {9}
rank   1 (local  1) thread 2/4    host node01  pid 20712  tid 20719  cpu  10  numa  1  place 2/4  mask[1] {10}
rank   1 (local  1) thread 3/4    host node01  pid 20712  tid 20720  cpu  11  numa  1  place 3/4  mask[1] {11}
```

Checklist:

- The `process` masks of ranks on the same node do not overlap.
- Each thread's mask is a subset of its rank's `process` mask.
- Every thread has a mask of size 1 (or one core), and no two threads of the
  node share a mask.
- The number of places matches the number of threads per rank.
- All threads of a rank are on the NUMA node(s) you intended.

### Common problems and how they show up

**All threads stuck on one core.** The launcher bound each rank to a single
core (Open MPI's default for small rank counts, for example), and the OpenMP
threads have nowhere else to go:

```
rank   0 (local  0) process       host node01  pid 20811  tid 20811  cpu   0  numa  0  mask[1] {0}
rank   0 (local  0) thread 0/4    host node01  pid 20811  tid 20811  cpu   0  numa  0  place 0/1  mask[1] {0}
rank   0 (local  0) thread 1/4    host node01  pid 20811  tid 20815  cpu   0  numa  0  place 0/1  mask[1] {0}
...
```

Every thread reports the same single-CPU mask and only one place exists. Give
each rank as many cores as it has threads (for example
`mpirun --map-by socket:PE=4`, `srun --cpus-per-task=4`).

**No pinning at all.** The masks span the whole node (e.g. `mask[16] {0-15}`),
`place` is `-1`, and `cpu` changes between runs. Threads can migrate freely,
including across NUMA domains. Set `OMP_PROC_BIND` and `OMP_PLACES`, or bind
in the launcher.

**Overlapping ranks.** Two ranks on the same node have overlapping or identical
`process` masks, so they compete for the same cores. Check the launcher's
mapping and binding options and `--cpus-per-task`.

**Rank spans NUMA domains unintentionally.** A rank's mask contains CPUs from
two NUMA nodes (visible as different `numa` values among its threads). This
is sometimes intended, but often it means the mapping is not aligned to
sockets or NUMA domains.

**Fewer threads than expected.** The OpenMP header shows the requested
`OMP_NUM_THREADS`, and the thread lines show the team size actually created
(`thread i/N`). If `OMP_NUM_THREADS` is not exported to all ranks, other ranks
may use a different default.

**pthreads variant.** The program does not pin pthreads itself, so every
thread inherits the rank's mask. Expect all thread lines of a rank to show the
same mask as its `process` line. This is what a threaded code that does no
pinning of its own gets from the launcher. If that mask is a single CPU, all
threads share one core.

### SMT / hyper-threading

On systems with SMT, one core contains several logical CPUs. Linux often
numbers them as `N` and `N + number_of_cores`. With `OMP_PLACES=cores`, each
thread is pinned to a whole core, so a mask like `mask[2] {0,64}` is correct:
the thread can run on either hardware thread of core 0. Use
`OMP_PLACES=threads` to pin to a single hardware thread. Use `lscpu -e` or
`likwid-topology` to see how logical CPUs map to cores, sockets and NUMA
domains on your machine.

### Process tree (`-ps`)

With `-ps`, the per-rank listing is followed by one block per node:

```
--- process tree on node01 (parent pid 20700 of rank 0) ---
prted(20700)─┬─pin_omp(20711)─┬─{pin_omp}(20715)
             │                ├─{pin_omp}(20716)
             ...
```

The tree starts at the parent of the node's first rank, which is typically the
launcher's node daemon (`slurmstepd`, `prted`/`orted`, `hydra_pmi_proxy`).
Thread IDs appear in braces and match the `tid` values above. This shows which
processes belong to the job and catches leftover or unexpected processes on
the node. If `pstree` is not installed, the block says
`(pstree not available)`. Output longer than about 16 KB per node is cut off.

## Cross-checking with other tools

- Open MPI: `mpirun --report-bindings`
- Intel MPI: `I_MPI_DEBUG=4`
- Slurm: `srun --cpu-bind=verbose`
- OpenMP 5.0+: `OMP_DISPLAY_AFFINITY=true`
- Node topology: `lscpu`, `numactl --hardware`, `likwid-topology`, `hwloc-ls`

## License

MIT, see [LICENSE](LICENSE).
