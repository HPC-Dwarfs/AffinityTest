/*
 * main.c - MPI_pin_test: show where MPI ranks and their threads run (Linux only).
 *
 * For every rank (and every thread) it prints: host, pid/tid, the CPU it is
 * currently running on, the NUMA node of that CPU and the complete affinity
 * mask. With OpenMP the OpenMP place is printed as well. All output is
 * collected on rank 0 and printed in rank/thread order.
 *
 * Build (or simply "make"):
 *   mpicc -O2 -fopenmp            -o pin_omp main.c   # MPI + OpenMP
 *   mpicc -O2 -DPTHREADS -pthread -o pin_pt  main.c   # MPI + pthreads
 *   mpicc -O2                     -o pin_mpi main.c   # pure MPI
 *
 * Run:
 *   OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=close mpirun -np 2 ./pin_omp
 *   PTHREAD_THREADS=4 mpirun -np 2 ./pin_pt
 *   Add -ps to also print the process tree (requires pstree) once per node.
 *
 * Note: "cpu" is only a snapshot. If the mask contains more than one CPU the
 * OS may move the thread at any time; the mask is what pinning controls.
 */
#ifndef __linux__
#error "MPI_pin_test requires Linux (sched_getaffinity, sched_getcpu, getcpu)"
#endif
#if defined(_OPENMP) && defined(PTHREADS)
#error "Build either with OpenMP or with -DPTHREADS, not both"
#endif

#define _GNU_SOURCE
#include <errno.h>
#include <limits.h>
#include <sched.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef PTHREADS
#include <pthread.h>
#endif

#define MAX_PTHREADS 4096

#if defined(_OPENMP) || defined(PTHREADS)
#define THREADED     1
#define THREAD_LEVEL MPI_THREAD_FUNNELED
#else
#define THREADED     0
#define THREAD_LEVEL MPI_THREAD_SINGLE
#endif

/* growable, always NUL-terminated string (once something was appended) */
typedef struct {
    char* s;
    size_t len;
    size_t cap;
} Str;

typedef struct {
    char host[MPI_MAX_PROCESSOR_NAME];
    int hostWidth; /* longest host name in the job, for aligned columns */
    int rank;      /* rank in MPI_COMM_WORLD */
    int localRank; /* rank within the node */
    int doPstree;  /* -ps given */
    size_t cpuBits; /* number of CPUs our CPU sets can hold */
    MPI_Comm nodeComm;
    Str* pstree; /* filled on node-local rank 0 */
} Context;

static void outOfMemory(void)
{
    fprintf(stderr, "MPI_pin_test: out of memory\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
}

static void* xcalloc(size_t n, size_t size)
{
    void* p = calloc(n, size);
    if (p == NULL && n > 0 && size > 0) outOfMemory();
    return p;
}

static void* xrealloc(void* p, size_t size)
{
    void* q = realloc(p, size);
    if (q == NULL) outOfMemory();
    return q;
}

/* Append printf-style text to b, growing it as needed. */
static void strAppendf(Str* b, const char* fmt, ...)
{
    if (b->cap == 0) {
        b->cap = 256;
        b->s   = xrealloc(NULL, b->cap);
        b->s[0] = '\0';
    }
    for (;;) {
        va_list ap;
        size_t room = b->cap - b->len;
        va_start(ap, fmt);
        int w = vsnprintf(b->s + b->len, room, fmt, ap);
        va_end(ap);
        if (w < 0) return;
        if ((size_t)w < room) {
            b->len += (size_t)w;
            return;
        }
        b->cap = 2 * (b->len + (size_t)w + 1);
        b->s   = xrealloc(b->s, b->cap);
    }
}

static int myGettid(void) { return (int)syscall(SYS_gettid); }

static const char* threadLevelName(int level)
{
    if (level == MPI_THREAD_SINGLE) return "MPI_THREAD_SINGLE";
    if (level == MPI_THREAD_FUNNELED) return "MPI_THREAD_FUNNELED";
    if (level == MPI_THREAD_SERIALIZED) return "MPI_THREAD_SERIALIZED";
    if (level == MPI_THREAD_MULTIPLE) return "MPI_THREAD_MULTIPLE";
    return "unknown";
}

/* Allocate an empty CPU set for bits CPUs. */
static cpu_set_t* xcpuAlloc(size_t bits)
{
    cpu_set_t* set = CPU_ALLOC(bits);
    if (set == NULL) outOfMemory();
    CPU_ZERO_S(CPU_ALLOC_SIZE(bits), set);
    return set;
}

/*
 * Number of CPUs a CPU set must hold so that sched_getaffinity succeeds.
 * The kernel rejects sets smaller than its own CPU mask, which can be larger
 * than the fixed CPU_SETSIZE (1024) of cpu_set_t.
 */
static size_t probeCpuBits(void)
{
    long conf   = sysconf(_SC_NPROCESSORS_CONF);
    size_t bits = (conf > CPU_SETSIZE) ? (size_t)conf : CPU_SETSIZE;

    for (; bits <= ((size_t)1 << 22); bits *= 2) {
        cpu_set_t* set = xcpuAlloc(bits);
        int rc         = sched_getaffinity(0, CPU_ALLOC_SIZE(bits), set);
        int err        = errno;
        CPU_FREE(set);
        if (rc == 0 || err != EINVAL) break;
    }
    return bits;
}

/* Append a CPU set as a compact range list, e.g. "0-3,8,64-67". */
static void cpusetToStr(const cpu_set_t* set, size_t bits, Str* out)
{
    size_t setSize  = CPU_ALLOC_SIZE(bits);
    const char* sep = "";

    strAppendf(out, "%s", ""); /* empty set: still a valid string */
    for (size_t cpu = 0; cpu < bits; cpu++) {
        if (!CPU_ISSET_S(cpu, setSize, set)) continue;

        size_t last = cpu;
        while (last + 1 < bits && CPU_ISSET_S(last + 1, setSize, set)) {
            last++;
        }
        if (last == cpu) {
            strAppendf(out, "%s%zu", sep, cpu);
        } else {
            strAppendf(out, "%s%zu-%zu", sep, cpu, last);
        }
        sep = ",";
        cpu = last;
    }
}

/*
 * Append one line describing the calling thread to out (without newline).
 * tid < 0 means "the process". pre is printed just before the mask.
 */
static void describeSelf(
    Str* out, const Context* ctx, int tid, int nthreads, const char* pre)
{
    char who[32];
    Str mask = { 0 };
    int ncpus = 0, cpu = -1, numa = -1;
    unsigned int c = 0, node = 0;
    cpu_set_t* set = xcpuAlloc(ctx->cpuBits);
    size_t setSize = CPU_ALLOC_SIZE(ctx->cpuBits);

    if (sched_getaffinity(0, setSize, set) == 0) {
        ncpus = CPU_COUNT_S(setSize, set);
        cpusetToStr(set, ctx->cpuBits, &mask);
    } else {
        strAppendf(&mask, "?");
    }
    CPU_FREE(set);

    /* cpu and numa from one call, so they match even if the thread migrates */
    if (syscall(SYS_getcpu, &c, &node, NULL) == 0) {
        cpu  = (int)c;
        numa = (int)node;
    } else {
        cpu = sched_getcpu();
    }

    if (tid < 0) {
        snprintf(who, sizeof(who), "process");
    } else {
        snprintf(who, sizeof(who), "thread %d/%d", tid, nthreads);
    }

    strAppendf(out,
        "rank %3d (local %2d) %-13s host %-*s  pid %d  tid %d  cpu %3d  numa %2d%s  "
        "mask[%d] {%s}",
        ctx->rank,
        ctx->localRank,
        who,
        ctx->hostWidth,
        ctx->host,
        (int)getpid(),
        myGettid(),
        cpu,
        numa,
        pre,
        ncpus,
        mask.s);
    free(mask.s);
}

/* Run a shell command and append its complete stdout to out. */
static void captureCmd(const char* cmd, Str* out)
{
    char buf[4096];
    size_t got;
    FILE* fp = popen(cmd, "r");

    if (fp == NULL) return;
    while ((got = fread(buf, 1, sizeof(buf), fp)) > 0) {
        strAppendf(out, "%.*s", (int)got, buf);
    }
    pclose(fp);
}

/*
 * Called by thread 0 of every rank while all its threads are still alive.
 * The node barriers make sure all ranks of the node are inside their threaded
 * region, so the tree shows every rank together with its threads.
 */
static void nodePstree(const Context* ctx)
{
    MPI_Barrier(ctx->nodeComm);
    if (ctx->localRank == 0) {
        char cmd[96];
        pid_t parent = getppid();
        size_t hdr;

        strAppendf(ctx->pstree,
            "--- process tree on %s (parent pid %d of rank %d) ---\n",
            ctx->host,
            (int)parent,
            ctx->rank);
        hdr = ctx->pstree->len;
        /* exec: no extra shell in the tree, -l: no cut at 132 columns, -A: ASCII */
        snprintf(cmd, sizeof(cmd), "exec pstree -p -l -A %d 2>/dev/null", (int)parent);
        captureCmd(cmd, ctx->pstree);
        if (ctx->pstree->len == hdr) {
            strAppendf(ctx->pstree, "(pstree not available)\n");
        }
    }
    MPI_Barrier(ctx->nodeComm);
}

/* Maximum number of threads per rank (slots needed), -1 if invalid. */
static int threadSlots(void)
{
#if defined(_OPENMP)
    return omp_get_max_threads();
#elif defined(PTHREADS)
    const char* s = getenv("PTHREAD_THREADS");
    char* end     = NULL;
    long v;

    if (s == NULL) return 4;
    v = strtol(s, &end, 10);
    if (end == s || *end != '\0' || v < 1 || v > MAX_PTHREADS) return -1;
    return (int)v;
#else
    return 0;
#endif
}

#if defined(_OPENMP)

/* The OpenMP place column; val is "-" for the process line. */
static void placeField(char* out, size_t n, const char* val)
{
    snprintf(out, n, "  place %-7s", val);
}

/*
 * Append the union of all OpenMP places. For OMP_PLACES=threads/cores/...
 * the runtime builds the places from the mask it found at startup, so this
 * is the launcher's mask even if the runtime has already bound the main
 * thread (libgomp does that before main() runs).
 */
static void appendPlacesMask(const Context* ctx, Str* out)
{
    int nplaces = omp_get_num_places();
    size_t setSize;
    cpu_set_t* set;
    Str mask = { 0 };

    if (nplaces <= 0) return;
    set     = xcpuAlloc(ctx->cpuBits);
    setSize = CPU_ALLOC_SIZE(ctx->cpuBits);
    for (int p = 0; p < nplaces; p++) {
        int n    = omp_get_place_num_procs(p);
        int* ids = xcalloc((size_t)(n > 0 ? n : 1), sizeof(int));
        omp_get_place_proc_ids(p, ids);
        for (int i = 0; i < n; i++) {
            if (ids[i] >= 0 && (size_t)ids[i] < ctx->cpuBits) {
                CPU_SET_S((size_t)ids[i], setSize, set);
            }
        }
        free(ids);
    }
    cpusetToStr(set, ctx->cpuBits, &mask);
    strAppendf(out, "  places-mask[%d] {%s}", CPU_COUNT_S(setSize, set), mask.s);
    free(mask.s);
    CPU_FREE(set);
}

/* Start the OpenMP team; every thread writes its line into slots[tid]. */
static int runThreads(const Context* ctx, Str* slots, int nslots)
{
    int nthreads = 0;

#pragma omp parallel
    {
        int tid  = omp_get_thread_num();
        int nthr = omp_get_num_threads();
        char val[32], extra[48];
        snprintf(val, sizeof(val), "%d/%d", omp_get_place_num(), omp_get_num_places());
        placeField(extra, sizeof(extra), val);

        /* make sure the whole team is up and running before we look */
#pragma omp barrier
        if (tid < nslots) {
            describeSelf(&slots[tid], ctx, tid, nthr, extra);
        }
        /* thread 0 is the thread that called MPI_Init_thread (FUNNELED) */
        if (tid == 0) {
            nthreads = (nthr < nslots) ? nthr : nslots;
            if (ctx->doPstree) nodePstree(ctx);
        }
    } /* the other threads stay alive at this implicit barrier meanwhile */

    return nthreads;
}

#elif defined(PTHREADS)

typedef struct {
    pthread_t handle;
    int tid;
    int nthreads;
    pthread_barrier_t* barrier;
    Str* line;
    const Context* ctx;
} ThreadInfo;

static void* threadStart(void* arg)
{
    ThreadInfo* ti = arg;

    /* make sure all threads are up and running before we look */
    pthread_barrier_wait(ti->barrier);
    describeSelf(ti->line, ti->ctx, ti->tid, ti->nthreads, "");
    /* thread 0 is the main thread, so it may call MPI (FUNNELED) */
    if (ti->tid == 0 && ti->ctx->doPstree) nodePstree(ti->ctx);
    pthread_barrier_wait(ti->barrier);
    return NULL;
}

/* Start nslots pthreads (the main thread is thread 0). */
static int runThreads(const Context* ctx, Str* slots, int nslots)
{
    ThreadInfo* ti = xcalloc((size_t)nslots, sizeof(*ti));
    pthread_barrier_t barrier;
    int err = pthread_barrier_init(&barrier, NULL, (unsigned)nslots);

    if (err != 0) {
        fprintf(stderr, "rank %d: pthread_barrier_init failed: %s\n", ctx->rank, strerror(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < nslots; i++) {
        ti[i].tid      = i;
        ti[i].nthreads = nslots;
        ti[i].barrier  = &barrier;
        ti[i].line     = &slots[i];
        ti[i].ctx      = ctx;
    }
    for (int i = 1; i < nslots; i++) {
        err = pthread_create(&ti[i].handle, NULL, threadStart, &ti[i]);
        if (err != 0) {
            /* the barrier could never complete: abort instead of hanging */
            fprintf(stderr,
                "rank %d: pthread_create %d failed: %s\n",
                ctx->rank,
                i,
                strerror(err));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    threadStart(&ti[0]);
    for (int i = 1; i < nslots; i++) {
        pthread_join(ti[i].handle, NULL);
    }
    pthread_barrier_destroy(&barrier);
    free(ti);
    return nslots;
}

#else

static int runThreads(const Context* ctx, Str* slots, int nslots)
{
    (void)slots;
    (void)nslots;
    if (ctx->doPstree) nodePstree(ctx);
    return 0;
}

#endif

#ifdef _OPENMP
static const char* envOr(const char* name)
{
    const char* v = getenv(name);
    return v ? v : "(unset)";
}
#endif

/*
 * Gather one string per rank of comm on its rank 0. There it returns all
 * strings back to back and *displsOut[r] is the offset of rank r's string.
 */
static char* gatherStrings(const char* s, MPI_Comm comm, int** displsOut)
{
    int me, n, len = (int)strlen(s) + 1;
    int* counts = NULL;
    int* displs = NULL;
    char* all   = NULL;

    MPI_Comm_rank(comm, &me);
    MPI_Comm_size(comm, &n);
    if (me == 0) {
        counts = xcalloc((size_t)n, sizeof(int));
        displs = xcalloc((size_t)n, sizeof(int));
    }
    MPI_Gather(&len, 1, MPI_INT, counts, 1, MPI_INT, 0, comm);
    if (me == 0) {
        size_t total = 0;
        for (int r = 0; r < n; r++) {
            if (total > INT_MAX) {
                fprintf(stderr, "MPI_pin_test: output exceeds 2 GB\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            displs[r] = (int)total;
            total += (size_t)counts[r];
        }
        all = xcalloc(total, 1);
    }
    MPI_Gatherv(s, len, MPI_CHAR, all, counts, displs, MPI_CHAR, 0, comm);
    free(counts);
    *displsOut = displs;
    return all;
}

int main(int argc, char** argv)
{
    Context ctx;
    Str pstree = { 0 }, block = { 0 };
    int provided, size, len, isLeader, nnodes = 0;
    const char* pre = "";

    MPI_Init_thread(&argc, &argv, THREAD_LEVEL, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(ctx.host, &len);
    MPI_Allreduce(&len, &ctx.hostWidth, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Comm_split_type(MPI_COMM_WORLD,
        MPI_COMM_TYPE_SHARED,
        ctx.rank,
        MPI_INFO_NULL,
        &ctx.nodeComm);
    MPI_Comm_rank(ctx.nodeComm, &ctx.localRank);
    isLeader = (ctx.localRank == 0);
    MPI_Reduce(&isLeader, &nnodes, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    ctx.cpuBits = probeCpuBits();
    ctx.pstree  = &pstree;

    ctx.doPstree = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-ps") == 0) ctx.doPstree = 1;
    }
    MPI_Bcast(&ctx.doPstree, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /*
     * The process line comes first, before any OpenMP call, because some
     * runtimes bind the main thread when they initialize.
     */
#ifdef _OPENMP
    char procPlace[32];
    placeField(procPlace, sizeof(procPlace), "-");
    pre = procPlace;
#endif
    describeSelf(&block, &ctx, -1, 0, pre);
#ifdef _OPENMP
    appendPlacesMask(&ctx, &block);
#endif
    strAppendf(&block, "\n");

    /* any rank with an invalid thread count: the first one reports, all exit */
    int nslots = threadSlots();
    int bad    = (nslots < 0) ? ctx.rank : size, firstBad;
    MPI_Allreduce(&bad, &firstBad, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (firstBad < size) {
        if (ctx.rank == firstBad) {
            fprintf(stderr,
                "MPI_pin_test: invalid PTHREAD_THREADS='%s' on rank %d (allowed: 1 to %d)\n",
                getenv("PTHREAD_THREADS"),
                ctx.rank,
                MAX_PTHREADS);
        }
        free(block.s);
        MPI_Comm_free(&ctx.nodeComm);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* the lines of the threads follow the process line */
    Str* slots   = xcalloc((size_t)nslots, sizeof(Str));
    int nthreads = runThreads(&ctx, slots, nslots);
    for (int t = 0; t < nslots; t++) {
        if (t < nthreads) strAppendf(&block, "%s\n", slots[t].s);
        free(slots[t].s);
    }
    free(slots);

    /* collect all blocks on rank 0 (their lengths differ) */
    int* displs = NULL;
    char* all   = gatherStrings(block.s, MPI_COMM_WORLD, &displs);

#ifdef _OPENMP
    Str env = { 0 };
    strAppendf(&env,
        "OMP_NUM_THREADS=%s OMP_PLACES=%s OMP_PROC_BIND=%s",
        envOr("OMP_NUM_THREADS"),
        envOr("OMP_PLACES"),
        envOr("OMP_PROC_BIND"));
    int* envDispls = NULL;
    char* envAll   = gatherStrings(env.s, MPI_COMM_WORLD, &envDispls);
    free(env.s);
#endif

    /* collect the per-node process trees on rank 0 */
    char* trees     = NULL;
    int* treeDispls = NULL;
    if (ctx.doPstree) {
        MPI_Comm leaders;
        MPI_Comm_split(MPI_COMM_WORLD, isLeader ? 0 : MPI_UNDEFINED, ctx.rank, &leaders);
        if (leaders != MPI_COMM_NULL) {
            trees = gatherStrings(pstree.s ? pstree.s : "", leaders, &treeDispls);
            MPI_Comm_free(&leaders);
        }
    }

    if (ctx.rank == 0) {
        printf("MPI_pin_test: %d rank(s) on %d node(s), thread support %s",
            size,
            nnodes,
            threadLevelName(provided));
        if (provided < THREAD_LEVEL) {
            printf(" (WARNING: requested %s)", threadLevelName(THREAD_LEVEL));
        }
        printf("\n");
#ifdef _OPENMP
        printf("OpenMP: %s (rank 0 env)\n", envAll);
        int ndiff = 0, firstDiff = -1;
        for (int r = 1; r < size; r++) {
            if (strcmp(envAll + envDispls[r], envAll) != 0) {
                if (ndiff++ == 0) firstDiff = r;
            }
        }
        if (ndiff > 0) {
            printf("WARNING: %d rank(s) have a different OpenMP environment, e.g. rank %d: %s\n",
                ndiff,
                firstDiff,
                envAll + envDispls[firstDiff]);
        }
#endif
        for (int r = 0; r < size; r++) {
            if (THREADED) printf("\n");
            fputs(all + displs[r], stdout);
        }
        for (int n = 0; trees != NULL && n < nnodes; n++) {
            printf("\n%s", trees + treeDispls[n]);
        }
        fflush(stdout);
    }

#ifdef _OPENMP
    free(envAll);
    free(envDispls);
#endif
    free(trees);
    free(treeDispls);
    free(all);
    free(displs);
    free(block.s);
    free(pstree.s);
    MPI_Comm_free(&ctx.nodeComm);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
