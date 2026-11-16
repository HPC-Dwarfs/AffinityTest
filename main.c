/*
 * MPI_pin_test.c - show where MPI ranks and their threads run (Linux only).
 *
 * For every rank (and every thread) it prints: host, pid/tid, the CPU it is
 * currently running on, the NUMA node of that CPU and the complete affinity
 * mask. With OpenMP the OpenMP place is printed as well. All output is
 * collected on rank 0 and printed in rank/thread order.
 *
 * Build:
 *   mpicc -O2 -fopenmp           -o pin_omp MPI_pin_test.c   # MPI + OpenMP
 *   mpicc -O2 -DPTHREADS -pthread -o pin_pt  MPI_pin_test.c   # MPI + pthreads
 *   mpicc -O2                    -o pin_mpi MPI_pin_test.c   # pure MPI
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
#error "MPI_pin_test.c requires Linux (sched_getaffinity, sched_getcpu, getcpu)"
#endif
#if defined(_OPENMP) && defined(PTHREADS)
#error "Build either with OpenMP or with -DPTHREADS, not both"
#endif

#define _GNU_SOURCE
#include <sched.h>
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

#define LINE_LEN   512   /* one output line per rank/thread */
#define MASK_LEN   320   /* textual affinity mask, e.g. "0-15,64-79" */
#define PSTREE_LEN 16384 /* captured pstree output per node */

#if defined(_OPENMP) || defined(PTHREADS)
#define THREAD_LEVEL MPI_THREAD_FUNNELED
#else
#define THREAD_LEVEL MPI_THREAD_SINGLE
#endif

typedef struct {
    char host[MPI_MAX_PROCESSOR_NAME];
    int rank;      /* rank in MPI_COMM_WORLD */
    int localRank; /* rank within the node */
    int doPstree;  /* -ps given */
    MPI_Comm nodeComm;
    char* pstreeOut; /* PSTREE_LEN bytes, filled on node-local rank 0 */
} Context;

static void* xcalloc(size_t n, size_t size)
{
    void* p = calloc(n, size);
    if (p == NULL) {
        fprintf(stderr, "MPI_pin_test: out of memory\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return p;
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

/* Format a CPU set as a compact range list, e.g. "0-3,8,64-67". */
static void cpusetToStr(const cpu_set_t* set, char* out, size_t n)
{
    size_t pos = 0;
    out[0]     = '\0';

    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        if (!CPU_ISSET(cpu, set)) continue;

        int last = cpu;
        while (last + 1 < CPU_SETSIZE && CPU_ISSET(last + 1, set)) {
            last++;
        }

        const char* sep = (pos > 0) ? "," : "";
        int w = (last == cpu) ? snprintf(out + pos, n - pos, "%s%d", sep, cpu)
                              : snprintf(out + pos, n - pos, "%s%d-%d", sep, cpu, last);
        if (w < 0 || (size_t)w >= n - pos) { /* out of space: mark truncation */
            if (n >= 4) strcpy(out + n - 4, "...");
            return;
        }
        pos += (size_t)w;
        cpu = last;
    }
}

/* Describe the calling thread in one line. tid < 0 means "the process". */
static void describeSelf(
    char* out, size_t n, const Context* ctx, int tid, int nthreads, const char* extra)
{
    char who[32];
    char mask[MASK_LEN] = "?";
    int ncpus           = 0;
    unsigned int cpu = 0, node = 0;
    int numa = -1;
    cpu_set_t set;

    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        ncpus = CPU_COUNT(&set);
        cpusetToStr(&set, mask, sizeof(mask));
    }
    if (syscall(SYS_getcpu, &cpu, &node, NULL) == 0) numa = (int)node;

    if (tid < 0) {
        snprintf(who, sizeof(who), "process");
    } else {
        snprintf(who, sizeof(who), "thread %d/%d", tid, nthreads);
    }

    snprintf(out,
        n,
        "rank %3d (local %2d) %-13s host %s  pid %d  tid %d  cpu %3d  numa %2d%s  "
        "mask[%d] {%s}",
        ctx->rank,
        ctx->localRank,
        who,
        ctx->host,
        (int)getpid(),
        myGettid(),
        sched_getcpu(),
        numa,
        extra,
        ncpus,
        mask);
}

/* Run a shell command and capture (at most n-1 bytes of) its stdout. */
static void captureCmd(const char* cmd, char* out, size_t n)
{
    size_t len = 0;
    FILE* fp   = popen(cmd, "r");

    if (fp != NULL) {
        size_t got;
        while (len < n - 1 && (got = fread(out + len, 1, n - 1 - len, fp)) > 0) {
            len += got;
        }
        pclose(fp);
    }
    out[len] = '\0';
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
        char cmd[64];
        pid_t parent = getppid();
        int hdr      = snprintf(ctx->pstreeOut,
            PSTREE_LEN,
            "--- process tree on %s (parent pid %d of rank %d) ---\n",
            ctx->host,
            (int)parent,
            ctx->rank);
        snprintf(cmd, sizeof(cmd), "pstree -p %d 2>/dev/null", (int)parent);
        captureCmd(cmd, ctx->pstreeOut + hdr, PSTREE_LEN - (size_t)hdr);
        if (ctx->pstreeOut[hdr] == '\0') {
            snprintf(ctx->pstreeOut + hdr,
                PSTREE_LEN - (size_t)hdr,
                "(pstree not available)\n");
        }
    }
    MPI_Barrier(ctx->nodeComm);
}

/* Maximum number of threads per rank, i.e. number of line slots needed. */
static int threadSlots(const Context* ctx)
{
#if defined(_OPENMP)
    (void)ctx;
    return omp_get_max_threads();
#elif defined(PTHREADS)
    const char* s = getenv("PTHREAD_THREADS");
    char* end     = NULL;
    long v;

    if (s == NULL) return 4;
    v = strtol(s, &end, 10);
    if (end == s || *end != '\0' || v < 1 || v > 4096) {
        fprintf(stderr, "rank %d: invalid PTHREAD_THREADS='%s'\n", ctx->rank, s);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return (int)v;
#else
    (void)ctx;
    return 0;
#endif
}

#if defined(_OPENMP)

/* Start the OpenMP team; every thread writes its line into slots[tid]. */
static int runThreads(const Context* ctx, char* slots, int nslots)
{
    int nthreads = 0;

#pragma omp parallel
    {
        int tid  = omp_get_thread_num();
        int nthr = omp_get_num_threads();
        char extra[48];
        snprintf(extra,
            sizeof(extra),
            "  place %d/%d",
            omp_get_place_num(),
            omp_get_num_places());

        /* make sure the whole team is up and running before we look */
#pragma omp barrier
        if (tid < nslots) {
            describeSelf(slots + (size_t)tid * LINE_LEN, LINE_LEN, ctx, tid, nthr, extra);
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
    char* line;
    const Context* ctx;
} ThreadInfo;

static void* threadStart(void* arg)
{
    ThreadInfo* ti = arg;

    /* make sure all threads are up and running before we look */
    pthread_barrier_wait(ti->barrier);
    describeSelf(ti->line, LINE_LEN, ti->ctx, ti->tid, ti->nthreads, "");
    /* thread 0 is the main thread, so it may call MPI (FUNNELED) */
    if (ti->tid == 0 && ti->ctx->doPstree) nodePstree(ti->ctx);
    pthread_barrier_wait(ti->barrier);
    return NULL;
}

/* Start nslots pthreads (the main thread is thread 0). */
static int runThreads(const Context* ctx, char* slots, int nslots)
{
    ThreadInfo* ti = xcalloc((size_t)nslots, sizeof(*ti));
    pthread_barrier_t barrier;

    pthread_barrier_init(&barrier, NULL, (unsigned)nslots);
    for (int i = 0; i < nslots; i++) {
        ti[i].tid      = i;
        ti[i].nthreads = nslots;
        ti[i].barrier  = &barrier;
        ti[i].line     = slots + (size_t)i * LINE_LEN;
        ti[i].ctx      = ctx;
    }
    for (int i = 1; i < nslots; i++) {
        int err = pthread_create(&ti[i].handle, NULL, threadStart, &ti[i]);
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

static int runThreads(const Context* ctx, char* slots, int nslots)
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

int main(int argc, char** argv)
{
    Context ctx;
    int provided, size, len, isLeader, nnodes = 0;

    MPI_Init_thread(&argc, &argv, THREAD_LEVEL, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(ctx.host, &len);
    MPI_Comm_split_type(MPI_COMM_WORLD,
        MPI_COMM_TYPE_SHARED,
        ctx.rank,
        MPI_INFO_NULL,
        &ctx.nodeComm);
    MPI_Comm_rank(ctx.nodeComm, &ctx.localRank);
    isLeader = (ctx.localRank == 0);
    MPI_Reduce(&isLeader, &nnodes, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    ctx.doPstree = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-ps") == 0) ctx.doPstree = 1;
    }
    MPI_Bcast(&ctx.doPstree, 1, MPI_INT, 0, MPI_COMM_WORLD);
    ctx.pstreeOut = xcalloc(PSTREE_LEN, 1);

    /* line 0: the process itself, lines 1..: its threads */
    int maxLines = 1 + threadSlots(&ctx);
    char* lines  = xcalloc((size_t)maxLines, LINE_LEN);
    describeSelf(lines, LINE_LEN, &ctx, -1, 0, "");
    int nlines = 1 + runThreads(&ctx, lines + LINE_LEN, maxLines - 1);

    /* collect all lines on rank 0 (thread counts may differ between ranks) */
    MPI_Datatype lineType;
    MPI_Type_contiguous(LINE_LEN, MPI_CHAR, &lineType);
    MPI_Type_commit(&lineType);

    int* counts = NULL;
    int* displs = NULL;
    char* all   = NULL;
    if (ctx.rank == 0) {
        counts = xcalloc((size_t)size, sizeof(int));
        displs = xcalloc((size_t)size, sizeof(int));
    }
    MPI_Gather(&nlines, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (ctx.rank == 0) {
        int total = 0;
        for (int r = 0; r < size; r++) {
            displs[r] = total;
            total += counts[r];
        }
        all = xcalloc((size_t)total, LINE_LEN);
    }
    MPI_Gatherv(lines,
        nlines,
        lineType,
        all,
        counts,
        displs,
        lineType,
        0,
        MPI_COMM_WORLD);

    /* collect the per-node process trees on rank 0 */
    char* trees = NULL;
    if (ctx.doPstree) {
        MPI_Comm leaders;
        MPI_Comm_split(MPI_COMM_WORLD, isLeader ? 0 : MPI_UNDEFINED, ctx.rank, &leaders);
        if (leaders != MPI_COMM_NULL) {
            if (ctx.rank == 0) trees = xcalloc((size_t)nnodes, PSTREE_LEN);
            MPI_Gather(ctx.pstreeOut,
                PSTREE_LEN,
                MPI_CHAR,
                trees,
                PSTREE_LEN,
                MPI_CHAR,
                0,
                leaders);
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
        printf("OpenMP: OMP_NUM_THREADS=%s OMP_PLACES=%s OMP_PROC_BIND=%s (rank 0 env)\n",
            envOr("OMP_NUM_THREADS"),
            envOr("OMP_PLACES"),
            envOr("OMP_PROC_BIND"));
#endif
        for (int r = 0; r < size; r++) {
            if (counts[r] > 1) printf("\n");
            for (int l = 0; l < counts[r]; l++) {
                printf("%s\n", all + (size_t)(displs[r] + l) * LINE_LEN);
            }
        }
        for (int n = 0; trees != NULL && n < nnodes; n++) {
            printf("\n%s", trees + (size_t)n * PSTREE_LEN);
        }
        fflush(stdout);
    }

    free(trees);
    free(all);
    free(displs);
    free(counts);
    free(lines);
    free(ctx.pstreeOut);
    MPI_Type_free(&lineType);
    MPI_Comm_free(&ctx.nodeComm);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
