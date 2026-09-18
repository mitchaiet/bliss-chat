/* Row-parallel worker pool for the linear layers.
 * The caller is thread 0 and always does a share of the rows. Helpers spin
 * briefly on a generation counter (cheap while tokens stream), then sleep on
 * an event or condition variable so an idle chat costs no CPU. Every row is
 * computed by exactly the same code as the serial path, so results are
 * bit-identical for any thread count. Windows XP: CreateThread + events. */
#ifndef SLM_THREADS_H
#define SLM_THREADS_H
#include <stdlib.h>
#if defined(__SSE2__)
#include <emmintrin.h>
#define SLM_PAUSE() _mm_pause()
#else
#define SLM_PAUSE() ((void)0)
#endif
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

typedef void (*SlmRowFn)(void *ctx, int r0, int r1);
typedef struct {
#ifdef _WIN32
    HANDLE thread, wake;
#else
    pthread_t thread;
#endif
    int index;
    volatile long sleeping;
} SlmWorker;
static struct {
    int count;                 /* threads including the caller; 1 = serial */
    SlmWorker *workers;        /* count - 1 helpers */
    volatile long generation, done;
    SlmRowFn fn; void *ctx; int rows, align;
#ifndef _WIN32
    pthread_mutex_t mutex; pthread_cond_t cond;
#endif
} slm_pool;
#define SLM_SPIN_ITERATIONS 200000

static void slm_pool_range(int index, int *r0, int *r1) {
    int blocks = (slm_pool.rows + slm_pool.align - 1) / slm_pool.align;
    int per = (blocks + slm_pool.count - 1) / slm_pool.count;
    long a = (long)index * per * slm_pool.align, b = a + (long)per * slm_pool.align;
    *r0 = a > slm_pool.rows ? slm_pool.rows : (int)a;
    *r1 = b > slm_pool.rows ? slm_pool.rows : (int)b;
}
static void slm_atomic_inc(volatile long *v) {
#ifdef _WIN32
    InterlockedIncrement(v);
#else
    __sync_fetch_and_add(v, 1);
#endif
}
static long slm_atomic_load(volatile long *v) {
#ifdef _WIN32
    return InterlockedCompareExchange(v, 0, 0);
#else
    return __sync_fetch_and_add(v, 0);
#endif
}
static void slm_worker_loop(SlmWorker *w) {
    long seen = 0;
    for (;;) {
        int spins = 0;
        while (slm_atomic_load(&slm_pool.generation) == seen) {
            if (++spins < SLM_SPIN_ITERATIONS) { SLM_PAUSE(); continue; }
            w->sleeping = 1;
#ifdef _WIN32
            if (slm_atomic_load(&slm_pool.generation) == seen) WaitForSingleObject(w->wake, INFINITE);
#else
            pthread_mutex_lock(&slm_pool.mutex);
            while (slm_atomic_load(&slm_pool.generation) == seen) pthread_cond_wait(&slm_pool.cond, &slm_pool.mutex);
            pthread_mutex_unlock(&slm_pool.mutex);
#endif
            w->sleeping = 0; spins = 0;
        }
        seen = slm_atomic_load(&slm_pool.generation);
        if (!slm_pool.fn) return;
        int r0, r1; slm_pool_range(w->index, &r0, &r1);
        if (r1 > r0) slm_pool.fn(slm_pool.ctx, r0, r1);
        slm_atomic_inc(&slm_pool.done);
    }
}
#ifdef _WIN32
static DWORD WINAPI slm_worker_entry(LPVOID p) { slm_worker_loop((SlmWorker *)p); return 0; }
#else
static void *slm_worker_entry(void *p) { slm_worker_loop((SlmWorker *)p); return NULL; }
#endif
static int slm_default_threads(void) {
#ifdef _WIN32
    SYSTEM_INFO info; GetSystemInfo(&info); return (int)info.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN); return n > 0 ? (int)n : 1;
#endif
}
/* Start count-1 helpers; count <= 0 means one thread per processor. */
static int slm_threads_init(int count) {
    if (slm_pool.count) return slm_pool.count;
    if (count <= 0) count = slm_default_threads();
    if (count > 64) count = 64;
    if (count < 1) count = 1;
    slm_pool.count = 1;
    if (count == 1) return 1;
    slm_pool.workers = (SlmWorker *)calloc((size_t)count - 1, sizeof(SlmWorker));
    if (!slm_pool.workers) return 1;
#ifndef _WIN32
    pthread_mutex_init(&slm_pool.mutex, NULL); pthread_cond_init(&slm_pool.cond, NULL);
#endif
    for (int i = 0; i < count - 1; i++) {
        SlmWorker *w = slm_pool.workers + i; w->index = i + 1;
#ifdef _WIN32
        w->wake = CreateEventA(NULL, FALSE, FALSE, NULL);
        w->thread = w->wake ? CreateThread(NULL, 0, slm_worker_entry, w, 0, NULL) : NULL;
        if (!w->thread) break;
#else
        if (pthread_create(&w->thread, NULL, slm_worker_entry, w)) break;
#endif
        slm_pool.count = i + 2;
    }
    return slm_pool.count;
}
/* Run fn over [0,rows) split into contiguous ranges aligned to `align` rows. */
static void slm_parallel_rows(int rows, int align, SlmRowFn fn, void *ctx) {
    if (!slm_pool.count) slm_threads_init(0);
    if (slm_pool.count == 1 || rows < align * 2) { fn(ctx, 0, rows); return; }
    slm_pool.fn = fn; slm_pool.ctx = ctx; slm_pool.rows = rows; slm_pool.align = align;
    slm_pool.done = 0;
    slm_atomic_inc(&slm_pool.generation);
#ifdef _WIN32
    for (int i = 0; i < slm_pool.count - 1; i++)
        if (slm_pool.workers[i].sleeping) SetEvent(slm_pool.workers[i].wake);
#else
    pthread_mutex_lock(&slm_pool.mutex); pthread_cond_broadcast(&slm_pool.cond); pthread_mutex_unlock(&slm_pool.mutex);
#endif
    int r0, r1; slm_pool_range(0, &r0, &r1);
    if (r1 > r0) fn(ctx, r0, r1);
    while (slm_atomic_load(&slm_pool.done) < slm_pool.count - 1) SLM_PAUSE();
}
#endif
