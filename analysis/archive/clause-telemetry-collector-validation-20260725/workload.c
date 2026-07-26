/* Stage-2 spike workloads: real parallel CPU + distinct-mm RSS ground truth.
 *
 *   cpu-threads N SEC : one process, N pthreads each burning SEC of
 *                       CLOCK_THREAD_CPUTIME_ID -> ~N cores for ~SEC wall.
 *   cpu-forks   N SEC : one process forks N children each burning SEC of
 *                       CLOCK_PROCESS_CPUTIME_ID (distinct mm, no exec).
 *   rss         MB SEC: parent (mm A) + 2 threads sharing mm A, and one
 *                       forked child (mm B) — both address spaces alloc MB
 *                       and stay active (touch-looping) for SEC so aligned
 *                       samples see two distinct live mm simultaneously.
 */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static double clock_seconds(clockid_t clock) {
    struct timespec ts;
    clock_gettime(clock, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void burn(clockid_t clock, double seconds) {
    double start = clock_seconds(clock);
    volatile unsigned long x = 0;
    while (clock_seconds(clock) - start < seconds) {
        for (int i = 0; i < 200000; i++) x += i;
    }
}

static char *alloc_touch(size_t mb) {
    size_t n = mb * 1024UL * 1024UL;
    char *p = malloc(n);
    if (!p) { perror("malloc"); _exit(2); }
    for (size_t i = 0; i < n; i += 4096) p[i] = 1;
    return p;
}

typedef struct { double seconds; } burn_arg_t;

static void *thread_burn(void *arg) {
    burn(CLOCK_THREAD_CPUTIME_ID, ((burn_arg_t *)arg)->seconds);
    return NULL;
}

/* Keep an address space resident AND on-CPU (perf CPU-clock only samples
 * on-CPU tasks) by walking the buffer for the wall duration. */
static void *thread_touch(void *arg) {
    double seconds = ((burn_arg_t *)arg)->seconds;
    size_t mb = 150;
    char *p = alloc_touch(mb);
    double start = clock_seconds(CLOCK_MONOTONIC);
    volatile char sink = 0;
    while (clock_seconds(CLOCK_MONOTONIC) - start < seconds) {
        for (size_t i = 0; i < mb * 1024UL * 1024UL; i += 4096) sink ^= p[i];
    }
    (void)sink;
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s MODE N SEC\n", argv[0]); return 1; }
    const char *mode = argv[1];

    if (strcmp(mode, "cpu-threads") == 0) {
        int n = atoi(argv[2]);
        burn_arg_t a = {atof(argv[3])};
        pthread_t t[64];
        for (int i = 0; i < n; i++) pthread_create(&t[i], NULL, thread_burn, &a);
        for (int i = 0; i < n; i++) pthread_join(t[i], NULL);
    } else if (strcmp(mode, "cpu-forks") == 0) {
        int n = atoi(argv[2]);
        double sec = atof(argv[3]);
        for (int i = 0; i < n; i++) {
            pid_t pid = fork();
            if (pid == 0) { burn(CLOCK_PROCESS_CPUTIME_ID, sec); _exit(0); }
        }
        for (int i = 0; i < n; i++) wait(NULL);
    } else if (strcmp(mode, "rss") == 0) {
        size_t mb = atoi(argv[2]);
        double sec = atof(argv[3]);
        pid_t child = fork();
        if (child == 0) {  /* distinct mm B */
            burn_arg_t a = {sec};
            thread_touch(&a);
            _exit(0);
        }
        char *p = alloc_touch(mb);  /* mm A */
        burn_arg_t a = {sec};
        pthread_t t[2];  /* two threads sharing mm A */
        for (int i = 0; i < 2; i++) pthread_create(&t[i], NULL, thread_touch, &a);
        /* main also stays active on mm A */
        double start = clock_seconds(CLOCK_MONOTONIC);
        volatile char sink = 0;
        while (clock_seconds(CLOCK_MONOTONIC) - start < sec) {
            for (size_t i = 0; i < mb * 1024UL * 1024UL; i += 4096) sink ^= p[i];
        }
        (void)sink;
        for (int i = 0; i < 2; i++) pthread_join(t[i], NULL);
        wait(NULL);
    } else {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 1;
    }
    printf("WORKLOAD_DONE\n");
    fflush(stdout);
    return 0;
}
