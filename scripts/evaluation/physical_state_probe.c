#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    char *path;
    off_t size;
} file_entry;

static const size_t MAX_FILES = 4096;
static const uintmax_t MAX_TOTAL_BYTES = 512ULL * 1024ULL * 1024ULL;

static void fail(const char *message, const char *path) {
    if (path != NULL) {
        fprintf(stderr, "%s: %s: %s\n", message, path, strerror(errno));
    } else {
        fprintf(stderr, "%s\n", message);
    }
    exit(1);
}

static double monotonic_ms(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        fail("clock_gettime failed", NULL);
    }
    return value.tv_sec * 1000.0 + value.tv_nsec / 1000000.0;
}

static file_entry *load_template(const char *path, size_t *count_out) {
    FILE *input = fopen(path, "r");
    if (input == NULL) {
        fail("open template failed", path);
    }
    file_entry *entries = NULL;
    size_t count = 0;
    size_t capacity = 0;
    char *line = NULL;
    size_t line_capacity = 0;
    ssize_t length;
    uintmax_t total_bytes = 0;
    while ((length = getline(&line, &line_capacity, input)) >= 0) {
        if (length == 0 || line[length - 1] != '\n') {
            fail("template line lacks newline", path);
        }
        line[length - 1] = '\0';
        char *separator = strchr(line, '\t');
        if (separator == NULL || strchr(separator + 1, '\t') != NULL) {
            fail("invalid template line", path);
        }
        *separator = '\0';
        errno = 0;
        char *end = NULL;
        uintmax_t size = strtoumax(line, &end, 10);
        if (errno != 0 || end == line || *end != '\0' || size == 0 ||
            size > (uintmax_t)INT64_MAX || separator[1] != '/') {
            fail("invalid template record", path);
        }
        if (count >= MAX_FILES || total_bytes + size > MAX_TOTAL_BYTES) {
            fail("template exceeds frozen bounds", path);
        }
        if (count == capacity) {
            capacity = capacity == 0 ? 64 : capacity * 2;
            file_entry *grown = realloc(entries, capacity * sizeof(*entries));
            if (grown == NULL) {
                fail("template allocation failed", NULL);
            }
            entries = grown;
        }
        entries[count].path = strdup(separator + 1);
        if (entries[count].path == NULL) {
            fail("path allocation failed", NULL);
        }
        entries[count].size = (off_t)size;
        count++;
        total_bytes += size;
    }
    if (ferror(input)) {
        fail("read template failed", path);
    }
    free(line);
    if (fclose(input) != 0) {
        fail("close template failed", path);
    }
    *count_out = count;
    return entries;
}

static int checked_open(const file_entry *entry) {
    int fd = open(entry->path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        fail("open file failed", entry->path);
    }
    struct stat info;
    if (fstat(fd, &info) != 0) {
        fail("stat file failed", entry->path);
    }
    if (!S_ISREG(info.st_mode) || info.st_size != entry->size) {
        errno = EINVAL;
        fail("file changed after discovery", entry->path);
    }
    return fd;
}

static void intervene(const file_entry *entries, size_t count, int warm) {
    char *buffer = warm ? malloc(1024 * 1024) : NULL;
    if (warm && buffer == NULL) {
        fail("read buffer allocation failed", NULL);
    }
    for (size_t index = 0; index < count; index++) {
        int fd = checked_open(&entries[index]);
        if (!warm) {
            int error = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            if (error != 0) {
                errno = error;
                fail("POSIX_FADV_DONTNEED failed", entries[index].path);
            }
        } else {
            while (1) {
                ssize_t received = read(fd, buffer, 1024 * 1024);
                if (received == 0) {
                    break;
                }
                if (received < 0) {
                    fail("sequential read failed", entries[index].path);
                }
            }
        }
        if (close(fd) != 0) {
            fail("close file failed", entries[index].path);
        }
    }
    free(buffer);
}

static void residency(const file_entry *entries, size_t count,
                      uint64_t *pages_out, uint64_t *resident_out) {
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        fail("invalid page size", NULL);
    }
    uint64_t pages_total = 0;
    uint64_t resident_total = 0;
    for (size_t index = 0; index < count; index++) {
        int fd = checked_open(&entries[index]);
        size_t length = (size_t)entries[index].size;
        size_t pages = (length + (size_t)page_size - 1) / (size_t)page_size;
        void *mapping = mmap(NULL, length, PROT_NONE, MAP_SHARED, fd, 0);
        if (mapping == MAP_FAILED) {
            fail("mmap failed", entries[index].path);
        }
        if (close(fd) != 0) {
            fail("close file failed", entries[index].path);
        }
        unsigned char *vector = calloc(pages, 1);
        if (vector == NULL) {
            fail("mincore vector allocation failed", NULL);
        }
        if (mincore(mapping, length, vector) != 0) {
            fail("mincore failed", entries[index].path);
        }
        for (size_t page = 0; page < pages; page++) {
            resident_total += vector[page] & 1U;
        }
        pages_total += pages;
        free(vector);
        if (munmap(mapping, length) != 0) {
            fail("munmap failed", entries[index].path);
        }
    }
    *pages_out = pages_total;
    *resident_out = resident_total;
}

int main(int argc, char **argv) {
    if (argc != 3 || (strcmp(argv[1], "cold") != 0 && strcmp(argv[1], "warm") != 0)) {
        fprintf(stderr, "usage: physical_state_probe cold|warm TEMPLATE_TSV\n");
        return 2;
    }
    int warm = strcmp(argv[1], "warm") == 0;
    size_t count = 0;
    file_entry *entries = load_template(argv[2], &count);
    double intervention_start = monotonic_ms();
    intervene(entries, count, warm);
    double intervention_ms = monotonic_ms() - intervention_start;
    uint64_t pages = 0;
    uint64_t resident = 0;
    double probe_start = monotonic_ms();
    residency(entries, count, &pages, &resident);
    double probe_ms = monotonic_ms() - probe_start;
    uint64_t total_bytes = 0;
    for (size_t index = 0; index < count; index++) {
        total_bytes += (uint64_t)entries[index].size;
        free(entries[index].path);
    }
    free(entries);
    double fraction = pages == 0 ? 0.0 : (double)resident / (double)pages;
    printf("{\"condition\":\"%s\",\"file_count\":%zu,"
           "\"total_bytes\":%" PRIu64 ",\"total_pages\":%" PRIu64 ","
           "\"resident_pages\":%" PRIu64 ",\"resident_fraction\":%.9f,"
           "\"intervention_ms\":%.6f,\"probe_ms\":%.6f}\n",
           warm ? "warm" : "cold", count, total_bytes, pages, resident, fraction,
           intervention_ms, probe_ms);
    return 0;
}
