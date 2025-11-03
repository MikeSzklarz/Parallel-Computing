#ifndef TIMER_H
#define TIMER_H

#include <time.h>

// Returns the current wall-clock time (seconds) as a double.
// Uses CLOCK_MONOTONIC for stable timing.
static inline double get_time_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// Macro for consistency with legacy code.
#define GET_TIME(var) \
    do { (var) = get_time_seconds(); } while (0)

#endif // TIMER_H