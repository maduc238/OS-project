#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static pthread_cond_t threadDied = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t threadMutex = PTHREAD_MUTEX_INITIALIZER;
static int totThreads = 0;  /* Tổng số thread sẽ được tạo */
static int numLive = 0;     /* Tổng số thread còn sống hoặc đã bị hủy nhưng chưa được join */
static int numUnjoined = 0; /* Số thread đã bị hủy mà chưa được join */
enum tstate {   /* Trạng thái của thread */
    TS_ALIVE,       /* Thread còn sống */
    TS_TERMINATED,  /* Thread bị hủy, nhưng chưa join */
    TS_JOINED       /* Thread bị hủy, và đã join */
};

static struct { /* Thông tin của mỗi thread */
    pthread_t tid;      /* Thread ID */
    enum tstate state;  /* Thread state (TS_*) */
    int sleepTime;      /* Thời gian sống trước khi bị hủy (tính theo giây) */
} *thread;

static void *threadFunc(void *arg) {
    int idx = (int) arg;
    int s;
    sleep(thread[idx].sleepTime);
    printf("Thread %d terminating\n", idx);
    s = pthread_mutex_lock(&threadMutex);
    numUnjoined++;
    thread[idx].state = TS_TERMINATED;
    s = pthread_mutex_unlock(&threadMutex);
    s = pthread_cond_signal(&threadDied);
    return NULL;
}

int main(int argc, char *argv[]) {
    int s, idx;
    char *p;
    thread = calloc(argc - 1, sizeof(*thread));
    if (thread == NULL)
        return 0;
    /* Tạo tất cả các thread */
    for (idx = 0; idx < argc - 1; idx++) {
        thread[idx].sleepTime = strtol(argv[idx+1], &p, 10);
        thread[idx].state = TS_ALIVE;
        s = pthread_create(&thread[idx].tid, NULL, threadFunc, idx);
    }
    totThreads = argc - 1;
    numLive = totThreads;
    /* Join một thread đã bị hủy */
    while (numLive > 0) {
        s = pthread_mutex_lock(&threadMutex);
        while (numUnjoined == 0)
            s = pthread_cond_wait(&threadDied, &threadMutex);
        for (idx = 0; idx < totThreads; idx++) {
            if (thread[idx].state == TS_TERMINATED){
                s = pthread_join(thread[idx].tid, NULL);
                thread[idx].state = TS_JOINED;
                numLive--;
                numUnjoined--;
                printf("Reaped thread %d (numLive=%d)\n", idx, numLive);
            }
        }
        s = pthread_mutex_unlock(&threadMutex);
    }
    return 0;
}
