#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/**
 * Thread check trạng thái
*/
static void *threadFunc(void *arg) {
    int loc, j;
    for (;;) {
        printf("ok\n");
        sleep(1);
    }
    return NULL;
}

/**
 * Thread tăng ram
*/
static void *incRam(void *arg) {
    int *a;
    for (;;) {
        a = (int *) malloc(10000000);
        usleep(100);
    }
    return NULL;
}

int main(int argc, char *argv[]) {
    pthread_t t1, t2;
    int s;
    s = pthread_create(&t1, NULL, threadFunc, NULL);
    s = pthread_create(&t2, NULL, incRam, NULL);

    s = pthread_join(t1, NULL);
    s = pthread_join(t2, NULL);

    // printf("glob = %d\n", glob);
    return(0);
}
