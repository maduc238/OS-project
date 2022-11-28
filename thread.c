#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

static int glob = 0;
static void *threadFunc(void *arg) {
    int loops = *((int *) arg);
    int loc, j;
    for (j = 0; j < loops; j++) {
        loc = glob;
        loc++;
        glob = loc;
    }
    return NULL;
}

int main(int argc, char *argv[]) {
    pthread_t t1, t2;
    int loops, s;
    char *p;
    loops = (argc > 1) ? strtol(argv[1], &p, 10) : 10000000;
    s = pthread_create(&t1, NULL, threadFunc, &loops);
    s = pthread_create(&t2, NULL, threadFunc, &loops);

    s = pthread_join(t1, NULL);
    s = pthread_join(t2, NULL);

    printf("glob = %d\n", glob);
    return(0);
}
