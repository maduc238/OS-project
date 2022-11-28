#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

static void *threadFunc(void *arg) {
    int j;
    printf("New thread started\n");    /* Có thể là một cancellation point */
    for(j=1;;j++) {
        printf("Loop %d\n", j);     /* Có thể là một cancellation point */
        sleep(1);   /* Một cancellation point */
    }
    return NULL;
}

int main(int argc, char *argv[]) {
    pthread_t thr;
    int s;
    void *res;
    s = pthread_create(&thr, NULL, threadFunc, NULL);
    sleep(3);   /* Cho thread này chạy một lúc */
    s = pthread_cancel(thr);
    s = pthread_join(thr, &res);
    if (res == PTHREAD_CANCELED)
        printf("Thread was canceled\n");
    else
        printf("Thread was not canceled\n");    /* Có thể không xảy ra */
    return 0;
}
