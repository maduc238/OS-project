#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
static int glob = 0;

/* Free bộ nhớ được trỏ tới `arg` và mở khóa mutex */
static void cleanupHandler(void *arg) {
    int s;
    printf("cleanup: freeing block tại %p\n", arg);
    free(arg);
    printf("cleanup: mở khóa mutex\n");
    s = pthread_mutex_unlock(&mtx);
}
static void *threadFunc(void *arg) {
    int s;
    /* Buffer được allocate bởi thread */
    void *buf = NULL;
    /* Phân bổ một khối bộ nhớ có vị trí lưu tại buf */
    buf = malloc(0x10000);
    printf("thread: allocated memory tại %p\n", buf);
    /* Sau đó khó mutex. Đây không phải là một cancellation point */
    s = pthread_mutex_lock(&mtx);
    /* Vì thread có thể bị hủy với việc chưa cleanup, nên sử dụng hàm này
    để thiết đặt cleanup handler với địa chỉ lưu trong buf và mở khóa mutex
    nếu như thread bị hủy */
    pthread_cleanup_push(cleanupHandler, buf);
    /* Vòng lặp này chờ tín hiệu hủy thread */
    while (glob == 0) {
        /* Cancellation point */
        s = pthread_cond_wait(&cond, &mtx);
    }
    printf("thread: thoát vòng lặp do condition variable\n");
    pthread_cleanup_pop(1); /* Executes cleanup handler */
    return NULL;
}

int main(int argc, char *argv[]) {
    pthread_t thr;
    void *res;
    int s;
    s = pthread_create(&thr, NULL, threadFunc, NULL);
    sleep(2); /* Give thread a chance to get started */
    if (argc == 1) {
        /* Nếu không nhập arg nào trên terminal */
        printf("main: cancel thread\n");
        /* Cancel thread. Sau đó pthread_cleanup_push() tự động
        được gọi */
        s = pthread_cancel(thr);
    } else {
        /* Nếu nhập arg trên terminal */
        printf("main: tạo tín hiệu condition variable\n");
        /* Đặt tín hiệu thủ cộng: đổi biến glob thành 1 để thoát vòng lặp
        và thực thi pthread_cleanup_pop(). Điều này cũng gọi cleanup handler */
        glob = 1;
        s = pthread_cond_signal(&cond);
    }
    s = pthread_join(thr, &res);
    if (res == PTHREAD_CANCELED)
        printf("main: thread được hủy\n");
    else
        printf("main: thread được kết thúc một cách thông thường\n");
    return 0;
}
