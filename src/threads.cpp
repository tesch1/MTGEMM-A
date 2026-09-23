// One persistent worker thread; both threads run at QoS user-interactive so they land on P-clusters.
#include "internal.h"
#include <pthread.h>
#include <pthread/qos.h>
#include <condition_variable>
#include <mutex>

namespace mt {
namespace {

struct Worker {
  std::mutex mu;
  std::condition_variable cv;
  void (*fn)(void*) = nullptr;
  void* arg = nullptr;
  long posted = 0, done = 0;
  pthread_t th;

  Worker() {
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setstacksize(&at, 64 << 20);
    pthread_attr_set_qos_class_np(&at, QOS_CLASS_USER_INTERACTIVE, 0);
    pthread_create(&th, &at, &Worker::entry, this);
    pthread_attr_destroy(&at);
  }
  static void* entry(void* p) {
    static_cast<Worker*>(p)->loop();
    return nullptr;
  }
  void loop() {
    long seen = 0;
    for (;;) {
      void (*f)(void*);
      void* a;
      {
        std::unique_lock<std::mutex> g(mu);
        cv.wait(g, [&] { return posted != seen; });
        seen = posted;
        f = fn;
        a = arg;
      }
      f(a);
      {
        std::lock_guard<std::mutex> g(mu);
        done = seen;
      }
      cv.notify_all();
    }
  }
};

}  // namespace

void run_pair(void (*fn)(void*), void* a0, void* a1) {
  static Worker* w = new Worker;  // never joined: lives for the process
  static std::mutex call_mu;
  std::lock_guard<std::mutex> cg(call_mu);
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  long ticket;
  {
    std::lock_guard<std::mutex> g(w->mu);
    w->fn = fn;
    w->arg = a1;
    ticket = ++w->posted;
  }
  w->cv.notify_all();
  fn(a0);
  std::unique_lock<std::mutex> g(w->mu);
  w->cv.wait(g, [&] { return w->done == ticket; });
}

}  // namespace mt
