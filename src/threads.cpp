// One persistent worker thread; both threads run at QoS user-interactive so they land on P-clusters. run_pair_e:
// a second worker at QoS background, which runs on the efficiency cluster (AMX backend, threads = 3).
// The worker spins on an atomic for a short while before it sleeps, so back-to-back calls do not pay a wake-up.
#include "internal.h"
#include <pthread.h>
#include <pthread/qos.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

namespace mt {
namespace {

constexpr auto kSpin = std::chrono::microseconds(200);

struct Worker {
  std::mutex mu;
  std::condition_variable cv;
  void (*fn)(void*) = nullptr;
  void* arg = nullptr;
  std::atomic<long> posted{0}, done{0};
  std::atomic<bool> sleeping{false};
  pthread_t th;

  explicit Worker(qos_class_t qos) {
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setstacksize(&at, 64 << 20);
    pthread_attr_set_qos_class_np(&at, qos, 0);
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
      const auto t0 = std::chrono::steady_clock::now();
      while (posted.load(std::memory_order_acquire) == seen && std::chrono::steady_clock::now() - t0 < kSpin) {
      }
      if (posted.load(std::memory_order_acquire) == seen) {
        std::unique_lock<std::mutex> g(mu);
        sleeping.store(true);
        cv.wait(g, [&] { return posted.load() != seen; });
        sleeping.store(false);
      }
      seen = posted.load(std::memory_order_acquire);
      fn(arg);
      done.store(seen, std::memory_order_release);
    }
  }
};

void run_on(Worker* w, void (*fn)(void*), void* a0, void* a1) {
  static std::mutex call_mu;
  std::lock_guard<std::mutex> cg(call_mu);
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  w->fn = fn;
  w->arg = a1;
  long ticket;
  {
    std::lock_guard<std::mutex> g(w->mu);  // orders the post against a worker that is about to sleep
    ticket = w->posted.fetch_add(1, std::memory_order_acq_rel) + 1;
  }
  if (w->sleeping.load()) w->cv.notify_one();
  fn(a0);
  while (w->done.load(std::memory_order_acquire) != ticket) {
  }
}

}  // namespace

void run_pair(void (*fn)(void*), void* a0, void* a1) {
  static Worker* w = new Worker(QOS_CLASS_USER_INTERACTIVE);  // never joined: lives for the process
  run_on(w, fn, a0, a1);
}

void run_pair_e(void (*fn)(void*), void* a0, void* a1) {
  static Worker* w = new Worker(QOS_CLASS_BACKGROUND);
  run_on(w, fn, a0, a1);
}

}  // namespace mt
