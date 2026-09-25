// Device batch runner: each line of Documents/queue.txt to out_<batch>/NN_<prog>.txt; a crashed line is skipped on relaunch.
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

int test_main(int, char**);
int bench_main(int, char**);
int ubench_main(int, char**);
extern "C" int amx_probe_main(FILE*);

static std::mutex g_mu;
static std::string g_status = "starting";

static void set_status(const std::string& s) {
  std::lock_guard<std::mutex> g(g_mu);
  g_status = s;
}

static int probe_main(int, char**) { return amx_probe_main(stdout); }

static int run_line(const std::string& line) {
  std::vector<std::string> words;
  std::istringstream is(line);
  for (std::string w; is >> w;) words.push_back(w);
  if (words.empty()) return 0;
  std::vector<char*> argv;
  for (auto& w : words) argv.push_back(w.data());
  argv.push_back(nullptr);
  const int argc = int(words.size());
  const std::string& p = words[0];
  if (p == "test") return test_main(argc, argv.data());
  if (p == "bench") return bench_main(argc, argv.data());
  if (p == "ubench") return ubench_main(argc, argv.data());
  if (p == "probe") return probe_main(argc, argv.data());
  std::printf("unknown program %s\n", p.c_str());
  return 127;
}

static double wall() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }

extern "C" const char* mt_vp_status(void) {
  static thread_local std::string copy;
  std::lock_guard<std::mutex> g(g_mu);
  copy = g_status;
  return copy.c_str();
}

extern "C" int mt_vp_batch(const char* docs) {
  // The queue's "#batch <id>" line names the output directory out_<id>, so a new queue starts fresh.
  const std::string d = docs, qpath = d + "/queue.txt";
  std::string id = "default";
  std::vector<std::string> q;
  {
    std::ifstream f(qpath);
    for (std::string l; std::getline(f, l);) {
      if (l.rfind("#batch ", 0) == 0) id = l.substr(7);
      else if (!l.empty() && l[0] != '#') q.push_back(l);
    }
  }
  const std::string out = d + "/out_" + id, spath = out + "/state.txt";
  mkdir(out.c_str(), 0755);
  std::set<int> finished;
  int started = -1;
  {
    std::ifstream f(spath);
    for (std::string w; f >> w;) {
      int i;
      if (w == "start" && f >> i) started = i;
      else if ((w == "done" || w == "crashed") && f >> i) finished.insert(i);
    }
  }
  FILE* st = std::fopen(spath.c_str(), "a");
  if (started >= 0 && !finished.count(started)) {
    std::fprintf(st, "crashed %d\n", started);
    finished.insert(started);
  }
  std::fflush(st);
  int fails = 0;
  for (int i = 0; i < int(q.size()); ++i) {
    if (finished.count(i)) continue;
    std::string prog = q[i].substr(0, q[i].find(' '));
    char name[64];
    std::snprintf(name, sizeof name, "/%02d_%s.txt", i, prog.c_str());
    set_status(std::to_string(i + 1) + "/" + std::to_string(q.size()) + ": " + q[i]);
    std::printf("[%d/%zu] %s\n", i + 1, q.size(), q[i].c_str());
    std::fflush(stdout);
    std::fprintf(st, "start %d\n", i);
    std::fflush(st);
    fsync(fileno(st));
    const double t0 = wall();
    std::fflush(stdout);
    const int saved = dup(1);
    const int fd = open((out + name).c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    dup2(fd, 1);
    close(fd);
    std::printf("# %s\n", q[i].c_str());
    const int rc = run_line(q[i]);
    std::fflush(stdout);
    dup2(saved, 1);
    close(saved);
    fails += rc != 0;
    std::fprintf(st, "done %d rc=%d sec=%.1f\n", i, rc, wall() - t0);
    std::fflush(st);
    std::printf("    rc=%d %.1f s\n", rc, wall() - t0);
    std::fflush(stdout);
  }
  std::fprintf(st, "batch-complete fails=%d\n", fails);
  std::fclose(st);
  set_status("done");
  return fails;
}
