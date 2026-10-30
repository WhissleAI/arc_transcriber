/* Errors, memory, files, timing, thread pool, JSON.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#define _GNU_SOURCE
#include "arc_base.h"

#include <math.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>   /* _mm_pause */
#elif defined(_MSC_VER) && defined(_M_ARM64)
#include <intrin.h>      /* __yield */
#endif
#else
#include <pthread.h>
#include <sys/resource.h>
#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif
#endif

/* ------------------------------------------------------------------ errors */
static _Thread_local char g_err[1024];

void arc_seterr(const char* fmt, ...) {
  char tmp[sizeof g_err];   /* callers may pass arc_geterr() as an argument: format into a copy first */
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(tmp, sizeof tmp, fmt, ap);
  va_end(ap);
  memcpy(g_err, tmp, sizeof g_err);
}

double arc_strtod(const char* s, char** end) {
  const char* p = s;
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
  const char* a = p;
  if (*p == '+' || *p == '-') ++p;
  if (((p[0] | 32) == 'i' && (p[1] | 32) == 'n') || ((p[0] | 32) == 'n' && (p[1] | 32) == 'a'))
    return strtod(s, end);   /* inf / nan: no decimal point involved */
  int digits = 0;
  while (*p >= '0' && *p <= '9') ++p, ++digits;
  if (*p == '.') {
    ++p;
    while (*p >= '0' && *p <= '9') ++p, ++digits;
  }
  if (!digits) {
    if (end) *end = (char*)s;
    return 0.0;
  }
  if (*p == 'e' || *p == 'E') {
    const char* q = p + 1;
    if (*q == '+' || *q == '-') ++q;
    if (*q >= '0' && *q <= '9') {
      while (*q >= '0' && *q <= '9') ++q;
      p = q;
    }
  }
  size_t n = (size_t)(p - a);
  const char* dp = localeconv()->decimal_point;
  size_t dl = dp ? strlen(dp) : 1;
  char buf[128];
  char* t = n * (dl ? dl : 1) + 1 <= sizeof buf ? buf : (char*)malloc(n * (dl ? dl : 1) + 1);
  if (!t) {
    if (end) *end = (char*)s;
    return 0.0;
  }
  char* o = t;
  for (size_t i = 0; i < n; ++i) {
    if (a[i] == '.' && dp && dl && !(dp[0] == '.' && dl == 1)) {
      memcpy(o, dp, dl);
      o += dl;
    } else
      *o++ = a[i];
  }
  *o = 0;
  double v = strtod(t, NULL);
  if (t != buf) free(t);
  if (end) *end = (char*)p;
  return v;
}

const char* arc_geterr(void) { return g_err; }

/* ------------------------------------------------------------------ memory */
void* arc_malloc(size_t n) {
  void* p = NULL;
  if (n == 0) n = 1;
#ifdef _WIN32
  p = _aligned_malloc(n, 64);
#else
  if (posix_memalign(&p, 64, (n + 63) & ~(size_t)63) != 0) p = NULL;
#endif
  if (!p) arc_seterr("out of memory (%zu bytes)", n);
  return p;
}

void* arc_calloc(size_t n) {
  void* p = arc_malloc(n);
  if (p) memset(p, 0, n);
  return p;
}

void arc_free(void* p) {
#ifdef _WIN32
  _aligned_free(p);
#else
  free(p);
#endif
}

char* arc_strdup(const char* s) {
  size_t n = strlen(s) + 1;
  char* d = (char*)arc_malloc(n);
  if (d) memcpy(d, s, n);
  return d;
}

int arc_read_file(const char* path, char** data, size_t* len) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    arc_seterr("cannot open %s", path);
    return -1;
  }
#ifdef _WIN32
  __int64 n = -1;   /* long is 32-bit on Windows: use the 64-bit calls so files > 2 GB work */
  if (_fseeki64(f, 0, SEEK_END) == 0) n = _ftelli64(f);
  if (n >= 0 && _fseeki64(f, 0, SEEK_SET) != 0) n = -1;
#else
  long n = -1;
  if (fseek(f, 0, SEEK_END) == 0) n = ftell(f);
  if (n >= 0 && fseek(f, 0, SEEK_SET) != 0) n = -1;
#endif
  if (n < 0 || (uint64_t)n >= (uint64_t)SIZE_MAX) {
    fclose(f);
    arc_seterr("cannot read %s", path);
    return -1;
  }
  char* d = (char*)arc_malloc((size_t)n + 1);
  if (!d) {
    fclose(f);
    return -1;
  }
  if (fread(d, 1, (size_t)n, f) != (size_t)n) {
    fclose(f);
    arc_free(d);
    arc_seterr("cannot read %s", path);
    return -1;
  }
  fclose(f);
  d[n] = 0;
  *data = d;
  if (len) *len = (size_t)n;
  return 0;
}

double arc_time_ms(void) {
#ifdef _WIN32
  LARGE_INTEGER f, c;
  QueryPerformanceFrequency(&f);
  QueryPerformanceCounter(&c);
  return 1000.0 * (double)c.QuadPart / (double)f.QuadPart;
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
#endif
}

size_t arc_peak_rss_kb(void) {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS pmc;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) return pmc.PeakWorkingSetSize / 1024;
  return 0;
#else
  struct rusage ru;
  if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
#ifdef __APPLE__
  return (size_t)ru.ru_maxrss / 1024;
#else
  return (size_t)ru.ru_maxrss;
#endif
#endif
}

/* ------------------------------------------------------------------ thread pool
 * Workers sleep on a condition variable between jobs (after a short, time-bounded spin, so back-to-back
 * parallel_for calls in a layer stack do not pay a futex wake-up each). Tasks are claimed from an atomic counter;
 * the caller works too. A parallel_for issued from inside a task, or while another thread is using the pool, runs
 * serially on the calling thread instead of deadlocking. */
#ifdef _WIN32
typedef HANDLE arc_thread_t;
#else
typedef pthread_t arc_thread_t;
#endif

enum { ARC_POOL_STACK = 1 << 20 };   /* worker stack: the GEMM tile keeps a 64 KB panel on the stack (musl: 128 KB) */
#define ARC_WORKER_SPIN_MS 0.1       /* a worker spins this long for the next job before sleeping */
#define ARC_CALLER_SPIN_MS 0.5       /* the caller spins this long for the workers to finish before sleeping */

struct arc_pool {
  int n;                   /* threads including the caller (1 + workers that actually started) */
  arc_thread_t* th;
#ifdef _WIN32
  CRITICAL_SECTION mu;
  CONDITION_VARIABLE cv, done_cv;
#else
  pthread_mutex_t mu;
  pthread_cond_t cv, done_cv;
#endif
  int n_tasks;             /* written by the caller before gen is bumped, read by workers after they see it */
  arc_task_fn fn;
  void* ctx;
  atomic_int quit;
  atomic_int* tids;        /* Linux: each worker's kernel thread id (0 until it has started), for arc_pool_tids */
  atomic_int tid_slot;
  _Alignas(64) atomic_int gen;       /* job generation (each hot counter on its own cache line) */
  _Alignas(64) atomic_int next;      /* next task to claim */
  _Alignas(64) atomic_int pending;   /* workers still inside the current job */
  _Alignas(64) atomic_int busy;      /* a parallel_for is running */
};

static _Thread_local const arc_pool* t_in_pool;   /* the pool whose task this thread is running, if any */

static inline void cpu_relax(void) {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
  __builtin_ia32_pause();
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  _mm_pause();
#elif defined(__GNUC__) && defined(__aarch64__)
  __asm__ __volatile__("yield");
#elif defined(_MSC_VER) && defined(_M_ARM64)
  __yield();
#endif
}

/* spin until *a != v (want_ne) or *a <= 0 (!want_ne), for at most ms; 1 if the condition became true */
static int spin_wait(atomic_int* a, int v, int want_ne, double ms) {
  double t0 = 0;
  for (unsigned i = 1;; ++i) {
    int x = atomic_load_explicit(a, memory_order_acquire);
    if (want_ne ? x != v : x <= 0) return 1;
    cpu_relax();
    if ((i & 63) == 0) {   /* the clock is read only every 64 relax instructions */
      double t = arc_time_ms();
      if (i == 64) t0 = t;
      else if (t - t0 > ms) return 0;
    }
  }
}

static void pool_work(arc_pool* p) {
  for (;;) {
    int t = atomic_fetch_add(&p->next, 1);
    if (t >= p->n_tasks) break;
    p->fn(p->ctx, t, p->n_tasks);
  }
}

#ifdef _WIN32
static DWORD WINAPI pool_main(LPVOID arg)
#else
static void* pool_main(void* arg)
#endif
{
  arc_pool* p = (arc_pool*)arg;
  int seen = 0;
  t_in_pool = p;
#if defined(__linux__)
  if (p->tids) atomic_store(&p->tids[atomic_fetch_add(&p->tid_slot, 1)], (int)syscall(SYS_gettid));
#endif
  for (;;) {
    if (!spin_wait(&p->gen, seen, 1, ARC_WORKER_SPIN_MS)) {
#ifdef _WIN32
      EnterCriticalSection(&p->mu);
      while (atomic_load(&p->gen) == seen && !atomic_load(&p->quit)) SleepConditionVariableCS(&p->cv, &p->mu, INFINITE);
      LeaveCriticalSection(&p->mu);
#else
      pthread_mutex_lock(&p->mu);
      while (atomic_load(&p->gen) == seen && !atomic_load(&p->quit)) pthread_cond_wait(&p->cv, &p->mu);
      pthread_mutex_unlock(&p->mu);
#endif
    }
    if (atomic_load(&p->quit)) break;
    seen = atomic_load(&p->gen);
    pool_work(p);
    if (atomic_fetch_sub(&p->pending, 1) == 1) {
#ifdef _WIN32
      EnterCriticalSection(&p->mu);
      WakeAllConditionVariable(&p->done_cv);
      LeaveCriticalSection(&p->mu);
#else
      pthread_mutex_lock(&p->mu);
      pthread_cond_broadcast(&p->done_cv);
      pthread_mutex_unlock(&p->mu);
#endif
    }
  }
#ifdef _WIN32
  return 0;
#else
  return NULL;
#endif
}

static void pool_sync_destroy(arc_pool* p) {
#ifdef _WIN32
  DeleteCriticalSection(&p->mu);
#else
  pthread_mutex_destroy(&p->mu);
  pthread_cond_destroy(&p->cv);
  pthread_cond_destroy(&p->done_cv);
#endif
}

/* stop and join the first n_started workers */
static void pool_stop(arc_pool* p, int n_started) {
#ifdef _WIN32
  EnterCriticalSection(&p->mu);
  atomic_store(&p->quit, 1);
  atomic_fetch_add(&p->gen, 1);
  WakeAllConditionVariable(&p->cv);
  LeaveCriticalSection(&p->mu);
  for (int i = 0; i < n_started; ++i) {
    WaitForSingleObject(p->th[i], INFINITE);
    CloseHandle(p->th[i]);
  }
#else
  pthread_mutex_lock(&p->mu);
  atomic_store(&p->quit, 1);
  atomic_fetch_add(&p->gen, 1);
  pthread_cond_broadcast(&p->cv);
  pthread_mutex_unlock(&p->mu);
  for (int i = 0; i < n_started; ++i) pthread_join(p->th[i], NULL);
#endif
}

arc_pool* arc_pool_create(int n_threads) {
  arc_pool* p = (arc_pool*)arc_calloc(sizeof *p);
  if (!p) return NULL;
  p->n = n_threads < 1 ? 1 : n_threads;
  if (p->n == 1) return p;
  p->th = (arc_thread_t*)arc_calloc(sizeof(arc_thread_t) * (size_t)(p->n - 1));
  p->tids = (atomic_int*)arc_calloc(sizeof(atomic_int) * (size_t)(p->n - 1));   /* optional: NULL = no ids */
  if (!p->th) {   /* no worker array: a serial pool still works */
    arc_free(p->tids);
    p->tids = NULL;
    p->n = 1;
    return p;
  }
#ifdef _WIN32
  InitializeCriticalSection(&p->mu);
  InitializeConditionVariable(&p->cv);
  InitializeConditionVariable(&p->done_cv);
#else
  pthread_mutex_init(&p->mu, NULL);
  pthread_cond_init(&p->cv, NULL);
  pthread_cond_init(&p->done_cv, NULL);
  pthread_attr_t attr;
  int have_attr = pthread_attr_init(&attr) == 0;
  if (have_attr) pthread_attr_setstacksize(&attr, ARC_POOL_STACK);   /* keep the default if refused */
#endif
  int started = 0;
  for (int i = 0; i < p->n - 1; ++i) {
#ifdef _WIN32
    p->th[i] = CreateThread(NULL, ARC_POOL_STACK, pool_main, p, STACK_SIZE_PARAM_IS_A_RESERVATION, NULL);
    if (!p->th[i]) break;
#else
    if (pthread_create(&p->th[i], have_attr ? &attr : NULL, pool_main, p) != 0) break;
#endif
    ++started;
  }
#ifndef _WIN32
  if (have_attr) pthread_attr_destroy(&attr);
#endif
  if (started == 0) {   /* no worker could be started: run serially */
    pool_sync_destroy(p);
    arc_free(p->th);
    arc_free(p->tids);
    p->th = NULL;
    p->tids = NULL;
    p->n = 1;
    return p;
  }
  /* fewer workers than asked for: the pool is smaller. Workers only read n via pending, which parallel_for sets. */
  p->n = started + 1;
  return p;
}

void arc_pool_destroy(arc_pool* p) {
  if (!p) return;
  if (p->n > 1) {
    pool_stop(p, p->n - 1);
    pool_sync_destroy(p);
  }
  arc_free(p->th);
  arc_free(p->tids);
  arc_free(p);
}

int arc_pool_size(const arc_pool* p) { return p ? p->n : 1; }

int arc_pool_tids(const arc_pool* p, int* out, int max) {
#if defined(__linux__)
  if (!p || !p->tids || p->n < 2) return 0;
  int n = 0;
  for (int i = 0; i < p->n - 1 && n < max; ++i) {
    int t = 0;
    for (int w = 0; w < 2000 && !(t = atomic_load(&p->tids[i])); ++w) {   /* a worker still starting: up to 100 ms */
      struct timespec ts = {0, 50000};
      nanosleep(&ts, NULL);
    }
    if (t) out[n++] = t;
  }
  return n;
#else
  (void)p, (void)out, (void)max;
  return 0;
#endif
}

static void run_serial(int n_tasks, arc_task_fn fn, void* ctx) {
  for (int t = 0; t < n_tasks; ++t) fn(ctx, t, n_tasks);
}

void arc_parallel_for(arc_pool* p, int n_tasks, arc_task_fn fn, void* ctx) {
  if (n_tasks <= 0) return;
  if (!p || p->n == 1 || n_tasks == 1 || t_in_pool == p) {   /* t_in_pool == p: called from one of p's tasks */
    run_serial(n_tasks, fn, ctx);
    return;
  }
  int expected = 0;
  if (!atomic_compare_exchange_strong(&p->busy, &expected, 1)) {   /* another thread owns the pool right now */
    run_serial(n_tasks, fn, ctx);
    return;
  }
  p->fn = fn;
  p->ctx = ctx;
  p->n_tasks = n_tasks;
  atomic_store(&p->next, 0);
  atomic_store(&p->pending, p->n - 1);
#ifdef _WIN32
  EnterCriticalSection(&p->mu);
  atomic_fetch_add(&p->gen, 1);
  WakeAllConditionVariable(&p->cv);
  LeaveCriticalSection(&p->mu);
#else
  pthread_mutex_lock(&p->mu);
  atomic_fetch_add(&p->gen, 1);
  pthread_cond_broadcast(&p->cv);
  pthread_mutex_unlock(&p->mu);
#endif
  const arc_pool* outer = t_in_pool;
  t_in_pool = p;
  pool_work(p);
  t_in_pool = outer;
  if (!spin_wait(&p->pending, 0, 0, ARC_CALLER_SPIN_MS)) {
#ifdef _WIN32
    EnterCriticalSection(&p->mu);
    while (atomic_load(&p->pending) > 0) SleepConditionVariableCS(&p->done_cv, &p->mu, INFINITE);
    LeaveCriticalSection(&p->mu);
#else
    pthread_mutex_lock(&p->mu);
    while (atomic_load(&p->pending) > 0) pthread_cond_wait(&p->done_cv, &p->mu);
    pthread_mutex_unlock(&p->mu);
#endif
  }
  atomic_store(&p->busy, 0);
}

/* ------------------------------------------------------------------ JSON */
typedef struct {
  const char* s;
  const char* e;
  int depth;
} jp;

static void jws(jp* p) {
  while (p->s < p->e && (*p->s == ' ' || *p->s == '\t' || *p->s == '\n' || *p->s == '\r')) p->s++;
}

static void put_utf8(char** o, uint32_t c) {
  char* d = *o;
  if (c < 0x80) *d++ = (char)c;
  else if (c < 0x800) {
    *d++ = (char)(0xC0 | (c >> 6));
    *d++ = (char)(0x80 | (c & 0x3F));
  } else if (c < 0x10000) {
    *d++ = (char)(0xE0 | (c >> 12));
    *d++ = (char)(0x80 | ((c >> 6) & 0x3F));
    *d++ = (char)(0x80 | (c & 0x3F));
  } else {
    *d++ = (char)(0xF0 | (c >> 18));
    *d++ = (char)(0x80 | ((c >> 12) & 0x3F));
    *d++ = (char)(0x80 | ((c >> 6) & 0x3F));
    *d++ = (char)(0x80 | (c & 0x3F));
  }
  *o = d;
}

static int hex4(const char* s, uint32_t* v) {
  *v = 0;
  for (int i = 0; i < 4; ++i) {
    char c = s[i];
    *v <<= 4;
    if (c >= '0' && c <= '9') *v |= (uint32_t)(c - '0');
    else if (c >= 'a' && c <= 'f') *v |= (uint32_t)(c - 'a' + 10);
    else if (c >= 'A' && c <= 'F') *v |= (uint32_t)(c - 'A' + 10);
    else return -1;
  }
  return 0;
}

/* a string at the opening quote. As Python's json: only the JSON escapes, no raw control characters. The input
 * is not NUL-terminated (an mmapped header), so every read is bounded by p->e. */
static char* jstring(jp* p) {
  const char* s = ++p->s;
  size_t n = 0;
  while (s < p->e && *s != '"') {
    if (*s == '\\') ++s;
    ++s;
    ++n;
  }
  if (s >= p->e) {
    arc_seterr("JSON: unterminated string");
    return NULL;
  }
  const char* end = s;   /* the closing quote */
  char* out = (char*)arc_malloc(n * 4 + 1);
  if (!out) return NULL;
  char* o = out;
  while (p->s < end) {
    char c = *p->s++;
    if ((unsigned char)c < 0x20) {
      arc_free(out);
      arc_seterr("JSON: control character in a string");
      return NULL;
    }
    if (c != '\\') {
      *o++ = c;
      continue;
    }
    c = *p->s++;   /* p->s < end here: the first pass skipped the escaped character */
    switch (c) {
      case '"': *o++ = '"'; break;
      case '\\': *o++ = '\\'; break;
      case '/': *o++ = '/'; break;
      case 'n': *o++ = '\n'; break;
      case 't': *o++ = '\t'; break;
      case 'r': *o++ = '\r'; break;
      case 'b': *o++ = '\b'; break;
      case 'f': *o++ = '\f'; break;
      case 'u': {
        uint32_t u;
        if (end - p->s < 4 || hex4(p->s, &u)) {
          arc_free(out);
          arc_seterr("JSON: bad \\u escape");
          return NULL;
        }
        p->s += 4;
        if (u >= 0xD800 && u < 0xDC00 && end - p->s >= 6 && p->s[0] == '\\' && p->s[1] == 'u') {
          uint32_t lo;
          if (!hex4(p->s + 2, &lo) && lo >= 0xDC00 && lo < 0xE000) {
            u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
            p->s += 6;
          }
        }
        if (u >= 0xD800 && u < 0xE000) u = 0xFFFD;   /* a lone surrogate: keep the output valid UTF-8 */
        if (u == 0) {
          arc_free(out);
          arc_seterr("JSON: \\u0000 is not supported in strings");
          return NULL;
        }
        put_utf8(&o, u);
        break;
      }
      default:
        arc_free(out);
        arc_seterr("JSON: invalid escape '\\%c'", c);
        return NULL;
    }
  }
  p->s = end + 1;
  *o = 0;
  return out;
}

/* the literal lit (n bytes) at p->s, without reading past the end */
static int jlit(const jp* p, const char* lit, size_t n) { return (size_t)(p->e - p->s) >= n && !memcmp(p->s, lit, n); }

/* a JSON number -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)? at p->s, bounded by p->e; Python's NaN / Infinity /
 * -Infinity literals too (json.dumps writes them). */
static int jnumber(jp* p, arc_json* v) {
  v->type = ARC_JNUM;
  if (jlit(p, "NaN", 3)) return v->num = NAN, p->s += 3, 0;
  if (jlit(p, "Infinity", 8)) return v->num = INFINITY, p->s += 8, 0;
  if (jlit(p, "-Infinity", 9)) return v->num = -INFINITY, p->s += 9, 0;
  const char *a = p->s, *q = a, *e = p->e;
  int frac = 0;
  if (q < e && *q == '-') ++q;
  if (q < e && *q == '0') ++q;
  else if (q < e && *q >= '1' && *q <= '9')
    while (q < e && *q >= '0' && *q <= '9') ++q;
  else goto bad;
  if (q < e && *q == '.') {
    ++q, frac = 1;
    if (!(q < e && *q >= '0' && *q <= '9')) goto bad;
    while (q < e && *q >= '0' && *q <= '9') ++q;
  }
  if (q < e && (*q == 'e' || *q == 'E')) {
    ++q, frac = 1;
    if (q < e && (*q == '+' || *q == '-')) ++q;
    if (!(q < e && *q >= '0' && *q <= '9')) goto bad;
    while (q < e && *q >= '0' && *q <= '9') ++q;
  }
  {
    size_t n = (size_t)(q - a);
    char small[64];
    char* buf = n < sizeof small ? small : (char*)malloc(n + 1);
    if (!buf) {
      arc_seterr("out of memory");
      return -1;
    }
    memcpy(buf, a, n);
    buf[n] = 0;
    v->num = arc_strtod(buf, NULL);
    if (buf != small) free(buf);
  }
  v->is_int = !frac;
  p->s = q;
  return 0;
bad:
  arc_seterr("JSON: bad number");
  return -1;
}

static int jvalue(jp* p, arc_json* v);
static void jfree_inner(arc_json* j);

static int jgrow(arc_json* v, int* cap, int obj) {
  if (v->n < *cap) return 0;
  if (*cap > (1 << 28)) {
    arc_seterr("JSON: too many items");
    return -1;
  }
  int nc = *cap ? *cap * 2 : 8;
  arc_json* it = (arc_json*)realloc(v->items, sizeof(arc_json) * (size_t)nc);
  if (!it) {
    arc_seterr("out of memory");
    return -1;
  }
  v->items = it;
  if (obj) {
    char** k = (char**)realloc(v->keys, sizeof(char*) * (size_t)nc);
    if (!k) {
      arc_seterr("out of memory");
      return -1;
    }
    v->keys = k;
  }
  *cap = nc;
  return 0;
}

static int jvalue(jp* p, arc_json* v) {
  memset(v, 0, sizeof *v);
  jws(p);
  if (p->s >= p->e) {
    arc_seterr("JSON: unexpected end");
    return -1;
  }
  if (++p->depth > 200) {
    arc_seterr("JSON: nested too deeply");
    return -1;
  }
  char c = *p->s;
  int rc = 0;
  if (c == '{' || c == '[') {
    int obj = c == '{', cap = 0;
    v->type = obj ? ARC_JOBJ : ARC_JARR;
    p->s++;
    jws(p);
    if (p->s < p->e && *p->s == (obj ? '}' : ']')) {
      p->s++;
    } else {
      for (;;) {
        if (jgrow(v, &cap, obj)) goto fail;
        if (obj) {
          jws(p);
          if (p->s >= p->e || *p->s != '"') {
            arc_seterr("JSON: expected a key");
            goto fail;
          }
          v->keys[v->n] = jstring(p);
          if (!v->keys[v->n]) goto fail;
          jws(p);
          if (p->s >= p->e || *p->s != ':') {
            arc_free(v->keys[v->n]);
            arc_seterr("JSON: expected ':'");
            goto fail;
          }
          p->s++;
        }
        if (jvalue(p, &v->items[v->n])) {
          if (obj) arc_free(v->keys[v->n]);
          goto fail;
        }
        v->n++;
        jws(p);
        if (p->s < p->e && *p->s == ',') {
          p->s++;
          continue;
        }
        if (p->s < p->e && *p->s == (obj ? '}' : ']')) {
          p->s++;
          break;
        }
        arc_seterr("JSON: expected ',' or a closing bracket");
        goto fail;
      }
    }
  } else if (c == '"') {
    v->type = ARC_JSTR;
    v->str = jstring(p);
    if (!v->str) return -1;
  } else if (jlit(p, "true", 4)) {
    v->type = ARC_JBOOL, v->num = 1, p->s += 4;
  } else if (jlit(p, "false", 5)) {
    v->type = ARC_JBOOL, v->num = 0, p->s += 5;
  } else if (jlit(p, "null", 4)) {
    v->type = ARC_JNULL, p->s += 4;
  } else if (c == '-' || (c >= '0' && c <= '9') || c == 'N' || c == 'I') {
    if (jnumber(p, v)) return -1;
  } else {
    arc_seterr("JSON: unexpected character '%c'", c);
    rc = -1;
  }
  p->depth--;
  return rc;
fail:   /* a container that failed half way frees what it already holds (the failing child freed itself) */
  jfree_inner(v);
  memset(v, 0, sizeof *v);
  return -1;
}

static void jfree_inner(arc_json* j) {
  if (j->type == ARC_JSTR) arc_free(j->str);
  if (j->type == ARC_JARR || j->type == ARC_JOBJ) {
    for (int i = 0; i < j->n; ++i) {
      jfree_inner(&j->items[i]);
      if (j->type == ARC_JOBJ) arc_free(j->keys[i]);
    }
    free(j->items);
    free(j->keys);
  }
}

arc_json* arc_json_parse(const char* text, size_t len) {
  jp p = {text, text + len, 0};
  arc_json* j = (arc_json*)calloc(1, sizeof *j);
  if (!j) {
    arc_seterr("out of memory");
    return NULL;
  }
  if (jvalue(&p, j)) {
    jfree_inner(j);   /* frees what was built before the error */
    free(j);
    return NULL;
  }
  jws(&p);
  if (p.s != p.e) {
    arc_json_free(j);
    arc_seterr("JSON: trailing characters");
    return NULL;
  }
  return j;
}

void arc_json_free(arc_json* j) {
  if (!j) return;
  jfree_inner(j);
  free(j);
}

const arc_json* arc_json_get(const arc_json* o, const char* key) {
  if (!o || o->type != ARC_JOBJ) return NULL;
  for (int i = o->n - 1; i >= 0; --i)   /* a repeated key: the last one wins, as in Python's json */
    if (!strcmp(o->keys[i], key)) return &o->items[i];
  return NULL;
}

const arc_json* arc_json_at(const arc_json* a, int i) {
  if (!a || (a->type != ARC_JARR && a->type != ARC_JOBJ) || i < 0 || i >= a->n) return NULL;
  return &a->items[i];
}

double arc_json_num(const arc_json* j, double d) { return j && (j->type == ARC_JNUM || j->type == ARC_JBOOL) ? j->num : d; }
const char* arc_json_str(const arc_json* j, const char* d) { return j && j->type == ARC_JSTR ? j->str : d; }
