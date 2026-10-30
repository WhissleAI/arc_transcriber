/* libarc shared internals (errors, memory, thread pool, safetensors, JSON, weights, kernels, FFT). C11, no
 * dependencies beyond libc, libm and pthreads.
 *
 * Conventions used by every model file:
 *   - activations are row-major [T, C] (time-major, batch 1) unless a function says otherwise;
 *   - a function that can fail returns int (0 = ok, -1 = error) and records the message with arc_seterr();
 *     the public API copies it into the caller's buffer (arc_geterr() is thread-local);
 *   - weights are read from safetensors files that stay mmapped for the engine's lifetime; quantized weights
 *     (int8 per output row, int4 per group of g input values) stay quantized and are dequantized panel by panel
 *     inside the matmul.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#ifndef ARC_BASE_H
#define ARC_BASE_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ errors */
void arc_seterr(const char* fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;
const char* arc_geterr(void);
/* strtod for C-syntax decimal numbers that ignores LC_NUMERIC (an embedding app may set a comma-decimal locale).
 * Hexadecimal floats are not accepted. */
double arc_strtod(const char* s, char** end);
#define ARC_CHECK(x) do { if ((x) != 0) return -1; } while (0)

/* ------------------------------------------------------------------ memory */
void* arc_malloc(size_t n);           /* 64-byte aligned, NULL + arc_seterr on failure */
void* arc_calloc(size_t n);           /* zeroed */
void arc_free(void* p);
char* arc_strdup(const char* s);
int arc_read_file(const char* path, char** data, size_t* len);   /* whole file, NUL-terminated, arc_free() it */
double arc_time_ms(void);              /* monotonic clock */
size_t arc_peak_rss_kb(void);          /* 0 where unsupported */

/* ------------------------------------------------------------------ thread pool */
typedef struct arc_pool arc_pool;
arc_pool* arc_pool_create(int n_threads);   /* n <= 1: everything runs on the calling thread */
void arc_pool_destroy(arc_pool* p);
int arc_pool_size(const arc_pool* p);       /* 1 for NULL */
/* Linux: the workers' kernel thread ids (not the caller's); the count. 0 elsewhere. */
int arc_pool_tids(const arc_pool* p, int* out, int max);
/* Runs fn(ctx, task, n_tasks) for task in [0, n_tasks) on the pool (the caller works too) and waits. A call made
 * from inside one of the pool's tasks, or while another thread is using the pool, runs serially on the caller. */
typedef void (*arc_task_fn)(void* ctx, int task, int n_tasks);
void arc_parallel_for(arc_pool* p, int n_tasks, arc_task_fn fn, void* ctx);

/* ------------------------------------------------------------------ JSON (read-only DOM) */
typedef enum { ARC_JNULL, ARC_JBOOL, ARC_JNUM, ARC_JSTR, ARC_JARR, ARC_JOBJ } arc_jtype;
typedef struct arc_json {
  arc_jtype type;
  double num;              /* JNUM, JBOOL (0/1) */
  char* str;               /* JSTR: UTF-8, NUL-terminated */
  int n;                   /* JARR / JOBJ: number of items */
  struct arc_json* items;   /* JARR / JOBJ values */
  char** keys;             /* JOBJ keys (parallel to items) */
  int is_int;              /* JNUM written without fraction or exponent ("3", not "3.0") */
} arc_json;
arc_json* arc_json_parse(const char* text, size_t len);   /* NULL + arc_seterr on a syntax error */
void arc_json_free(arc_json* j);
const arc_json* arc_json_get(const arc_json* obj, const char* key);   /* NULL if absent / not an object */
const arc_json* arc_json_at(const arc_json* arr, int i);
double arc_json_num(const arc_json* j, double dflt);
const char* arc_json_str(const arc_json* j, const char* dflt);

/* ------------------------------------------------------------------ safetensors */
typedef enum { ARC_F32, ARC_F16, ARC_BF16, ARC_I8, ARC_U8, ARC_I16, ARC_I32, ARC_I64, ARC_BOOL, ARC_F64 } arc_dtype;
typedef struct {
  char* name;
  arc_dtype dtype;
  int ndim;
  int64_t shape[8];
  const void* data;        /* into the mapping; may be unaligned */
  size_t nbytes;
} arc_tensor;
typedef struct arc_file arc_file;
arc_file* arc_file_open(const char* path);       /* mmap (read() fallback); NULL + arc_seterr */
void arc_file_close(arc_file* f);
const arc_tensor* arc_file_find(const arc_file* f, const char* name);   /* NULL if absent */
int arc_file_count(const arc_file* f);
const arc_tensor* arc_file_tensor(const arc_file* f, int i);
const char* arc_file_metadata(const arc_file* f, const char* key);   /* __metadata__ value, NULL if absent */
int64_t arc_numel(const arc_tensor* t);
float arc_half_to_float(uint16_t h);
float arc_bf16_to_float(uint16_t h);

/* ------------------------------------------------------------------ weights
 * arc_weights wraps one safetensors file in either layout used by the releases:
 *   plain   : name -> F32 / F16 / BF16 tensor (v1 fp32 files, converted externals);
 *   packed  : metadata "steertts_quant" (the shared loader's own packed layout; Arc releases use "int4_layout", read by
 *             arc_store.c) = {"layout": {name: {"s": raw|fp16|int8|int4, "shape", "g", "Kp"}}}, data in
 *             name (raw/fp16) or name::q + name::s (int8: int8 [O,K] + fp16 scale per row; int4: two values per byte,
 *             low nibble first, value = (nibble - 8) * scale, groups of g along the flattened input dim padded to
 *             Kp, fp16 scale [O, Kp/g]). Used by the v1 int8/ files and the edge package. */
typedef enum { ARC_W_F32, ARC_W_I8, ARC_W_I4 } arc_wtype;
typedef struct {
  arc_wtype type;
  int O, K;                /* output rows, flattened input size (a conv weight [O,C,k] has K = C*k) */
  const float* w;          /* F32: [O, K] row-major */
  const int8_t* q8;        /* I8: [O, K] */
  const uint8_t* q4;       /* I4: [O, Kp/2] bytes */
  const float* s;          /* I8: [O]; I4: [O, Kp/g] */
  int g, Kp;
} arc_mat;

typedef struct arc_weights arc_weights;
arc_weights* arc_weights_open(const char* path);   /* NULL + arc_seterr */
void arc_weights_close(arc_weights* w);
int arc_weights_has(const arc_weights* w, const char* name);
/* A float tensor (any storage, dequantized if needed) as an owned or mapped f32 array. n_expect > 0 checks the
 * element count. The returned pointer lives as long as the arc_weights. NULL + arc_seterr if missing. */
const float* arc_weights_f32(arc_weights* w, const char* name, int64_t n_expect);
/* A >=2-D weight as a matrix [shape[0], prod(shape[1:])], quantized storage kept. NULL + arc_seterr if missing or
 * the shape disagrees (O_expect / K_expect <= 0: not checked). */
const arc_mat* arc_weights_mat(arc_weights* w, const char* name, int O_expect, int K_expect);
/* Shape of a tensor in the logical (unpacked) layout; returns ndim, 0 if absent or if the layout shape is invalid
 * (more than 8 dims, a negative or non-integer dim). */
int arc_weights_shape(const arc_weights* w, const char* name, int64_t shape[8]);
const char* arc_weights_path(const arc_weights* w);
/* printf-style name helper: a static thread-local buffer, valid until the next call. */
const char* arc_namef(const char* fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;

/* ------------------------------------------------------------------ kernels */
/* 1 when the CPU and OS support AVX2 + FMA (x86-64) and ARC_NO_SIMD is not set; 0 elsewhere. */
int arc_cpu_avx2(void);
/* C[M,N] (+)= A[M,K] * B[N,K]^T (+ bias[N]).  acc: add to C instead of overwriting. */
void arc_gemm_nt(arc_pool* p, int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc,
                const float* bias, int acc);
/* C[M,N] (+)= A[M,K] * B[K,N] (+ bias[N]). */
void arc_gemm_nn(arc_pool* p, int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc,
                const float* bias, int acc);
/* y[T,O] (+)= x[T,K] W^T + bias: a linear layer / 1x1 conv with any weight storage. */
void arc_linear(arc_pool* p, const float* x, int T, int ldx, const arc_mat* W, const float* bias, float* y, int ldy,
               int acc);
/* Conv1d on time-major data (groups = 1): x [T, C] (row stride ldx), W [O, C, k] as arc_mat (K = C*k), zero padding
 * pad_l / pad_r, stride, dilation; y [T_out, O] with T_out = (T + pad_l + pad_r - dil*(k-1) - 1) / stride + 1, which
 * is returned (-1 on allocation failure). For a grouped conv call it per group with arc_mat_rows() and column
 * offsets into x / y. 0 when the padded input is shorter than the kernel span; -1 also for a bad geometry
 * (k, stride, dilation < 1 or negative padding). */
int arc_conv1d(arc_pool* p, const float* x, int T, int C, int ldx, const arc_mat* W, int k, int stride, int dil,
              int pad_l, int pad_r, const float* bias, float* y, int ldy);
/* A view of rows [r0, r0+n) of a matrix (no copy). */
void arc_mat_rows(const arc_mat* m, int r0, int n, arc_mat* out);
/* Row r of a matrix, dequantized: out[K] (embedding lookups). */
void arc_mat_row(const arc_mat* m, int r, float* out);
/* Depthwise conv1d (groups = C): w [C, k] f32, x [T, C] -> y [T, C], zero padding pad_l (right: same total). */
void arc_dwconv1d(const float* x, int T, int C, const float* w, int k, int dil, int pad_l, const float* bias, float* y);
/* In-place row ops on [T, C]. gamma / beta may be NULL. */
void arc_layernorm(float* x, int T, int C, int ld, const float* gamma, const float* beta, float eps);
void arc_rmsnorm(float* x, int T, int C, int ld, const float* gamma, float eps);
void arc_gelu(float* x, size_t n);        /* exact erf GELU (torch default) */
void arc_gelu_tanh(float* x, size_t n);   /* tanh approximation */
void arc_silu(float* x, size_t n);
void arc_relu(float* x, size_t n);
void arc_softmax_rows(float* x, int rows, int cols, int ld);
/* Multi-head scaled-dot-product attention, batch 1. q [Tq, H*dh] (stride ldq), k / v [Tk, H*dh] (ldk, ldv),
 * out [Tq, H*dh] (ldo). key_valid: NULL (all valid) or Tk bytes (1 = attend). bias: NULL or a [H?] additive
 * per-(head, q, k) bias callback-free buffer [H, Tq, Tk] (WavLM relative positions). scale <= 0: 1/sqrt(dh). */
/* Returns 0, or -1 (arc_seterr, out filled with NaN) if the score scratch cannot be allocated. */
int arc_attention(arc_pool* p, const float* q, int ldq, const float* k, int ldk, const float* v, int ldv, float* out,
                 int ldo, int Tq, int Tk, int H, int dh, const uint8_t* key_valid, const float* bias, float scale);

/* ------------------------------------------------------------------ FFT / DSP */
typedef struct arc_fft arc_fft;
arc_fft* arc_fft_create(int n);              /* n a power of two */
void arc_fft_destroy(arc_fft* f);
/* in-place complex FFT on interleaved double (re, im) pairs; inverse = 1 gives the unnormalised inverse */
void arc_fft_run(const arc_fft* f, double* a, int inverse);
/* real -> half spectrum (n/2+1 complex, interleaved) */
void arc_rfft(const arc_fft* f, const float* x, double* spec, double* work);
/* half spectrum -> real (numpy irfft: imag of DC / Nyquist ignored), normalised by 1/n */
void arc_irfft(const arc_fft* f, const double* spec, float* y, double* work);
void arc_hann_periodic(float* w, int n);

#ifdef __cplusplus
}
#endif
#endif
