/* The n-gram LM (arc_lm.c), no weights needed: the key hash (the same values the Python writer gives), loading and
 * backoff on a toy model written here, every malformed-file error, and the fusion maths on a small row (weight 0 leaves
 * the row bit for bit; tags neutral; the end token follows the audio's best; the deficit cap).
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "arc_lm.h"

static int fails = 0, checks = 0;
#define CHECK(c, ...)                   \
  do {                                  \
    checks++;                           \
    if (!(c)) {                         \
      fails++;                          \
      printf("FAIL %d: ", __LINE__);    \
      printf(__VA_ARGS__);              \
      printf("\n");                     \
    }                                   \
  } while (0)

typedef struct {
  int ids[8], n;
  float logp, bow;
} gram;

typedef struct {
  uint64_t key;
  float logp, bow;
} entry;

/* a minimal ARCLM1 writer (the format in arc_lm.h); cap 0 = the smallest power of 2 >= 2 x grams */
static int write_lm(const char* path, int order, int vocab, float uns, const unsigned char* scored, const gram* g, int ng,
                    int cap) {
  if (!cap)
    for (cap = 2; cap < 2 * ng; cap *= 2) {}
  entry* tab = calloc((size_t)cap, sizeof *tab);
  for (int i = 0; i < ng; ++i) {
    uint64_t k = arc_lm_key(g[i].ids, g[i].n);
    uint32_t s = (uint32_t)(k ^ (k >> 32)) & (uint32_t)(cap - 1);
    int tries = 0;
    while (tab[s].key && tries++ < cap) s = (s + 1) & (uint32_t)(cap - 1);
    if (tab[s].key) break;   /* full: written as it is (a test of the loader) */
    tab[s] = (entry){k, g[i].logp, g[i].bow};
  }
  FILE* f = fopen(path, "wb");
  if (!f) return -1;
  int32_t h[4] = {order, vocab, 0, cap};
  memcpy(&h[2], &uns, 4);
  fwrite("ARCLM1\0\0", 1, 8, f);
  fwrite(h, 4, 4, f);
  fwrite(scored, 1, (size_t)vocab, f);
  static const unsigned char zero[8];
  fwrite(zero, 1, (size_t)(((vocab + 7) & ~7) - vocab), f);
  fwrite(tab, sizeof *tab, (size_t)cap, f);
  fclose(f);
  free(tab);
  return 0;
}

static void write_bytes(const char* path, const void* d, size_t n) {
  FILE* f = fopen(path, "wb");
  fwrite(d, 1, n, f);
  fclose(f);
}

static unsigned char* slurp(const char* path, size_t* n) {
  FILE* f = fopen(path, "rb");
  fseek(f, 0, SEEK_END);
  *n = (size_t)ftell(f);
  fseek(f, 0, SEEK_SET);
  unsigned char* d = malloc(*n);
  if (fread(d, 1, *n, f) != *n) *n = 0;
  fclose(f);
  return d;
}

static int expect_error(const char* path, const char* what) {
  char err[256] = "";
  arc_lm* lm = arc_lm_load(path, err, sizeof err);
  CHECK(!lm, "%s loaded", what);
  CHECK(strstr(err, what) != NULL, "%s: error was [%s]", what, err);
  arc_lm_free(lm);
  return 0;
}

int main(int argc, char** argv) {
  const char* dir = argc > 1 ? argv[1] : ".";
  char p_ok[1024], p_bad[1024];
  snprintf(p_ok, sizeof p_ok, "%s/test_lm_ok.arclm", dir);
  snprintf(p_bad, sizeof p_bad, "%s/test_lm_bad.arclm", dir);

  /* the key: these values come from the Python writer (FNV-1a 64 over int32 bytes) */
  {
    int a[] = {1}, b[] = {1, 5, 7}, c[] = {32599, 3};
    CHECK(arc_lm_key(a, 1) == 0x0f6ee3214f2608b7ULL, "key [1]");
    CHECK(arc_lm_key(b, 3) == 0xd249f93dd7fb53afULL, "key [1,5,7]");
    CHECK(arc_lm_key(c, 2) == 0x26fde75946d1bec9ULL, "key [32599,3]");
  }

  /* a toy trigram over vocab 10: 1 = <s>, 2 = </s>, 3-7 text, 8-9 tags (not scored) */
  enum { VOC = 10 };
  unsigned char scored[VOC] = {0, 1, 1, 1, 1, 1, 1, 1, 0, 0};
  gram g[] = {
      {{3}, 1, -1.0f, -0.5f},   {{4}, 1, -2.0f, -0.25f}, {{5}, 1, -3.0f, 0},  {{6}, 1, -4.0f, 0}, {{7}, 1, -5.0f, 0},
      {{2}, 1, -2.5f, 0},       {{1}, 1, -20.0f, -0.1f}, {{1, 3}, 2, -0.2f, -0.7f}, {{3, 4}, 2, -0.3f, -0.6f},
      {{1, 3, 4}, 3, -0.05f, 0}, {{4, 5}, 2, 1.0f, -0.4f},   /* (4 5): a context only (positive logp) */
  };
  int ng = (int)(sizeof g / sizeof g[0]);
  write_lm(p_ok, 3, VOC, -3.0f, scored, g, ng, 0);
  char err[256] = "";
  arc_lm* lm = arc_lm_load(p_ok, err, sizeof err);
  CHECK(lm != NULL, "load: %s", err);
  if (!lm) return 1;
  CHECK(arc_lm_order(lm) == 3 && arc_lm_vocab(lm) == VOC && arc_lm_entries(lm) == ng, "header");
  CHECK(arc_lm_scored(lm, 3) && !arc_lm_scored(lm, 8) && !arc_lm_scored(lm, 0) && !arc_lm_scored(lm, 99) &&
            !arc_lm_scored(lm, -1), "scored");
  {
    int h0[] = {1}, h1[] = {1, 3}, h2[] = {3}, h3[] = {7, 3}, h4[] = {4, 5}, h5[] = {5};
    CHECK(fabsf(arc_lm_logp(lm, h1, 2, 4) - (-0.05f)) < 1e-6f, "trigram");
    CHECK(fabsf(arc_lm_logp(lm, h0, 1, 3) - (-0.2f)) < 1e-6f, "bigram");
    CHECK(fabsf(arc_lm_logp(lm, h3, 2, 4) - (-0.3f)) < 1e-6f, "(7 3) unseen, no bow: bigram (3 4)");
    CHECK(fabsf(arc_lm_logp(lm, h2, 1, 5) - (-0.5f - 3.0f)) < 1e-6f, "backoff to the unigram: bow(3) + p(5)");
    CHECK(fabsf(arc_lm_logp(lm, h1, 2, 5) - (-0.7f - 0.5f - 3.0f)) < 1e-6f, "two backoffs: bow(1 3) + bow(3) + p(5)");
    CHECK(fabsf(arc_lm_logp(lm, h4, 2, 6) - (-0.4f + 0.0f - 4.0f)) < 1e-6f, "context-only entry: its bow, never its logp");
    CHECK(fabsf(arc_lm_logp(lm, h5, 1, 5) - (-3.0f)) < 1e-6f, "(5 5): unigram");
    CHECK(fabsf(arc_lm_logp(lm, NULL, 0, 7) - (-5.0f)) < 1e-6f, "no history");
    int h6[] = {1, 3, 4};   /* longer than order - 1: only the last two count */
    CHECK(fabsf(arc_lm_logp(lm, h6, 3, 5) - (-0.6f - 0.25f - 3.0f)) < 1e-6f, "long history: bow(3 4) + bow(4) + p(5)");
  }

  /* fusion on a row of 10 */
  {
    float raw[VOC] = {-9.0f, -9.0f, 1.0f, 3.0f, 2.9f, 2.5f, -0.5f, -9.0f, 2.95f, -9.0f};
    float row[VOC], ref[VOC];
    memcpy(row, raw, sizeof row);
    int gen[] = {1, 3};   /* <s> then 3 */
    arc_lm_fuse(lm, 0.0f, 16, row, VOC, gen, 2, 2);
    CHECK(!memcmp(row, raw, sizeof row), "weight 0 leaves the row as it is");
    arc_lm_fuse(NULL, 0.5f, 16, row, VOC, gen, 2, 2);
    CHECK(!memcmp(row, raw, sizeof row), "no LM leaves the row as it is");

    /* by hand: log-softmax, then the candidates within 4 of the best (3 at 3.0): 3, 8 (a tag), 4, 5, 6 and 2 (the end) */
    double mx = 3.0, s = 0;
    for (int i = 0; i < VOC; ++i) s += exp((double)raw[i] - mx);
    float lse = (float)(mx + log(s));
    for (int i = 0; i < VOC; ++i) ref[i] = raw[i] - lse;
    const float w = 0.5f;
    int hist[] = {1, 3};
    float l3 = arc_lm_logp(lm, hist, 2, 3), l4 = arc_lm_logp(lm, hist, 2, 4), l5 = arc_lm_logp(lm, hist, 2, 5),
          l6 = arc_lm_logp(lm, hist, 2, 6), l7 = arc_lm_logp(lm, hist, 2, 7);
    float top = fmaxf(fmaxf(l3, l4), fmaxf(l5, l6));
    float want[VOC];
    for (int i = 0; i < VOC; ++i) want[i] = ref[i] - w * 6.0f;   /* not candidates (rows below the margin) */
    want[3] = ref[3] + w * fmaxf(l3 - top, -6.0f);
    want[4] = ref[4] + w * fmaxf(l4 - top, -6.0f);
    want[5] = ref[5] + w * fmaxf(l5 - top, -6.0f);
    want[6] = ref[6] + w * fmaxf(l6 - top, -6.0f);
    want[8] = ref[8];                                      /* a tag: neutral */
    want[2] = ref[2] + w * fmaxf(l3 - top, -6.0f);         /* the end token: the best token's adjustment */
    want[7] = ref[7] - w * 6.0f;                           /* implausible (and l7 not asked) */
    (void)l7;
    memcpy(row, raw, sizeof row);
    arc_lm_fuse(lm, w, 16, row, VOC, gen, 2, 2);
    float md = 0;
    for (int i = 0; i < VOC; ++i) md = fmaxf(md, fabsf(row[i] - want[i]));
    CHECK(md < 1e-5f, "fusion by hand: max diff %g", md);
    CHECK(row[4] > row[3], "the LM's favourite (4 after <s> 3) now leads");

    /* topk 2: only 3 and 8 are candidates; 4 gets the floor */
    memcpy(row, raw, sizeof row);
    arc_lm_fuse(lm, w, 2, row, VOC, gen, 2, 2);
    CHECK(fabsf(row[4] - (ref[4] - w * 6.0f)) < 1e-5f, "topk 2: outside the top gets the floor");
    CHECK(fabsf(row[3] - ref[3]) < 1e-5f, "topk 2: the only scored candidate is its own favourite");
    CHECK(fabsf(row[2] - ref[2]) < 1e-5f, "topk 2: the end token follows the best (no charge)");

    /* the deficit is capped at 6 */
    float raw2[VOC] = {-9, -9, -9, -9, 3.0f, -9, -9, 2.9f, -9, -9};
    int g2[] = {1, 3};
    memcpy(row, raw2, sizeof row);
    arc_lm_fuse(lm, 1.0f, 16, row, VOC, g2, 2, 2);
    s = 0;
    for (int i = 0; i < VOC; ++i) s += exp((double)raw2[i] - 3.0);
    lse = (float)(3.0 + log(s));
    float d7 = arc_lm_logp(lm, hist, 2, 7) - arc_lm_logp(lm, hist, 2, 4);
    CHECK(d7 < -6.0f, "the deficit is past the cap (%g)", d7);
    CHECK(fabsf(row[7] - (raw2[7] - lse - 6.0f)) < 1e-5f, "cap");
  }
  arc_lm_free(lm);

  /* malformed files */
  {
    size_t n;
    unsigned char* good = slurp(p_ok, &n);
    unsigned char* b = malloc(n + 16);
    snprintf(p_bad, sizeof p_bad, "%s/test_lm_missing.arclm", dir);
    remove(p_bad);
    expect_error(p_bad, "cannot open");
    snprintf(p_bad, sizeof p_bad, "%s/test_lm_bad.arclm", dir);
    write_bytes(p_bad, good, 10);
    expect_error(p_bad, "too short");
    memcpy(b, good, n);
    memcpy(b, "GGUF\0\0\0\0", 8);
    write_bytes(p_bad, b, n);
    expect_error(p_bad, "bad magic");
    memcpy(b, good, n);
    memcpy(b, "ARCLM2\0\0", 8);
    write_bytes(p_bad, b, n);
    expect_error(p_bad, "unsupported ARCLM version");
    memcpy(b, good, n);
    int32_t v = 9;
    memcpy(b + 8, &v, 4);
    write_bytes(p_bad, b, n);
    expect_error(p_bad, "order");
    memcpy(b, good, n);
    v = -1;
    memcpy(b + 12, &v, 4);
    write_bytes(p_bad, b, n);
    expect_error(p_bad, "vocabulary");
    memcpy(b, good, n);
    float f = 0.5f;
    memcpy(b + 16, &f, 4);
    write_bytes(p_bad, b, n);
    expect_error(p_bad, "unscored");
    memcpy(b, good, n);
    v = 24;
    memcpy(b + 20, &v, 4);
    write_bytes(p_bad, b, n);
    expect_error(p_bad, "power of 2");
    write_bytes(p_bad, good, n - 16);
    expect_error(p_bad, "size");
    memcpy(b, good, n);
    memset(b + n, 0, 16);
    write_bytes(p_bad, b, n + 16);
    expect_error(p_bad, "size");
    /* a NaN log-probability in the first used slot */
    memcpy(b, good, n);
    for (size_t o = 24 + 16; o + 16 <= n; o += 16) {
      uint64_t k;
      memcpy(&k, b + o, 8);
      if (k) {
        f = NAN;
        memcpy(b + o + 8, &f, 4);
        break;
      }
    }
    write_bytes(p_bad, b, n);
    expect_error(p_bad, "bad table entry");
    free(b);
    free(good);
    /* a full table (no empty slot: a lookup could never end) */
    gram full[4] = {{{3}, 1, -1, 0}, {{4}, 1, -1, 0}, {{5}, 1, -1, 0}, {{6}, 1, -1, 0}};
    write_lm(p_bad, 1, VOC, -3.0f, scored, full, 4, 4);
    expect_error(p_bad, "no empty slot");
    remove(p_bad);
    remove(p_ok);
  }

  printf("lm: %d/%d checks\n", checks - fails, checks);
  return fails ? 1 : 0;
}
