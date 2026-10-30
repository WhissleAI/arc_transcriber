/* Arc's transcript text: segmentation, loop fix, tag readout, context prompt. See arc_text.h.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include "arc_text.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arc_base.h"

/* ================================================================ segmenter */

void arc_seg_cfg_default(arc_seg_cfg* c) {
  c->pause_s = 0.4f;
  c->min_s = 1.0f;
  c->max_s = 15.0f;
  c->cut_lookback_s = 3.0f;
  c->min_voiced_s = 0.3f;
}

typedef struct {
  int f;
  double db;
  int q;
} segf;

struct arc_seg {
  int pause_n, min_n, max_n, look, min_voiced;
  double hist[300];
  int n_hist, hist_pos;
  segf* cur;
  int n_cur, cap_cur;
  int low_run;
};

static int iround(double x) { return (int)lrint(x); }   /* Python round(): half to even, as lrint by default */

arc_seg* arc_seg_new(const arc_seg_cfg* c) {
  arc_seg* s = arc_calloc(sizeof *s);
  if (!s) return NULL;
  arc_seg_cfg d;
  if (!c) {
    arc_seg_cfg_default(&d);
    c = &d;
  }
  s->pause_n = iround(c->pause_s * 10);
  s->min_n = iround(c->min_s * 10);
  s->max_n = iround(c->max_s * 10);
  s->look = iround(c->cut_lookback_s * 10);
  s->min_voiced = iround(c->min_voiced_s * 10);
  return s;
}

void arc_seg_free(arc_seg* s) {
  if (!s) return;
  arc_free(s->cur);
  arc_free(s);
}

double arc_frame_db(const float* a, int n) {
  if (n <= 0) return -120.0;
  double s = 0;   /* numpy: mean of float32 squares (pairwise sum in float32); double is within rounding */
  for (int i = 0; i < n; ++i) s += (double)a[i] * a[i];
  return 20.0 * log10(sqrt(s / n) + 1e-6);
}

static int cmp_d(const void* a, const void* b) {
  double x = *(const double*)a, y = *(const double*)b;
  return x < y ? -1 : x > y;
}

/* np.percentile(hist, 10), linear interpolation */
static double pct10(const arc_seg* s) {
  double v[300];
  int n = s->n_hist;
  for (int i = 0; i < n; ++i) v[i] = s->hist[i];
  qsort(v, (size_t)n, sizeof(double), cmp_d);
  double pos = 0.1 * (n - 1);
  int lo = (int)floor(pos);
  double fr = pos - lo;
  return lo + 1 < n ? v[lo] + (v[lo + 1] - v[lo]) * fr : v[lo];
}

static int quiet(const arc_seg* s, double db) {
  double floor_ = s->n_hist >= 20 ? pct10(s) : -60.0;
  double thr = floor_ + 4.0 > -50.0 ? floor_ + 4.0 : -50.0;
  return db < thr || db < -55.0;
}

/* trim quiet frames at both ends; enough voiced frames -> an utterance */
static int emit(const arc_seg* s, const segf* fr, int n, int* f0, int* f1) {
  int a = 0, b = n;
  while (b > a && fr[b - 1].q) b--;
  while (a < b && fr[a].q) a++;
  int voiced = 0;
  for (int i = a; i < b; ++i) voiced += !fr[i].q;
  if (voiced < s->min_voiced || b <= a) return 0;
  *f0 = fr[a].f;
  *f1 = fr[b - 1].f;
  return 1;
}

int arc_seg_feed(arc_seg* s, int f, double db, int* f0, int* f1, const char** reason) {
  /* hist = (hist + [db])[-300:] (kept in order: the percentile does not care) */
  if (s->n_hist < 300) s->hist[s->n_hist++] = db;
  else {
    s->hist[s->hist_pos] = db;
    s->hist_pos = (s->hist_pos + 1) % 300;
  }
  int q = quiet(s, db);
  if (!s->n_cur && q) return 0;
  if (s->n_cur == s->cap_cur) {
    int nc = s->cap_cur ? s->cap_cur * 2 : 256;
    segf* p = arc_malloc(sizeof(segf) * (size_t)nc);
    if (!p) return 0;
    if (s->n_cur) memcpy(p, s->cur, sizeof(segf) * (size_t)s->n_cur);
    arc_free(s->cur);
    s->cur = p;
    s->cap_cur = nc;
  }
  s->cur[s->n_cur++] = (segf){f, db, q};
  s->low_run = q ? s->low_run + 1 : 0;
  int n = s->n_cur;
  if (s->low_run >= s->pause_n && n - s->low_run >= s->min_n) {
    int r = emit(s, s->cur, n, f0, f1);
    s->n_cur = 0;
    s->low_run = 0;
    if (r) *reason = "pause";
    return r;
  }
  if (n >= s->max_n) {
    int lt = s->look < n ? s->look : n;
    const segf* tail = s->cur + n - lt;
    int k = 0;
    for (int i = 1; i < lt; ++i)
      if (tail[i].db < tail[k].db) k = i;   /* min(): the first lowest */
    int cut = n - lt + k;
    int r = emit(s, s->cur, cut + 1, f0, f1);
    memmove(s->cur, s->cur + cut + 1, sizeof(segf) * (size_t)(n - cut - 1));
    s->n_cur = n - cut - 1;
    s->low_run = 0;
    for (int i = 0; i < s->n_cur; ++i) s->low_run = s->cur[i].q ? s->low_run + 1 : 0;
    if (r) *reason = "max_length";
    return r;
  }
  return 0;
}

int arc_seg_flush(arc_seg* s, int* f0, int* f1, const char** reason) {
  int r = s->n_cur ? emit(s, s->cur, s->n_cur, f0, f1) : 0;
  s->n_cur = 0;
  s->low_run = 0;
  if (r) *reason = "end";
  return r;
}

/* ================================================================ words */

int arc_split_words(const char* text, char*** out) {
  int cap = 16, n = 0;
  char** w = arc_malloc(sizeof(char*) * (size_t)cap);
  const char* p = text;
  while (w && *p) {
    while (*p && isspace((unsigned char)*p)) p++;
    if (!*p) break;
    const char* q = p;
    while (*q && !isspace((unsigned char)*q)) q++;
    if (n == cap) {
      char** nw = arc_malloc(sizeof(char*) * (size_t)cap * 2);
      memcpy(nw, w, sizeof(char*) * (size_t)n);
      arc_free(w);
      w = nw;
      cap *= 2;
    }
    w[n] = arc_malloc((size_t)(q - p) + 1);
    memcpy(w[n], p, (size_t)(q - p));
    w[n][q - p] = 0;
    n++;
    p = q;
  }
  *out = w;
  return n;
}

void arc_free_words(char** w, int n) {
  if (!w) return;
  for (int i = 0; i < n; ++i) arc_free(w[i]);
  arc_free(w);
}

/* re.sub("[^\w']", '', w.lower()): ASCII letters lowered, digits, '_', apostrophes and non-ASCII (letters) kept */
static char* lkey(const char* w) {
  size_t n = strlen(w);
  char* k = arc_malloc(n + 1);
  size_t j = 0;
  for (size_t i = 0; i < n; ++i) {
    unsigned char c = (unsigned char)w[i];
    if (c >= 0x80) {
      /* keep multibyte letters, drop typographic punctuation (U+2010..U+206F) */
      if (c == 0xE2 && i + 1 < n && ((unsigned char)w[i + 1] == 0x80 || (unsigned char)w[i + 1] == 0x81)) {
        i += 2;
        continue;
      }
      k[j++] = (char)c;
    } else if (isalnum(c) || c == '_' || c == '\'') k[j++] = (char)tolower(c);
  }
  k[j] = 0;
  return k;
}

static int all_dashes(const char* w) {
  if (!*w) return 0;
  for (const char* p = w; *p;) {
    if (*p == '-') p++;
    else if (!strncmp(p, "\xE2\x80\x93", 3) || !strncmp(p, "\xE2\x80\x94", 3)) p += 3;
    else return 0;
  }
  return 1;
}

char* arc_loopfix(const char* text, int K) {
  char** all;
  int na = arc_split_words(text, &all);
  char** w = arc_malloc(sizeof(char*) * (size_t)(na + 1));
  char** key = arc_malloc(sizeof(char*) * (size_t)(na + 1));
  int n = 0;
  for (int i = 0; i < na; ++i) {
    char* k = lkey(all[i]);
    if (*k || !all_dashes(all[i])) {
      w[n] = all[i];
      key[n++] = k;
    } else arc_free(k);
  }
  size_t cap = strlen(text) + 2;
  char* out = arc_malloc(cap);
  size_t ol = 0;
  out[0] = 0;
#define PUT(s)                                    \
  do {                                            \
    size_t l_ = strlen(s);                        \
    if (ol) out[ol++] = ' ';                      \
    memcpy(out + ol, s, l_);                      \
    ol += l_;                                     \
    out[ol] = 0;                                  \
  } while (0)
  int i = 0;
  while (i < n) {
    int done = 0;
    for (int g = 1; g <= 4 && !done; ++g) {
      if (i + g > n) continue;
      int nonempty = 1;
      for (int j = 0; j < g; ++j)
        if (!*key[i + j]) nonempty = 0;
      if (!nonempty) continue;
      int r = 1;
      while (i + (r + 1) * g <= n) {
        int eq = 1;
        for (int j = 0; j < g; ++j)
          if (strcmp(key[i + r * g + j], key[i + j])) {
            eq = 0;
            break;
          }
        if (!eq) break;
        r++;
      }
      if (r > K + 1) {
        for (int j = 0; j < K * g; ++j) PUT(w[i + j]);
        i += r * g;
        done = 1;
      }
    }
    if (!done) {
      PUT(w[i]);
      i++;
    }
  }
#undef PUT
  for (int j = 0; j < n; ++j) arc_free(key[j]);
  arc_free(key);
  arc_free(w);
  arc_free_words(all, na);
  return out;
}

/* ================================================================ tags */

static const char* TAGS[] = {"ENTITY_PERSON", "ENTITY_TEAM", "ENTITY_ORG", "ENTITY_PLACE", "ENTITY_WORK",
                             "ENTITY_EVENT", "ENTITY_LEGAL", "ENTITY_GAME", "ENTITY_MAP", "NUMBER", "NUMBER_CLOCK",
                             "NUMBER_COUNT", "NUMBER_DATE", "NUMBER_DURATION", "NUMBER_FIELDPOS", "NUMBER_ID",
                             "NUMBER_JERSEY", "NUMBER_MEASURE", "NUMBER_MONEY", "NUMBER_ODDS", "NUMBER_ORDINAL",
                             "NUMBER_PERCENT", "NUMBER_REF", "NUMBER_SCORE", "NUMBER_STAT", "NUMBER_TIME", "END",
                             "ENTITY_LAW", "ENTITY_PRODUCT", "ENTITY_TERM", "_RESERVED_A", "_RESERVED_B",
                             "CONTACT_PHONE", "CONTACT_EMAIL", "CONTACT_URL", "CONTACT_ADDRESS"};
#define N_TAGS (int)(sizeof TAGS / sizeof *TAGS)
#define BASE_ID 32600
#define END_ID (BASE_ID + 26)

const char* arc_tag_name(int id) {
  int i = id - BASE_ID;
  if (i < 0 || i >= N_TAGS || TAGS[i][0] == '_') return NULL;
  return TAGS[i];
}

int arc_tag_id(const char* name) {
  for (int i = 0; i < N_TAGS; ++i)
    if (TAGS[i][0] != '_' && !strcmp(TAGS[i], name)) return BASE_ID + i;
  return -1;
}

int arc_readout(const arc_tok* t, const int* ids, int n, char*** words_out, int* n_words, arc_span** spans_out,
                int* n_spans) {
  char** words = arc_malloc(sizeof(char*) * (size_t)(n + 1));
  arc_span* spans = arc_malloc(sizeof(arc_span) * (size_t)(n + 1));
  int* cur = arc_malloc(sizeof(int) * (size_t)(n + 1));
  int nw = 0, ns = 0, nc = 0, open_tag = -1, open_w = 0;
#define FLUSH()                                         \
  do {                                                  \
    if (nc) {                                           \
      words[nw++] = arc_tok_decode(t, cur, nc);         \
      nc = 0;                                           \
    }                                                   \
  } while (0)
  for (int i = 0; i < n; ++i) {
    int id = ids[i];
    if (arc_tag_name(id)) {
      FLUSH();
      if (open_tag >= 0 && nw > open_w) spans[ns++] = (arc_span){open_tag, open_w, nw};
      open_tag = -1;
      if (id != END_ID) {
        open_tag = id;
        open_w = nw;
      }
      continue;
    }
    if (id < 3 || id >= 32000) continue;
    const char* p = arc_tok_piece(t, id);
    if (p && !strncmp(p, "\xE2\x96\x81", 3) && nc) FLUSH();
    cur[nc++] = id;
  }
  FLUSH();
#undef FLUSH
  if (open_tag >= 0 && nw > open_w) spans[ns++] = (arc_span){open_tag, open_w, nw};
  /* drop empty words, remap spans */
  int* mp = arc_malloc(sizeof(int) * (size_t)(nw + 1));
  int n2 = 0;
  for (int j = 0; j < nw; ++j) {
    if (words[j][0]) {
      mp[j] = n2;
      words[n2++] = words[j];
    } else {
      mp[j] = -1;
      arc_free(words[j]);
    }
  }
  int ns2 = 0;
  for (int k = 0; k < ns; ++k) {
    int lo = -1, hi = -1;
    for (int j = spans[k].a; j < spans[k].b; ++j)
      if (mp[j] >= 0) {
        if (lo < 0) lo = mp[j];
        hi = mp[j];
      }
    if (lo >= 0) spans[ns2++] = (arc_span){spans[k].tag, lo, hi + 1};
  }
  arc_free(mp);
  arc_free(cur);
  *words_out = words;
  *n_words = n2;
  *spans_out = spans;
  *n_spans = ns2;
  return 0;
}

/* difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks() on string keys */
typedef struct {
  int a, b, n;
} block;

static void longest(char** A, char** B, int alo, int ahi, int blo, int bhi, int* bi, int* bj, int* bk) {
  int besti = alo, bestj = blo, bestsize = 0;
  int w = bhi - blo + 1;
  int* j2len = arc_calloc(sizeof(int) * (size_t)(w + 1));
  int* nj2 = arc_calloc(sizeof(int) * (size_t)(w + 1));
  for (int i = alo; i < ahi; ++i) {
    memset(nj2, 0, sizeof(int) * (size_t)(w + 1));
    for (int j = blo; j < bhi; ++j) {
      if (strcmp(A[i], B[j])) continue;
      int k = (j > blo ? j2len[j - 1 - blo] : 0) + 1;
      nj2[j - blo] = k;
      if (k > bestsize) {
        besti = i - k + 1;
        bestj = j - k + 1;
        bestsize = k;
      }
    }
    int* t = j2len;
    j2len = nj2;
    nj2 = t;
  }
  arc_free(j2len);
  arc_free(nj2);
  *bi = besti;
  *bj = bestj;
  *bk = bestsize;
}

static int cmp_block(const void* x, const void* y) {
  const block *a = x, *b = y;
  if (a->a != b->a) return a->a < b->a ? -1 : 1;
  if (a->b != b->b) return a->b < b->b ? -1 : 1;
  return a->n < b->n ? -1 : a->n > b->n;
}

int arc_map_spans(char** src, int ns, const arc_span* spans, int nsp, char** dst, int nd, arc_span* out) {
  if (!nsp) return 0;
  char** A = arc_malloc(sizeof(char*) * (size_t)(ns + 1));
  char** B = arc_malloc(sizeof(char*) * (size_t)(nd + 1));
  for (int i = 0; i < ns; ++i) A[i] = lkey(src[i]);
  for (int i = 0; i < nd; ++i) B[i] = lkey(dst[i]);
  int same = ns == nd;
  for (int i = 0; same && i < ns; ++i) same = !strcmp(A[i], B[i]);
  int no = 0;
  if (same) {
    memcpy(out, spans, sizeof(arc_span) * (size_t)nsp);
    no = nsp;
  } else {
    int* m = arc_malloc(sizeof(int) * (size_t)(ns + 1));
    for (int i = 0; i < ns; ++i) m[i] = -1;
    block* bl = arc_malloc(sizeof(block) * (size_t)(ns + nd + 2));
    int nb = 0, nq = 1;
    int(*q)[4] = arc_malloc(sizeof(int[4]) * (size_t)(ns + nd + 2));
    q[0][0] = 0;
    q[0][1] = ns;
    q[0][2] = 0;
    q[0][3] = nd;
    while (nq) {
      int alo = q[nq - 1][0], ahi = q[nq - 1][1], blo = q[nq - 1][2], bhi = q[nq - 1][3];
      nq--;
      int i, j, k;
      longest(A, B, alo, ahi, blo, bhi, &i, &j, &k);
      if (k) {
        bl[nb++] = (block){i, j, k};
        if (alo < i && blo < j) {
          q[nq][0] = alo; q[nq][1] = i; q[nq][2] = blo; q[nq][3] = j;
          nq++;
        }
        if (i + k < ahi && j + k < bhi) {
          q[nq][0] = i + k; q[nq][1] = ahi; q[nq][2] = j + k; q[nq][3] = bhi;
          nq++;
        }
      }
    }
    qsort(bl, (size_t)nb, sizeof(block), cmp_block);
    for (int x = 0; x < nb; ++x)
      for (int k = 0; k < bl[x].n; ++k) m[bl[x].a + k] = bl[x].b + k;
    for (int s = 0; s < nsp; ++s) {
      int lo = -1, hi = -1;
      for (int j = spans[s].a; j < spans[s].b && j < ns; ++j)
        if (m[j] >= 0) {
          if (lo < 0 || m[j] < lo) lo = m[j];
          if (m[j] > hi) hi = m[j];
        }
      if (lo >= 0) out[no++] = (arc_span){spans[s].tag, lo, hi + 1};
    }
    arc_free(q);
    arc_free(bl);
    arc_free(m);
  }
  for (int i = 0; i < ns; ++i) arc_free(A[i]);
  for (int i = 0; i < nd; ++i) arc_free(B[i]);
  arc_free(A);
  arc_free(B);
  return no;
}

/* ================================================================ context prompt */

#define CTX_PREV 32640
#define CTX_KB 32641
#define CTX_DOM 32642
#define CTX_SEP 32644
static const char* KB_TYPES[] = {"person", "team", "org", "place", "law", "product", "term", "id"};

static int type_id(const char* y) {
  if (!y) return -1;
  for (int i = 0; i < 8; ++i)
    if (!strcmp(KB_TYPES[i], y)) return 32645 + i;
  return -1;
}

int arc_ctx_prompt(const arc_tok* t, const arc_ctx* c, const char* mode, int max_tokens, int* out) {
  const int prev_max = 48, kb_max = 40;
  if (!c || !strcmp(mode, "none")) return 0;
  int buf[4096], n = 0, tmp[1024];
  if (c->dom && *c->dom) {
    char d[256];
    snprintf(d, sizeof d, "%s", c->dom);
    for (char* p = d; *p; ++p)
      if (*p == '_') *p = ' ';
    int m = arc_tok_encode(t, d, tmp, 1024);
    buf[n++] = CTX_DOM;
    for (int i = 0; i < m && i < 4; ++i) buf[n++] = tmp[i];
  }
  if (!strcmp(mode, "full") && c->n_kb > 0) {
    int any_typed = 0;
    for (int i = 0; i < c->n_kb; ++i) any_typed |= type_id(c->kb_type ? c->kb_type[i] : NULL) >= 0;
    int lim = kb_max - (any_typed ? 1 : 0);
    int sel_start[256], sel_len[256], sel_type[256], nsel = 0, cost = 0;
    int pool[4096], np = 0;
    for (int i = 0; i < c->n_kb && nsel < 256; ++i) {
      int m = arc_tok_encode(t, c->kb[i], tmp, 1024);
      if (m < 0 || cost + m + 1 > lim) continue;
      if (np + m > 4096) break;
      sel_start[nsel] = np;
      sel_len[nsel] = m;
      sel_type[nsel] = type_id(c->kb_type ? c->kb_type[i] : NULL);
      memcpy(pool + np, tmp, sizeof(int) * (size_t)m);
      np += m;
      cost += m + 1;
      nsel++;
    }
    if (nsel) {
      int typed = 1;
      for (int i = 0; i < nsel; ++i) typed &= sel_type[i] >= 0;
      typed = typed && cost + 1 <= kb_max;
      int k[4096], nk = 0;
      for (int i = 0; i < nsel; ++i) {
        if (typed) {
          k[nk++] = sel_type[i];
          memcpy(k + nk, pool + sel_start[i], sizeof(int) * (size_t)sel_len[i]);
          nk += sel_len[i];
        } else {
          memcpy(k + nk, pool + sel_start[i], sizeof(int) * (size_t)sel_len[i]);
          nk += sel_len[i];
          k[nk++] = CTX_SEP;
        }
      }
      if (!typed) nk--;   /* k_ids[:-1]: no separator after the last */
      buf[n++] = CTX_KB;
      memcpy(buf + n, k, sizeof(int) * (size_t)nk);
      n += nk;
    }
  }
  if (c->prev && *c->prev) {
    int ids[4096], m = arc_tok_encode(t, c->prev, ids, 4096);
    int budget = prev_max < max_tokens - n - 1 ? prev_max : max_tokens - n - 1;
    if (budget > 4 && m > 0) {
      buf[n++] = CTX_PREV;
      int from = m > budget ? m - budget : 0;
      for (int i = from; i < m; ++i) buf[n++] = ids[i];
    }
  }
  if (n > max_tokens) n = max_tokens;
  memcpy(out, buf, sizeof(int) * (size_t)n);
  return n;
}

/* ---------------------------------------------------------------- num2words (English cardinals) */

static const char* ONES[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                             "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                             "eighteen", "nineteen"};
static const char* TENS[] = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
static const char* MAGS[] = {"", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion"};

static void below_1000(long long n, char* out) {   /* 0 < n < 1000 */
  char b[128] = "";
  if (n >= 100) {
    snprintf(b, sizeof b, "%s hundred", ONES[n / 100]);
    n %= 100;
    if (n) strcat(b, " and ");
  }
  if (n >= 20) {
    strcat(b, TENS[n / 10]);
    if (n % 10) {
      strcat(b, "-");
      strcat(b, ONES[n % 10]);
    }
  } else if (n > 0) strcat(b, ONES[n]);
  strcpy(out, b);
}

static void cardinal(long long n, char* out) {
  if (n == 0) {
    strcpy(out, "zero");
    return;
  }
  char res[512] = "";
  if (n < 0) {
    strcpy(res, "minus ");
    n = -n;
  }
  int groups[8], ng = 0;
  while (n > 0 && ng < 7) {
    groups[ng++] = (int)(n % 1000);
    n /= 1000;
  }
  int first = 1;
  for (int g = ng - 1; g >= 0; --g) {
    if (!groups[g]) continue;
    char w[128];
    below_1000(groups[g], w);
    if (!first) strcat(res, (g == 0 && groups[g] < 100) ? " and " : ", ");
    strcat(res, w);
    if (g > 0) {
      strcat(res, " ");
      strcat(res, MAGS[g]);
    }
    first = 0;
  }
  strcpy(out, res);
}

char* arc_num2words(const char* digits) {
  char out[1024];
  const char* dot = strchr(digits, '.');
  if (!dot) {
    cardinal(atoll(digits), out);
  } else {
    /* float(...) then num2words: the integer part, "point", then each digit of the shortest repr */
    double v = atof(digits);
    char r[64];
    snprintf(r, sizeof r, "%.15g", v);
    if (!strchr(r, '.')) strcat(r, ".0");
    const char* d = strchr(r, '.');
    char ip[64];
    snprintf(ip, sizeof ip, "%.*s", (int)(d - r), r);
    cardinal(atoll(ip), out);
    strcat(out, " point");
    for (const char* p = d + 1; *p; ++p)
      if (*p >= '0' && *p <= '9') {
        strcat(out, " ");
        strcat(out, ONES[*p - '0']);
      }
  }
  /* .replace(',', '') */
  char* s = arc_strdup(out);
  char* w = s;
  for (char* p = s; *p; ++p)
    if (*p != ',') *w++ = *p;
  *w = 0;
  return s;
}

char* arc_norm_prev(const char* text) {
  size_t n = strlen(text ? text : "");
  char* a = arc_malloc(n * 8 + 64);
  size_t k = 0;
  /* % -> " percent ", & -> " and ", thousands commas out, numbers as words */
  for (size_t i = 0; i < n;) {
    char c = text[i];
    if (c == '%') {
      memcpy(a + k, " percent ", 9);
      k += 9;
      i++;
    } else if (c == '&') {
      memcpy(a + k, " and ", 5);
      k += 5;
      i++;
    } else if (isdigit((unsigned char)c)) {
      char num[128];
      size_t m = 0;
      while (i < n && m < 100) {
        if (isdigit((unsigned char)text[i])) num[m++] = text[i++];
        else if (text[i] == ',' && i + 3 < n + 1 && isdigit((unsigned char)text[i + 1]) && i + 3 < n &&
                 isdigit((unsigned char)text[i + 2]) && isdigit((unsigned char)text[i + 3]))
          i++;   /* (?<=\d),(?=\d{3}) */
        else break;
      }
      if (i + 1 < n && text[i] == '.' && isdigit((unsigned char)text[i + 1])) {
        num[m++] = text[i++];
        while (i < n && isdigit((unsigned char)text[i]) && m < 120) num[m++] = text[i++];
      }
      num[m] = 0;
      char* w = arc_num2words(num);
      a[k++] = ' ';
      size_t l = strlen(w);
      memcpy(a + k, w, l);
      k += l;
      a[k++] = ' ';
      arc_free(w);
    } else if (!strncmp(text + i, "\xE2\x80\x99", 3)) {   /* ’ -> ' */
      a[k++] = '\'';
      i += 3;
    } else {
      a[k++] = (char)tolower((unsigned char)c);
      i++;
    }
  }
  a[k] = 0;
  /* [^a-z' ]+ -> ' ' ; apostrophes not between letters -> ' ' ; collapse spaces */
  for (size_t i = 0; i < k; ++i)
    if (!((a[i] >= 'a' && a[i] <= 'z') || a[i] == '\'' || a[i] == ' ')) a[i] = ' ';
  for (size_t i = 0; i < k; ++i)
    if (a[i] == '\'') {
      int before = i > 0 && a[i - 1] >= 'a' && a[i - 1] <= 'z';
      int after = i + 1 < k && a[i + 1] >= 'a' && a[i + 1] <= 'z';
      if (!before || !after) a[i] = ' ';
    }
  char* o = arc_malloc(k + 1);
  size_t j = 0;
  for (size_t i = 0; i < k; ++i) {
    if (a[i] == ' ') {
      if (j && o[j - 1] != ' ') o[j++] = ' ';
    } else o[j++] = a[i];
  }
  while (j && o[j - 1] == ' ') j--;
  o[j] = 0;
  arc_free(a);
  return o;
}
