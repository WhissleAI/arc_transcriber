/* Arc Transcriber Space (NFT2): a C port of arc_audio/nft2.py's ONNX path (OnnxEncoder, OnnxRNNT, BoostTrie,
 * Pieces, Nft2Finals) and of what Arc.predict(transcript="nft2") does around it (server.py Heads / mem_with_pos /
 * ServerFeatures / pool_range, ctxprompt.build_prompt_v2, api.py context_dict / _post). The encoder and RNNT nets are
 * behind arc_nft2_be.h; the trunk, heads, names and numbers are arc_engine.c / arc_link.c with the NFT2 settings.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include "arc_nft2.h"
#include "arc_post.h"
#include "arc_echo.h"
#ifdef ARC_WITH_ARCTAG
#include "arctag.h"
#endif

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arc_base.h"
#include "arc_engine.h"
#include "arc_moon.h"
#include "arc_nft2_be.h"
#include "arc_store.h"
#include "arc_text.h"
#include "arc_tok.h"

#define SR 16000
#define HOP 160
#define NFFT 512
#define FR 1600          /* 100 ms */
#define ENC_STRIDE 1280  /* 80 ms encoder frame */
#define DIN 640

static __thread char g_err[512];

/* ================================================================== small helpers */

static char* xstrdup(const char* s) { return arc_strdup(s ? s : ""); }

typedef struct {
  char* s;
  size_t n, cap;
} sbuf;
static void sb_putn(sbuf* b, const char* s, size_t n) {
  if (b->n + n + 1 > b->cap) {
    size_t nc = (b->n + n + 1) * 2 + 32;
    char* ns = arc_malloc(nc);
    if (b->n) memcpy(ns, b->s, b->n);
    arc_free(b->s);
    b->s = ns;
    b->cap = nc;
  }
  memcpy(b->s + b->n, s, n);
  b->n += n;
  b->s[b->n] = 0;
}
static void sb_put(sbuf* b, const char* s) { sb_putn(b, s, strlen(s)); }
static char* sb_take(sbuf* b) {
  if (!b->s) return arc_strdup("");
  return b->s;
}

/* UTF-8 -> code points (invalid bytes pass as themselves) */
static int utf8_cps(const char* s, uint32_t* out) {
  int n = 0;
  const unsigned char* p = (const unsigned char*)s;
  while (*p) {
    uint32_t c = *p;
    int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    if (len > 1) {
      int okk = 1;
      for (int k = 1; k < len; ++k) okk &= (p[k] & 0xC0) == 0x80;
      if (!okk) len = 1;
    }
    if (len == 1) c = *p;
    else if (len == 2) c = ((c & 31) << 6) | (p[1] & 63);
    else if (len == 3) c = ((c & 15) << 12) | ((p[1] & 63) << 6) | (p[2] & 63);
    else c = ((c & 7) << 18) | ((p[1] & 63) << 12) | ((p[2] & 63) << 6) | (p[3] & 63);
    out[n++] = c;
    p += len;
  }
  return n;
}
static void utf8_put(sbuf* b, uint32_t c) {
  char t[4];
  size_t n;
  if (c < 0x80) t[0] = (char)c, n = 1;
  else if (c < 0x800) t[0] = (char)(0xC0 | (c >> 6)), t[1] = (char)(0x80 | (c & 63)), n = 2;
  else if (c < 0x10000)
    t[0] = (char)(0xE0 | (c >> 12)), t[1] = (char)(0x80 | ((c >> 6) & 63)), t[2] = (char)(0x80 | (c & 63)), n = 3;
  else
    t[0] = (char)(0xF0 | (c >> 18)), t[1] = (char)(0x80 | ((c >> 12) & 63)), t[2] = (char)(0x80 | ((c >> 6) & 63)),
    t[3] = (char)(0x80 | (c & 63)), n = 4;
  sb_putn(b, t, n);
}

/* ================================================================== numbers in words (nft2.num_words) */

static const char* ONES[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                             "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                             "eighteen", "nineteen"};
static const char* TENS[] = {"twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};

static void num_words_n(long n, sbuf* b) {
  if (n < 20) {
    sb_put(b, ONES[n]);
    return;
  }
  if (n < 100) {
    sb_put(b, TENS[n / 10 - 2]);
    if (n % 10) {
      sb_put(b, " ");
      sb_put(b, ONES[n % 10]);
    }
    return;
  }
  if (n < 1000) {
    sb_put(b, ONES[n / 100]);
    sb_put(b, " hundred");
    if (n % 100) {
      sb_put(b, " ");
      num_words_n(n % 100, b);
    }
    return;
  }
  /* years: (n < 10000 and n % 1000 and 1100 <= n < 2000) or 2010 <= n < 2100 */
  if ((n < 10000 && n % 1000 && n >= 1100 && n < 2000) || (n >= 2010 && n < 2100)) {
    num_words_n(n / 100, b);
    sb_put(b, " ");
    int r = (int)(n % 100);
    if (r >= 10) num_words_n(r, b);
    else if (r) {
      sb_put(b, "oh ");
      sb_put(b, ONES[r]);
    } else
      sb_put(b, "hundred");
    return;
  }
  if (n < 1000000) {
    num_words_n(n / 1000, b);
    sb_put(b, " thousand");
    if (n % 1000) {
      sb_put(b, " ");
      num_words_n(n % 1000, b);
    }
    return;
  }
  char d[32];
  snprintf(d, sizeof d, "%ld", n);
  for (int i = 0; d[i]; ++i) {
    if (i) sb_put(b, " ");
    sb_put(b, ONES[d[i] - '0']);
  }
}

/* digits: a run of ASCII digits (any length; int() drops leading zeros) */
static void num_words_digits(const char* d, size_t len, sbuf* b) {
  while (len > 1 && *d == '0') d++, len--;
  if (len <= 6) {
    long n = 0;
    for (size_t i = 0; i < len; ++i) n = n * 10 + (d[i] - '0');
    num_words_n(n, b);
    return;
  }
  for (size_t i = 0; i < len; ++i) {   /* >= 1,000,000: digit by digit */
    if (i) sb_put(b, " ");
    sb_put(b, ONES[d[i] - '0']);
  }
}

char* arc_nft2_num_words(const char* digits) {
  sbuf b = {0};
  num_words_digits(digits, strlen(digits), &b);
  return sb_take(&b);
}

/* Pieces.norm: lower case ('’' -> "'", '-' -> ' '), digit runs as words, then only [a-z' ] kept */
char* arc_nft2_norm(const char* text) {
  size_t L = strlen(text ? text : "");
  uint32_t* cp = arc_malloc(sizeof(uint32_t) * (L + 1));
  int n = utf8_cps(text ? text : "", cp);
  sbuf t = {0};
  sb_put(&t, "");
  for (int i = 0; i < n; ++i) {
    uint32_t c = cp[i];
    if (c >= 'A' && c <= 'Z') c += 32;
    if (c == 0x2019) c = '\'';
    if (c == '-') c = ' ';
    if (c == 0x130) {   /* 'İ'.lower() = 'i' + U+0307 (dropped below) */
      sb_put(&t, "i ");
      continue;
    }
    if (c == 0x212A) c = 'k';   /* Kelvin sign */
    if (c >= '0' && c <= '9') {
      int j = i;
      char dg[4096];
      size_t k = 0;
      while (j < n && cp[j] >= '0' && cp[j] <= '9') {
        if (k < sizeof dg - 1) dg[k++] = (char)cp[j];
        j++;
      }
      sb_put(&t, " ");
      num_words_digits(dg, k, &t);
      sb_put(&t, " ");
      i = j - 1;
      continue;
    }
    if ((c >= 'a' && c <= 'z') || c == '\'' || c == ' ') {
      char ch = (char)c;
      sb_putn(&t, &ch, 1);
    } else
      sb_put(&t, " ");
  }
  arc_free(cp);
  /* ' '.join(split()) */
  char* s = t.s;
  sbuf o = {0};
  sb_put(&o, "");
  const char* p = s;
  while (*p) {
    while (*p == ' ') p++;
    if (!*p) break;
    const char* q = p;
    while (*q && *q != ' ') q++;
    if (o.n) sb_put(&o, " ");
    sb_putn(&o, p, (size_t)(q - p));
    p = q;
  }
  arc_free(s);
  return sb_take(&o);
}

/* _wkey: re.sub("[^\w']", '', lower) (non-ASCII letters kept, general punctuation U+2010..U+206F dropped), a final
 * "'s" off, "'" stripped, digit runs as words without spaces. As code points into out (returns the count). */
static int wkey_cps(const char* w, uint32_t* out) {
  size_t L = strlen(w);
  uint32_t* cp = arc_malloc(sizeof(uint32_t) * (L + 1));
  int n = utf8_cps(w, cp), k = 0;
  for (int i = 0; i < n; ++i) {
    uint32_t c = cp[i];
    if (c >= 'A' && c <= 'Z') c += 32;
    if (c < 0x80) {
      if (isalnum((int)c) || c == '_' || c == '\'') cp[k++] = c;
    } else if (!(c >= 0x2010 && c <= 0x206F))
      cp[k++] = c;
  }
  if (k >= 2 && cp[k - 2] == '\'' && cp[k - 1] == 's') k -= 2;
  int a = 0;
  while (a < k && cp[a] == '\'') a++;
  while (k > a && cp[k - 1] == '\'') k--;
  int m = 0;
  for (int i = a; i < k; ++i) {
    if (cp[i] >= '0' && cp[i] <= '9') {
      char dg[4096];
      size_t q = 0;
      int j = i;
      while (j < k && cp[j] >= '0' && cp[j] <= '9') {
        if (q < sizeof dg - 1) dg[q++] = (char)cp[j];
        j++;
      }
      sbuf b = {0};
      num_words_digits(dg, q, &b);
      for (size_t z = 0; z < b.n; ++z)
        if (b.s[z] != ' ') out[m++] = (uint32_t)(unsigned char)b.s[z];
      arc_free(b.s);
      i = j - 1;
    } else
      out[m++] = cp[i];
  }
  arc_free(cp);
  return m;
}
/* room a key of w needs (digits expand) */
static size_t wkey_cap(const char* w) { return strlen(w) * 64 + 8; }

char* arc_nft2_wkey(const char* w) {
  uint32_t* k = arc_malloc(sizeof(uint32_t) * wkey_cap(w));
  int n = wkey_cps(w, k);
  sbuf b = {0};
  sb_put(&b, "");
  for (int i = 0; i < n; ++i) utf8_put(&b, k[i]);
  arc_free(k);
  return sb_take(&b);
}

/* tagdec._key / case_from's key: re.sub("[^\w']", '', lower) */
static char* lkey(const char* w) {
  size_t L = strlen(w);
  uint32_t* cp = arc_malloc(sizeof(uint32_t) * (L + 1));
  int n = utf8_cps(w, cp);
  sbuf b = {0};
  sb_put(&b, "");
  for (int i = 0; i < n; ++i) {
    uint32_t c = cp[i];
    if (c >= 'A' && c <= 'Z') c += 32;
    if (c < 0x80) {
      if (isalnum((int)c) || c == '_' || c == '\'') utf8_put(&b, c);
    } else if (!(c >= 0x2010 && c <= 0x206F))
      utf8_put(&b, c);
  }
  arc_free(cp);
  return sb_take(&b);
}

/* ================================================================== difflib.SequenceMatcher(autojunk=False) on ints */

typedef struct {
  int a, b, n;
} smblock;

static void sm_longest(const uint32_t* A, const uint32_t* B, int alo, int ahi, int blo, int bhi, int* j2len,
                       int* nj2, int* bi, int* bj, int* bk) {
  int besti = alo, bestj = blo, best = 0, w = bhi - blo;
  memset(j2len, 0, sizeof(int) * (size_t)(w + 1));
  for (int i = alo; i < ahi; ++i) {
    memset(nj2, 0, sizeof(int) * (size_t)(w + 1));
    for (int j = blo; j < bhi; ++j) {
      if (A[i] != B[j]) continue;
      int k = (j > blo ? j2len[j - 1 - blo] : 0) + 1;
      nj2[j - blo] = k;
      if (k > best) {
        besti = i - k + 1;
        bestj = j - k + 1;
        best = k;
      }
    }
    int* t = j2len;
    j2len = nj2;
    nj2 = t;
  }
  *bi = besti;
  *bj = bestj;
  *bk = best;
}

static int cmp_smblock(const void* x, const void* y) {
  const smblock *a = x, *b = y;
  if (a->a != b->a) return a->a < b->a ? -1 : 1;
  if (a->b != b->b) return a->b < b->b ? -1 : 1;
  return (a->n > b->n) - (a->n < b->n);
}

/* get_matching_blocks(): blocks (collapsed) + the (la, lb, 0) sentinel; *nb its count. arc_free the result. */
static smblock* sm_blocks(const uint32_t* A, int la, const uint32_t* B, int lb, int* nb) {
  smblock* bl = arc_malloc(sizeof(smblock) * (size_t)(la + lb + 2));
  int(*q)[4] = arc_malloc(sizeof(int[4]) * (size_t)(la + lb + 2));
  int* j2 = arc_malloc(sizeof(int) * (size_t)(lb + 2));
  int* nj = arc_malloc(sizeof(int) * (size_t)(lb + 2));
  int n = 0, nq = 1;
  q[0][0] = 0, q[0][1] = la, q[0][2] = 0, q[0][3] = lb;
  while (nq) {
    nq--;
    int alo = q[nq][0], ahi = q[nq][1], blo = q[nq][2], bhi = q[nq][3];
    int i, j, k;
    sm_longest(A, B, alo, ahi, blo, bhi, j2, nj, &i, &j, &k);
    if (k) {
      bl[n++] = (smblock){i, j, k};
      if (alo < i && blo < j) {
        q[nq][0] = alo, q[nq][1] = i, q[nq][2] = blo, q[nq][3] = j;
        nq++;
      }
      if (i + k < ahi && j + k < bhi) {
        q[nq][0] = i + k, q[nq][1] = ahi, q[nq][2] = j + k, q[nq][3] = bhi;
        nq++;
      }
    }
  }
  qsort(bl, (size_t)n, sizeof(smblock), cmp_smblock);
  int m = 0;   /* collapse adjacent */
  for (int x = 0; x < n; ++x) {
    if (m && bl[m - 1].a + bl[m - 1].n == bl[x].a && bl[m - 1].b + bl[m - 1].n == bl[x].b) bl[m - 1].n += bl[x].n;
    else bl[m++] = bl[x];
  }
  bl[m++] = (smblock){la, lb, 0};
  arc_free(q);
  arc_free(j2);
  arc_free(nj);
  *nb = m;
  return bl;
}

static double sm_ratio(const uint32_t* A, int la, const uint32_t* B, int lb) {
  if (la + lb == 0) return 1.0;
  int nb;
  smblock* bl = sm_blocks(A, la, B, lb, &nb);
  int M = 0;
  for (int i = 0; i < nb; ++i) M += bl[i].n;
  arc_free(bl);
  return 2.0 * M / (la + lb);
}

double arc_sm_ratio_bytes(const char* a, const char* b) {
  int la = (int)strlen(a), lb = (int)strlen(b);
  uint32_t* A = arc_malloc(sizeof(uint32_t) * (size_t)(la + 1));
  uint32_t* B = arc_malloc(sizeof(uint32_t) * (size_t)(lb + 1));
  for (int i = 0; i < la; ++i) A[i] = (unsigned char)a[i];
  for (int i = 0; i < lb; ++i) B[i] = (unsigned char)b[i];
  double r = sm_ratio(A, la, B, lb);
  arc_free(A);
  arc_free(B);
  return r;
}

/* ================================================================== the model */

typedef struct {
  char* name;
  char* kind;
  char** aliases;
  int n_aliases;
} kb_ent;

/* a key (tuple of word keys) of the names list: its code points with 0 between words */
typedef struct {
  uint32_t* k;
  int n, nw, chars;   /* code points incl. separators, words, chars without separators */
  const char* tag;    /* ENTITY_... */
} list_key;

typedef struct {
  /* BoostTrie node */
  int* tok;        /* children tokens */
  int* child;      /* children nodes */
  int nc, cap;
  float token_score, node_score;
  int is_end, fail;
  float* vec;      /* scores() cache [V] */
} trie_node;

typedef struct {
  trie_node* nd;
  int n, cap, V;
} boost_trie;

struct arc_nft2 {
  arc_engine* e;         /* the Arc layer: decoder, tokenizer, trunk, heads, linking */
  arc_nft2_be* be;
  arc_nft2_dims d;
  int tap;
  /* front end */
  float window[NFFT];    /* the 400-sample window, centred in 512 */
  float* fb;             /* [n_mels, 257] */
  arc_fft* fft;
  /* RNNT vocabulary (Pieces) */
  char** piece;          /* "▁" as \x01 */
  double* score;
  int n_piece, maxlen;
  int* hidx;             /* open-addressing table of pieces */
  int hcap;
  /* Arc adapters */
  arc_store* ad;
  const float *mem_ln_w, *mem_ln_b, *mem_fc1_w, *mem_fc1_b, *mem_fc2_w, *mem_fc2_b, *mem_out_w, *mem_out_b;
  const float *arc_ln_w, *arc_ln_b, *arc_proj_w, *arc_proj_b, *arc_mlp_ln_w, *arc_mlp_ln_b, *arc_mlp1_w, *arc_mlp1_b,
      *arc_mlp3_w, *arc_mlp3_b;
  const float *tap_ln_w, *tap_ln_b, *tap_proj_w, *tap_proj_b;
  int pos_off, d_arc, d_tap;
  /* settings (transcription.nft2, backbone) */
  double alpha, context_score, whole_max_s, edge_pad_s, tail_pad_s, list_add_ratio, chunk_s, context_s;
  int case_from, char_align, unify, list_on, list_add, list_min_chars, list_add_long_chars, names_in_prompt;
  /* read-outs (transcription.nft2 guard / guard_cfg / nbest / almost) */
  int guard_on, nbest, cur_nbest, almost_on;
  char** common;   /* common_build */
  int n_common;
  void* tagger;    /* arctag_t* (ARC_WITH_ARCTAG) */
  arc_guard_cfg guard_cfg;
  arc_almost_cfg almost_cfg;
  int ctx_max, ctx_prev_max, ctx_kb_max, ctx_ses_max, ctx_types;
  char* ctx_default_dom;
  /* names */
  kb_ent* kb;
  int n_kb;
  list_key* lk;
  int n_lk;
  boost_trie* trie;
  /* context */
  int has_ctx;
  char* c_dom;
  int c_dom_set;
  char* c_prev;
  int auto_prev;
  char** c_kb;
  char** c_kind;
  int c_nkb;
  int* ses_mark;   /* session entries: marker, text */
  char** ses_text;
  int n_ses;
  /* the last encoder pass (OnnxEncoder memo) */
  uint64_t memo_key;
  int memo_n, memo_T;
  float *memo_H, *memo_Ht;
  /* dump */
  arc_nft2_dump_fn dump;
  void* dump_user;
  int dump_enc, dump_utt;
};

void arc_nft2_set_dump(arc_nft2* m, arc_nft2_dump_fn fn, void* user) {
  m->dump = fn;
  m->dump_user = user;
}
static void dumpf(arc_nft2* m, const char* name, int idx, const float* f, int rows, int cols) {
  if (m->dump) m->dump(m->dump_user, name, idx, f, NULL, rows, cols);
}
static void dumpi(arc_nft2* m, const char* name, int idx, const int* v, int n) {
  if (m->dump) m->dump(m->dump_user, name, idx, NULL, v, n, 1);
}

const char* arc_nft2_backend(void) { return arc_nft2_be_name(); }

/* ------------------------------------------------------------------ files */

/* a .npy array of float32 inside an uncompressed .npz (np.savez): data copied to out (cap floats); count or -1 */
static int npz_f32(const char* path, const char* member, float* out, int cap) {
  char* z = NULL;
  size_t n = 0;
  if (arc_read_file(path, &z, &n)) return -1;
  const unsigned char* p = (const unsigned char*)z;
  size_t i = 0;
  int rc = -1;
  while (i + 30 <= n && p[i] == 0x50 && p[i + 1] == 0x4b && p[i + 2] == 3 && p[i + 3] == 4) {
    int method = p[i + 8] | p[i + 9] << 8;
    size_t csize = (size_t)p[i + 18] | (size_t)p[i + 19] << 8 | (size_t)p[i + 20] << 16 | (size_t)p[i + 21] << 24;
    size_t nl = (size_t)(p[i + 26] | p[i + 27] << 8), xl = (size_t)(p[i + 28] | p[i + 29] << 8);
    const char* nm = (const char*)p + i + 30;
    size_t data = i + 30 + nl + xl;
    if (data + csize > n) break;
    if (nl == strlen(member) && !memcmp(nm, member, nl)) {
      if (method != 0) {
        arc_seterr("%s: %s is compressed (np.savez expected)", path, member);
        break;
      }
      const unsigned char* a = p + data;
      if (csize < 10 || memcmp(a + 1, "NUMPY", 5)) break;
      size_t hl = a[6] == 1 ? (size_t)(a[8] | a[9] << 8) : (size_t)(a[8] | a[9] << 8 | a[10] << 16 | a[11] << 24);
      size_t h0 = a[6] == 1 ? 10 : 12;
      char hdr[512];
      snprintf(hdr, sizeof hdr, "%.*s", (int)(hl < 511 ? hl : 511), (const char*)a + h0);
      if (!strstr(hdr, "'<f4'") || strstr(hdr, "'fortran_order': True")) {
        arc_seterr("%s: %s is not little-endian float32 in C order", path, member);
        break;
      }
      size_t nb = csize - h0 - hl;
      int cnt = (int)(nb / 4);
      if (cnt > cap) break;
      memcpy(out, a + h0 + hl, (size_t)cnt * 4);
      rc = cnt;
      break;
    }
    i = data + csize;
  }
  if (rc < 0 && !arc_geterr()[0]) arc_seterr("%s: no %s", path, member);
  arc_free(z);
  return rc;
}

/* nft2._spm_scores: piece scores from a sentencepiece ModelProto (field 1: SentencePiece {1: piece, 2: score}) */
static int spm_scores(const unsigned char* b, size_t n, double* out, int cap) {
  size_t i = 0;
  int k = 0;
#define VARINT(buf, len, pos, v)                       \
  do {                                                 \
    v = 0;                                             \
    int s_ = 0;                                        \
    while (pos < len) {                                \
      unsigned c_ = buf[pos++];                        \
      v |= (uint64_t)(c_ & 127) << s_;                 \
      s_ += 7;                                         \
      if (c_ < 128) break;                             \
    }                                                  \
  } while (0)
  while (i < n) {
    uint64_t key, v;
    VARINT(b, n, i, key);
    int f = (int)(key >> 3), w = (int)(key & 7);
    if (w == 0) VARINT(b, n, i, v);
    else if (w == 1) i += 8;
    else if (w == 5) i += 4;
    else if (w == 2) {
      uint64_t len;
      VARINT(b, n, i, len);
      if (f == 1) {
        const unsigned char* s = b + i;
        size_t j = 0;
        float sc = 0;
        while (j < len) {
          uint64_t k2, v2;
          VARINT(s, len, j, k2);
          int f2 = (int)(k2 >> 3), w2 = (int)(k2 & 7);
          if (w2 == 5) {
            if (f2 == 2) memcpy(&sc, s + j, 4);
            j += 4;
          } else if (w2 == 0)
            VARINT(s, len, j, v2);
          else if (w2 == 2) {
            uint64_t l2;
            VARINT(s, len, j, l2);
            j += l2;
          } else if (w2 == 1)
            j += 8;
        }
        if (k < cap) out[k] = sc;
        k++;
      }
      i += len;
    } else
      break;
  }
#undef VARINT
  return k;
}

static uint64_t fnv(const char* s, size_t n) {
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n; ++i) h = (h ^ (unsigned char)s[i]) * 1099511628211ULL;
  return h;
}

static int piece_id(const arc_nft2* m, const char* s, size_t n) {
  size_t h = (size_t)(fnv(s, n) & (uint64_t)(m->hcap - 1));
  while (m->hidx[h] >= 0) {
    const char* p = m->piece[m->hidx[h]];
    if (strlen(p) == n && !memcmp(p, s, n)) return m->hidx[h];
    h = (h + 1) & (size_t)(m->hcap - 1);
  }
  return -1;
}

/* "▁" (U+2581) as \x01: pieces and the text are then one byte per code point */
static char* to_meta(const char* s) {
  size_t n = strlen(s);
  char* o = arc_malloc(n + 1);
  size_t k = 0;
  for (size_t i = 0; i < n; ++i) {
    if ((unsigned char)s[i] == 0xE2 && i + 2 < n && (unsigned char)s[i + 1] == 0x96 && (unsigned char)s[i + 2] == 0x81) {
      o[k++] = 1;
      i += 2;
    } else
      o[k++] = s[i];
  }
  o[k] = 0;
  return o;
}

static int load_pieces(arc_nft2* m, const char* od) {
  char path[1300];
  snprintf(path, sizeof path, "%s/tokens.txt", od);
  char* t = NULL;
  size_t n = 0;
  if (arc_read_file(path, &t, &n)) return -1;
  int V = m->d.vocab;
  m->piece = arc_calloc(sizeof(char*) * (size_t)(V + 1));
  m->score = arc_calloc(sizeof(double) * (size_t)(V + 1));
  int k = 0;
  char* p = t;
  while (*p && k < V) {
    char* e = strchr(p, '\n');
    if (e) *e = 0;
    size_t l = strlen(p);
    if (l && p[l - 1] == '\r') p[--l] = 0;
    char* sp = strrchr(p, ' ');   /* rsplit(' ', 1)[0] */
    if (sp) *sp = 0;
    m->piece[k++] = to_meta(p);
    if (!e) break;
    p = e + 1;
  }
  arc_free(t);
  m->n_piece = k;
  snprintf(path, sizeof path, "%s/tokenizer.model", od);
  char* sp = NULL;
  size_t sn = 0;
  if (!arc_read_file(path, &sp, &sn)) {
    spm_scores((const unsigned char*)sp, sn, m->score, V);
    arc_free(sp);
  }
  m->maxlen = 0;
  m->hcap = 1;
  while (m->hcap < 4 * k) m->hcap <<= 1;
  m->hidx = arc_malloc(sizeof(int) * (size_t)m->hcap);
  for (int i = 0; i < m->hcap; ++i) m->hidx[i] = -1;
  for (int i = 0; i < k; ++i) {
    size_t l = strlen(m->piece[i]);
    if ((int)l > m->maxlen) m->maxlen = (int)l;
    size_t h = (size_t)(fnv(m->piece[i], l) & (uint64_t)(m->hcap - 1));
    int done = 0;
    while (m->hidx[h] >= 0) {   /* a later duplicate wins ({p: i for i, p in enumerate(...)}) */
      if (!strcmp(m->piece[m->hidx[h]], m->piece[i])) {
        m->hidx[h] = i;
        done = 1;
        break;
      }
      h = (h + 1) & (size_t)(m->hcap - 1);
    }
    if (!done) m->hidx[h] = i;
  }
  return 0;
}

/* Pieces.encode without sentencepiece: Viterbi unigram over the piece scores (unknown characters: id 0, -20) */
static int pieces_encode(const arc_nft2* m, const char* text, int** out) {
  char* t = arc_nft2_norm(text);
  *out = NULL;
  if (!t[0]) {
    arc_free(t);
    return 0;
  }
  size_t tl = strlen(t);
  char* s = arc_malloc(tl + 2);
  s[0] = 1;
  for (size_t i = 0; i < tl; ++i) s[i + 1] = t[i] == ' ' ? 1 : t[i];
  s[tl + 1] = 0;
  arc_free(t);
  int n = (int)tl + 1;
  double* bv = arc_malloc(sizeof(double) * (size_t)(n + 1));
  int *bi = arc_malloc(sizeof(int) * (size_t)(n + 1)), *bk = arc_malloc(sizeof(int) * (size_t)(n + 1));
  for (int i = 0; i <= n; ++i) bv[i] = -1e30, bi[i] = -1, bk[i] = -1;
  bv[0] = 0.0;
  for (int i = 0; i < n; ++i) {
    if (bv[i] <= -1e29) continue;
    int lim = i + m->maxlen < n ? i + m->maxlen : n;
    for (int j = i + 1; j <= lim; ++j) {
      int k = piece_id(m, s + i, (size_t)(j - i));
      if (k > 0) {
        double v = bv[i] + m->score[k];
        if (v > bv[j]) bv[j] = v, bi[j] = i, bk[j] = k;
      }
    }
    if (bv[i + 1] <= -1e29) bv[i + 1] = bv[i] - 20.0, bi[i + 1] = i, bk[i + 1] = 0;
  }
  int* o = arc_malloc(sizeof(int) * (size_t)(n + 1));
  int no = 0;
  for (int j = n; j > 0;) {
    int k = bk[j];
    if (!(k == 0 && no && o[no - 1] == 0)) o[no++] = k;
    j = bi[j];
  }
  for (int a = 0, b = no - 1; a < b; ++a, --b) {
    int x = o[a];
    o[a] = o[b];
    o[b] = x;
  }
  arc_free(s);
  arc_free(bv);
  arc_free(bi);
  arc_free(bk);
  *out = o;
  return no;
}

/* ------------------------------------------------------------------ BoostTrie */

static int trie_new_node(boost_trie* t, float ts, float ns, int end) {
  if (t->n == t->cap) {
    int nc = t->cap ? t->cap * 2 : 64;
    trie_node* nn = arc_calloc(sizeof(trie_node) * (size_t)nc);
    if (t->n) memcpy(nn, t->nd, sizeof(trie_node) * (size_t)t->n);
    arc_free(t->nd);
    t->nd = nn;
    t->cap = nc;
  }
  trie_node* d = &t->nd[t->n];
  memset(d, 0, sizeof *d);
  d->token_score = ts;
  d->node_score = ns;
  d->is_end = end;
  d->fail = -1;
  return t->n++;
}
static int trie_child(const boost_trie* t, int s, int tok) {
  const trie_node* d = &t->nd[s];
  for (int i = 0; i < d->nc; ++i)
    if (d->tok[i] == tok) return d->child[i];
  return -1;
}
static void trie_add_child(boost_trie* t, int s, int tok, int c) {
  trie_node* d = &t->nd[s];
  if (d->nc == d->cap) {
    int nc = d->cap ? d->cap * 2 : 4;
    int* a = arc_malloc(sizeof(int) * (size_t)nc);
    int* b = arc_malloc(sizeof(int) * (size_t)nc);
    if (d->nc) {
      memcpy(a, d->tok, sizeof(int) * (size_t)d->nc);
      memcpy(b, d->child, sizeof(int) * (size_t)d->nc);
    }
    arc_free(d->tok);
    arc_free(d->child);
    d->tok = a;
    d->child = b;
    d->cap = nc;
  }
  d->tok[d->nc] = tok;
  d->child[d->nc++] = c;
}

static void trie_free(boost_trie* t) {
  if (!t) return;
  for (int i = 0; i < t->n; ++i) {
    arc_free(t->nd[i].tok);
    arc_free(t->nd[i].child);
    arc_free(t->nd[i].vec);
  }
  arc_free(t->nd);
  arc_free(t);
}

/* BoostTrie(token_lists, vocab, context_score, depth_scaling=2.0) */
static boost_trie* trie_build(int* const* lists, const int* lens, int nl, int V, double cs) {
  const double ds = 2.0;
  boost_trie* t = arc_calloc(sizeof *t);
  t->V = V;
  trie_new_node(t, 0, 0, 0);
  for (int l = 0; l < nl; ++l) {
    int node = 0;
    for (int i = 0; i < lens[l]; ++i) {
      int tok = lists[l][i], end = i == lens[l] - 1;
      int c = trie_child(t, node, tok);
      if (c < 0) {
        double ts = i > 0 ? cs * ds + log((double)(i + 1)) : cs;
        int nn = trie_new_node(t, (float)ts, (float)((double)t->nd[node].node_score + ts), end);
        trie_add_child(t, node, tok, nn);
        c = nn;
      } else {
        trie_node* d = &t->nd[c];
        d->token_score = (float)fmax(cs, (double)d->token_score);
        d->node_score = (float)((double)t->nd[node].node_score + (double)d->token_score);
        d->is_end = d->is_end || end;
      }
      node = c;
    }
  }
  /* fail links, breadth first in insertion order */
  int* q = arc_malloc(sizeof(int) * (size_t)(t->n + 1));
  int qh = 0, qt = 0;
  for (int i = 0; i < t->nd[0].nc; ++i) {
    t->nd[t->nd[0].child[i]].fail = 0;
    q[qt++] = t->nd[0].child[i];
  }
  while (qh < qt) {
    int cur = q[qh++];
    for (int i = 0; i < t->nd[cur].nc; ++i) {
      int tok = t->nd[cur].tok[i], c = t->nd[cur].child[i];
      int f = t->nd[cur].fail;
      while (f != 0 && trie_child(t, f, tok) < 0) f = t->nd[f].fail;
      int fc = trie_child(t, f, tok);
      t->nd[c].fail = fc >= 0 && fc != c ? fc : 0;
      q[qt++] = c;
    }
  }
  arc_free(q);
  return t;
}

/* node_score / token_score are Python floats: kept here as the float32 they round to only for the vector; the
 * arithmetic below is double, as in Python */
static const float* trie_scores(boost_trie* t, int s) {
  trie_node* d = &t->nd[s];
  if (d->vec) return d->vec;
  float* v = arc_malloc(sizeof(float) * (size_t)t->V);
  unsigned char* set = arc_calloc((size_t)t->V);
  double acc = 0.0;
  int node = s;
  for (;;) {
    trie_node* nd = &t->nd[node];
    for (int i = 0; i < nd->nc; ++i) {
      int tok = nd->tok[i];
      if (tok >= 0 && tok < t->V && !set[tok]) {
        v[tok] = (float)(acc + (double)t->nd[nd->child[i]].token_score);
        set[tok] = 1;
      }
    }
    if (node == 0) break;
    acc += nd->is_end ? 0.0 : (double)t->nd[nd->fail].node_score - (double)nd->node_score;
    node = nd->fail;
  }
  for (int k = 0; k < t->V; ++k)
    if (!set[k]) v[k] = (float)acc;
  arc_free(set);
  d->vec = v;
  return v;
}

static int trie_advance(const boost_trie* t, int s, int tok) {
  for (;;) {
    int c = trie_child(t, s, tok);
    if (c >= 0) return c;
    if (s == 0) return 0;
    s = t->nd[s].fail;
  }
}

/* ------------------------------------------------------------------ load */

static double cfg_num(const arc_json* o, const char* k, double d) { return arc_json_num(arc_json_get(o, k), d); }

void arc_nft2_destroy(arc_nft2* m) {
  if (!m) return;
  arc_nft2_set_tagger(m, NULL, NULL, 0, NULL, 0);
  for (int i = 0; i < m->n_common; ++i) arc_free(m->common[i]);
  arc_free(m->common);
  arc_nft2_set_names(m, NULL, 0);
  arc_nft2_set_context(m, NULL);
  arc_nft2_be_free(m->be);
  arc_destroy(m->e);
  arc_store_close(m->ad);
  arc_fft_destroy(m->fft);
  arc_free(m->fb);
  for (int i = 0; i < m->n_piece; ++i) arc_free(m->piece[i]);
  arc_free(m->piece);
  arc_free(m->score);
  arc_free(m->hidx);
  arc_free(m->memo_H);
  arc_free(m->memo_Ht);
  arc_free(m->ctx_default_dom);
  arc_free(m);
}

arc_nft2* arc_nft2_create(const char* dir, int threads, char* err, size_t err_len) {
  arc_seterr("%s", "");
  arc_nft2* m = arc_calloc(sizeof *m);
  if (!m) goto fail;
  if (threads <= 0) threads = 4;
  if (!(m->e = arc_engine_create_nft2(dir, threads))) goto fail;
  const arc_json* cfg = m->e->cfg;
  const arc_json* bc = arc_json_get(cfg, "backbone");
  if (strcmp(arc_json_str(arc_json_get(bc, "type"), ""), "nft2")) {
    arc_seterr("%s: not an NFT2 model (config backbone.type)", dir);
    goto fail;
  }
  m->tap = (int)cfg_num(bc, "tap", 9);
  m->pos_off = (int)cfg_num(bc, "pos_off", 2);
  m->chunk_s = cfg_num(bc, "chunk_s", 30.0);
  m->context_s = cfg_num(bc, "context_s", 4.0);
  char od[1200], path[1400];
  const char* sub = arc_json_str(arc_json_get(bc, "onnx_dir"), "encoder_onnx");
  snprintf(od, sizeof od, "%s/%s", dir, sub);
  /* encoder_onnx/config.json */
  snprintf(path, sizeof path, "%s/config.json", od);
  char* t = NULL;
  size_t tn = 0;
  if (arc_read_file(path, &t, &tn)) goto fail;
  arc_json* oc = arc_json_parse(t, tn);
  arc_free(t);
  if (!oc) goto fail;
  arc_nft2_dims* d = &m->d;
  d->n_layers = (int)cfg_num(oc, "n_layers", 17);
  d->d_model = (int)cfg_num(oc, "d_model", 512);
  d->n_mels = (int)cfg_num(oc, "n_mels", 80);
  d->cache_ch = (int)cfg_num(oc, "last_channel_cache_size", 70);
  d->conv_cache = (int)cfg_num(oc, "conv_cache", 8);
  d->chunk = (int)arc_json_num(arc_json_at(arc_json_get(oc, "shift_size_frames"), 1), 56);
  d->pre = (int)arc_json_num(arc_json_at(arc_json_get(oc, "pre_encode_cache_size"), 1), 9);
  d->first = arc_json_get(oc, "first_chunk_graph") ? (int)cfg_num(oc, "first_chunk_frames", 0) : 0;
  d->pred_layers = (int)cfg_num(oc, "pred_rnn_layers", 1);
  d->pred_hidden = (int)cfg_num(oc, "pred_hidden", 640);
  d->joint_hidden = (int)cfg_num(oc, "joint_hidden", 640);
  d->vocab = (int)cfg_num(oc, "vocab_size", 1024);
  d->blank = (int)cfg_num(oc, "blank_id", d->vocab);
  d->max_sym = (int)cfg_num(oc, "max_symbols_per_frame", 10);
  int bad = d->n_mels != 80 || cfg_num(oc, "n_fft", 512) != NFFT || cfg_num(oc, "hop_length", HOP) != HOP ||
            cfg_num(oc, "win_length", 400) != 400 || strcmp(arc_json_str(arc_json_get(oc, "normalize"), "NA"), "NA") ||
            d->blank != d->vocab;
  arc_json_free(oc);
  if (bad) {
    arc_seterr("%s: unsupported front end / vocabulary settings", path);
    goto fail;
  }
  /* front end */
  snprintf(path, sizeof path, "%s/frontend.npz", od);
  float win[400];
  m->fb = arc_malloc(sizeof(float) * 80 * 257);
  if (npz_f32(path, "window.npy", win, 400) != 400 || npz_f32(path, "fb.npy", m->fb, 80 * 257) != 80 * 257) goto fail;
  memset(m->window, 0, sizeof m->window);
  memcpy(m->window + (NFFT - 400) / 2, win, sizeof win);
  m->fft = arc_fft_create(NFFT);
  if (load_pieces(m, od)) goto fail;
  /* adapters */
  snprintf(path, sizeof path, "%s/%s", dir, arc_json_str(arc_json_get(bc, "adapters"), "adapters.safetensors"));
  if (!(m->ad = arc_store_open(path))) goto fail;
  int D = d->d_model;
  int64_t sh[8];
  m->d_arc = arc_store_shape(m->ad, "adapter.arc.proj.weight", sh) == 2 ? (int)sh[0] : 0;
  m->d_tap = arc_store_shape(m->ad, "adapter.tap.proj.weight", sh) == 2 ? (int)sh[0] : 0;
#define A(f, nm, n) if (!(m->f = arc_store_f32(m->ad, nm, n))) goto fail
  A(mem_ln_w, "adapter.mem.ln.weight", D);
  A(mem_ln_b, "adapter.mem.ln.bias", D);
  A(mem_fc1_w, "adapter.mem.fc1.weight", 1024 * D);
  A(mem_fc1_b, "adapter.mem.fc1.bias", 1024);
  A(mem_fc2_w, "adapter.mem.fc2.weight", ARC_MOON_D * 1024);
  A(mem_fc2_b, "adapter.mem.fc2.bias", ARC_MOON_D);
  A(mem_out_w, "adapter.mem.out.weight", ARC_MOON_D);
  A(mem_out_b, "adapter.mem.out.bias", ARC_MOON_D);
  A(arc_ln_w, "adapter.arc.ln.weight", D);
  A(arc_ln_b, "adapter.arc.ln.bias", D);
  A(arc_proj_w, "adapter.arc.proj.weight", (int64_t)m->d_arc * D);
  A(arc_proj_b, "adapter.arc.proj.bias", m->d_arc);
  A(arc_mlp_ln_w, "adapter.arc.mlp.0.weight", 2 * m->d_arc);
  A(arc_mlp_ln_b, "adapter.arc.mlp.0.bias", 2 * m->d_arc);
  A(arc_mlp1_w, "adapter.arc.mlp.1.weight", 1024 * 2 * m->d_arc);
  A(arc_mlp1_b, "adapter.arc.mlp.1.bias", 1024);
  A(arc_mlp3_w, "adapter.arc.mlp.3.weight", DIN * 1024);
  A(arc_mlp3_b, "adapter.arc.mlp.3.bias", DIN);
  A(tap_ln_w, "adapter.tap.ln.weight", D);
  A(tap_ln_b, "adapter.tap.ln.bias", D);
  A(tap_proj_w, "adapter.tap.proj.weight", (int64_t)m->d_tap * D);
  A(tap_proj_b, "adapter.tap.proj.bias", m->d_tap);
#undef A
  if (2 * m->d_tap != DIN) {
    arc_seterr("%s: the tap adapter must give %d pooled features", path, DIN);
    goto fail;
  }
  /* transcription.nft2 (Nft2Finals defaults) and the context budget */
  const arc_json* tr = arc_json_get(cfg, "transcription");
  const arc_json* nc = arc_json_get(tr, "nft2");
  m->alpha = cfg_num(nc, "alpha", 2.0);
  m->context_score = cfg_num(nc, "context_score", 1.0);
  m->case_from = (int)cfg_num(nc, "case_from_decoder", 1);
  m->whole_max_s = cfg_num(nc, "whole_max_s", 20.0);
  m->edge_pad_s = cfg_num(nc, "edge_pad_s", 0.15);
  m->tail_pad_s = cfg_num(nc, "tail_pad_s", 0.5);
  m->char_align = (int)cfg_num(nc, "char_align", 1);
  m->unify = (int)cfg_num(nc, "unify_types", 1);
  m->list_on = (int)cfg_num(nc, "list_spans", 1);
  m->list_add = (int)cfg_num(nc, "list_add", 1);
  m->list_min_chars = (int)cfg_num(nc, "list_min_chars", 3);
  m->list_add_ratio = cfg_num(nc, "list_add_ratio", 0.5);
  m->list_add_long_chars = (int)cfg_num(nc, "list_add_long_chars", 0);   /* opt-in, see nft2.list_spans */
  m->names_in_prompt = (int)cfg_num(nc, "names_in_prompt", 1);
  m->guard_on = (int)cfg_num(nc, "guard", 1);
  m->nbest = (int)cfg_num(nc, "nbest", 0);
  m->almost_on = (int)cfg_num(nc, "almost", 0);
  arc_guard_defaults(&m->guard_cfg);
  {
    const arc_json* gc = arc_json_get(nc, "guard_cfg");
    arc_guard_cfg* g = &m->guard_cfg;
    g->silence_db = cfg_num(gc, "silence_db", g->silence_db);
    g->flat_db = cfg_num(gc, "flat_db", g->flat_db);
    g->flat_conf = cfg_num(gc, "flat_conf", g->flat_conf);
    g->rep_conf = cfg_num(gc, "rep_conf", g->rep_conf);
    g->tail_s = cfg_num(gc, "tail_s", g->tail_s);
    g->tail_conf = cfg_num(gc, "tail_conf", g->tail_conf);
  }
  arc_almost_defaults(&m->almost_cfg);
  if (!m->char_align) {
    arc_seterr("%s: transcription.nft2.char_align false is not supported", dir);
    goto fail;
  }
  const arc_json* cb = arc_json_get(tr, "context_budget");
  if (cb && cfg_num(cb, "v", 1) != 2) {
    arc_seterr("%s: context_budget v%d is not supported (v2 expected)", dir, (int)cfg_num(cb, "v", 1));
    goto fail;
  }
  m->ctx_max = (int)cfg_num(cb, "max", 112);
  m->ctx_prev_max = (int)cfg_num(cb, "prev_max", 48);
  m->ctx_kb_max = (int)cfg_num(cb, "kb_max", 56);
  m->ctx_ses_max = (int)cfg_num(cb, "ses_max", 300);
  m->ctx_types = (int)cfg_num(cb, "types", 1);
  const char* dd = arc_json_str(arc_json_get(arc_json_get(tr, "context_default"), "dom"), NULL);
  m->ctx_default_dom = dd ? arc_strdup(dd) : NULL;
  /* the encoder and RNNT nets */
  if (!(m->be = arc_nft2_be_open(od, d, m->tap, threads))) goto fail;
  return m;
fail:
  if (err && err_len) snprintf(err, err_len, "%s", arc_geterr()[0] ? arc_geterr() : "arc_nft2_create failed");
  arc_nft2_destroy(m);
  return NULL;
}

/* ------------------------------------------------------------------ names and context */

static const char* const KIND_TAG[][2] = {
    {"lunar_feature", "ENTITY_LUNAR"}, {"site", "ENTITY_LUNAR"},       {"celestial", "ENTITY_CELESTIAL"},
    {"spacecraft", "ENTITY_SPACECRAFT"}, {"vehicle", "ENTITY_SPACECRAFT"}, {"mission", "ENTITY_MISSION"},
    {"person", "ENTITY_PERSON"},       {"crew", "ENTITY_PERSON"},      {"place", "ENTITY_PLACE"},
    {"team", "ENTITY_TEAM"},           {"organization", "ENTITY_ORG"}, {"org", "ENTITY_ORG"},
    {"product", "ENTITY_PRODUCT"},     {"law", "ENTITY_LAW"}};
static const char* kind_tag(const char* k) {
  for (size_t i = 0; k && i < sizeof KIND_TAG / sizeof *KIND_TAG; ++i)
    if (!strcmp(k, KIND_TAG[i][0])) return KIND_TAG[i][1];
  return NULL;
}
/* api.KB_KIND: the kb prompt's type of a list kind ('term' otherwise) */
static const char* kb_kind(const char* k) {
  static const char* const T[][2] = {{"person", "person"},   {"team", "team"},        {"organization", "org"},
                                     {"org", "org"},         {"place", "place"},      {"law", "law"},
                                     {"product", "product"}, {"lunar_feature", "place"}, {"site", "place"},
                                     {"celestial", "place"}, {"spacecraft", "product"}, {"vehicle", "product"},
                                     {"mission", "term"},    {"crew", "person"}};
  for (size_t i = 0; k && i < sizeof T / sizeof *T; ++i)
    if (!strcmp(k, T[i][0])) return T[i][1];
  return "term";
}

int arc_nft2_set_names(arc_nft2* m, const arc_name_entry* names, int n) {
  if (!m || n < 0 || (n && !names)) return -1;
  for (int i = 0; i < m->n_kb; ++i) {
    arc_free(m->kb[i].name);
    arc_free(m->kb[i].kind);
    for (int j = 0; j < m->kb[i].n_aliases; ++j) arc_free(m->kb[i].aliases[j]);
    arc_free(m->kb[i].aliases);
  }
  arc_free(m->kb);
  m->kb = NULL;
  m->n_kb = 0;
  for (int i = 0; i < m->n_lk; ++i) arc_free(m->lk[i].k);
  arc_free(m->lk);
  m->lk = NULL;
  m->n_lk = 0;
  trie_free(m->trie);
  m->trie = NULL;
  if (m->e && arc_engine_set_names(m->e, names, n)) return -1;
  if (!n) return 0;
  m->kb = arc_calloc(sizeof(kb_ent) * (size_t)n);
  m->n_kb = n;
  for (int i = 0; i < n; ++i) {
    m->kb[i].name = xstrdup(names[i].name);
    m->kb[i].kind = arc_strdup(names[i].kind ? names[i].kind : "other");
    int na = names[i].n_aliases > 0 ? names[i].n_aliases : 0;
    m->kb[i].aliases = arc_calloc(sizeof(char*) * (size_t)(na + 1));
    for (int j = 0; j < na; ++j) m->kb[i].aliases[j] = xstrdup(names[i].aliases[j]);
    m->kb[i].n_aliases = na;
  }
  /* list_keys: word-key tuples of every name / alias of a kind with a tag (first one wins) */
  int cap = 0;
  for (int i = 0; i < n; ++i) cap += 1 + m->kb[i].n_aliases;
  m->lk = arc_calloc(sizeof(list_key) * (size_t)(cap + 1));
  for (int i = 0; i < n; ++i) {
    const char* tg = kind_tag(m->kb[i].kind);
    if (!tg) continue;
    for (int a = -1; a < m->kb[i].n_aliases; ++a) {
      char* nm = arc_nft2_norm(a < 0 ? m->kb[i].name : m->kb[i].aliases[a]);
      char** w;
      int nw = arc_split_words(nm, &w);
      uint32_t* k = arc_malloc(sizeof(uint32_t) * (strlen(nm) * 64 + 8));
      int kn = 0, kw = 0, ch = 0;
      for (int j = 0; j < nw; ++j) {
        uint32_t* tmp = arc_malloc(sizeof(uint32_t) * wkey_cap(w[j]));
        int tn = wkey_cps(w[j], tmp);
        if (tn) {
          if (kw) k[kn++] = 0;
          memcpy(k + kn, tmp, sizeof(uint32_t) * (size_t)tn);
          kn += tn;
          kw++;
          ch += tn;
        }
        arc_free(tmp);
      }
      arc_free_words(w, nw);
      arc_free(nm);
      int dup = !kw;
      for (int q = 0; q < m->n_lk && !dup; ++q)
        dup = m->lk[q].n == kn && !memcmp(m->lk[q].k, k, sizeof(uint32_t) * (size_t)kn);
      if (dup) {
        arc_free(k);
        continue;
      }
      m->lk[m->n_lk++] = (list_key){k, kn, kw, ch, tg};
    }
  }
  /* the RNNT boosting trie over every name and alias */
  int** lists = arc_malloc(sizeof(int*) * (size_t)(cap + 1));
  int* lens = arc_malloc(sizeof(int) * (size_t)(cap + 1));
  int nl = 0;
  for (int i = 0; i < n; ++i)
    for (int a = -1; a < m->kb[i].n_aliases; ++a) {
      const char* ph = a < 0 ? m->kb[i].name : m->kb[i].aliases[a];
      int blank = 1;
      for (const char* c = ph; *c; ++c) blank &= isspace((unsigned char)*c) != 0;
      if (blank) continue;
      int* ids;
      int k = pieces_encode(m, ph, &ids);
      if (k > 0) {
        lists[nl] = ids;
        lens[nl++] = k;
      } else
        arc_free(ids);
    }
  if (nl) m->trie = trie_build(lists, lens, nl, m->d.blank, m->context_score);
  for (int i = 0; i < nl; ++i) arc_free(lists[i]);
  arc_free(lists);
  arc_free(lens);
  return 0;
}

static void free_ctx(arc_nft2* m) {
  arc_free(m->c_dom);
  arc_free(m->c_prev);
  for (int i = 0; i < m->c_nkb; ++i) {
    arc_free(m->c_kb[i]);
    arc_free(m->c_kind[i]);
  }
  arc_free(m->c_kb);
  arc_free(m->c_kind);
  for (int i = 0; i < m->n_ses; ++i) arc_free(m->ses_text[i]);
  arc_free(m->ses_text);
  arc_free(m->ses_mark);
  m->c_dom = m->c_prev = NULL;
  m->c_kb = m->c_kind = m->ses_text = NULL;
  m->ses_mark = NULL;
  m->c_nkb = m->n_ses = 0;
  m->c_dom_set = m->auto_prev = m->has_ctx = 0;
}

#define CTX_PREV 32640
#define CTX_KB 32641
#define CTX_DOM 32642
#define CTX_SEP 32644
#define CTX_SES 32653
static int type_marker(const char* t) {
  static const char* const T[] = {"person", "team", "org", "place", "law", "product", "term", "id"};
  for (int i = 0; t && i < 8; ++i)
    if (!strcmp(t, T[i])) return 32645 + i;
  return -1;
}
static char* under_to_space(const char* s) {
  char* o = xstrdup(s);
  for (char* p = o; *p; ++p)
    if (*p == '_') *p = ' ';
  return o;
}

int arc_nft2_set_context(arc_nft2* m, const arc_nft2_context* c) {
  if (!m) return -1;
  free_ctx(m);
  if (!c) return 0;
  m->has_ctx = 1;
  if (c->dom) {
    m->c_dom_set = 1;
    m->c_dom = arc_strdup(c->dom);
  }
  if (c->prev) m->c_prev = arc_strdup(c->prev);
  m->auto_prev = c->auto_prev;
  if (c->n_kb > 0) {
    m->c_kb = arc_calloc(sizeof(char*) * (size_t)c->n_kb);
    m->c_kind = arc_calloc(sizeof(char*) * (size_t)c->n_kb);
    for (int i = 0; i < c->n_kb; ++i) {
      m->c_kb[i] = xstrdup(c->kb[i]);
      m->c_kind[i] = arc_strdup(c->kb_kind && c->kb_kind[i] ? c->kb_kind[i] : "");
    }
    m->c_nkb = c->n_kb;
  }
  /* session_entries: speakers, numbers, entities in that order */
  int cap = (c->n_speakers > 0 ? c->n_speakers : 0) + (c->n_numbers > 0 ? c->n_numbers : 0) +
            (c->n_entities > 0 ? c->n_entities : 0);
  m->ses_mark = arc_calloc(sizeof(int) * (size_t)(cap + 1));
  m->ses_text = arc_calloc(sizeof(char*) * (size_t)(cap + 1));
  char buf[1024];
  for (int i = 0; i < c->n_speakers; ++i) {
    const char* slot = c->speakers[i].slot;
    if (!slot || !*slot) continue;
    char* role = c->speakers[i].role && *c->speakers[i].role ? under_to_space(c->speakers[i].role) : arc_strdup("");
    snprintf(buf, sizeof buf, "%s %s", slot, role);
    arc_free(role);
    size_t l = strlen(buf);   /* .strip() */
    while (l && buf[l - 1] == ' ') buf[--l] = 0;
    char* b = buf;
    while (*b == ' ') b++;
    m->ses_mark[m->n_ses] = CTX_SEP;
    m->ses_text[m->n_ses++] = arc_strdup(b);
  }
  for (int i = 0; i < c->n_numbers; ++i) {
    const char* v = c->numbers[i].value;
    if (!v || !*v) continue;
    char* kind = c->numbers[i].kind && *c->numbers[i].kind ? under_to_space(c->numbers[i].kind) : arc_strdup("number");
    char age[32] = "";
    if (c->numbers[i].minutes >= 0) snprintf(age, sizeof age, " %dm", c->numbers[i].minutes);
    snprintf(buf, sizeof buf, "%s %s%s", kind, v, age);
    arc_free(kind);
    m->ses_mark[m->n_ses] = CTX_SEP;
    m->ses_text[m->n_ses++] = arc_strdup(buf);
  }
  for (int i = 0; i < c->n_entities; ++i) {
    const char* s = c->entities[i].surface;
    if (!s || !*s) continue;
    const char* ty = c->entities[i].type ? c->entities[i].type : "";
    char age[32] = "";
    if (c->entities[i].minutes >= 0) snprintf(age, sizeof age, " %dm", c->entities[i].minutes);
    int mk = type_marker(ty);
    if (mk >= 0) snprintf(buf, sizeof buf, "%s%s", s, age);
    else {
      snprintf(buf, sizeof buf, "%s %s%s", ty, s, age);
    }
    char* b = buf;
    while (*b == ' ') b++;
    size_t l = strlen(b);
    while (l && b[l - 1] == ' ') b[--l] = 0;
    m->ses_mark[m->n_ses] = mk >= 0 ? mk : CTX_SEP;
    m->ses_text[m->n_ses++] = arc_strdup(b);
  }
  return 0;
}

/* Transcriber._prompt_ids with build_prompt_v2: the context (or the names list as kb) over context_default */
static int prompt_ids(arc_nft2* m, double t_now, int* out, int cap) {
  arc_engine* e = m->e;
  const char* dom = m->c_dom_set ? m->c_dom : m->ctx_default_dom;
  /* kb: the context's own, else the names list (names_in_prompt) */
  const char** kb = NULL;
  const char** ty = NULL;
  int nkb = 0;
  if (m->c_nkb) {
    kb = arc_malloc(sizeof(char*) * (size_t)m->c_nkb);
    ty = arc_malloc(sizeof(char*) * (size_t)m->c_nkb);
    for (int i = 0; i < m->c_nkb; ++i) {
      kb[nkb] = m->c_kb[i];
      ty[nkb++] = m->c_kind[i][0] ? kb_kind(m->c_kind[i]) : "";
    }
  } else if (m->n_kb && m->names_in_prompt) {
    kb = arc_malloc(sizeof(char*) * (size_t)m->n_kb);
    ty = arc_malloc(sizeof(char*) * (size_t)m->n_kb);
    for (int i = 0; i < m->n_kb; ++i) {
      kb[nkb] = m->kb[i].name;
      ty[nkb++] = kb_kind(m->kb[i].kind);
    }
  }
  char* prev = m->c_prev ? arc_strdup(m->c_prev) : NULL;
  if (!prev && m->auto_prev) {
    /* prev_text(hist, t_now) */
    sbuf b = {0};
    sb_put(&b, "");
    for (int i = 0; i < e->n_hist; ++i)
      if (e->hist[i].s >= t_now - 60.0 && e->hist[i].e <= t_now - 0.3) {
        char* w = arc_norm_prev(e->hist[i].text);
        if (w[0]) {
          if (b.n) sb_put(&b, " ");
          sb_put(&b, w);
        }
        arc_free(w);
      }
    char** ws;
    int nw = arc_split_words(b.s, &ws);
    sbuf o = {0};
    sb_put(&o, "");
    for (int i = nw > 40 ? nw - 40 : 0; i < nw; ++i) {
      if (o.n) sb_put(&o, " ");
      sb_put(&o, ws[i]);
    }
    arc_free_words(ws, nw);
    arc_free(b.s);
    prev = sb_take(&o);
  }
  int has = (dom && *dom) || nkb || (prev && *prev) || m->n_ses;
  int n = 0;
  if (has) {
    const int mx = m->ctx_max;
    int* buf = arc_malloc(sizeof(int) * 8192);
    int* tmp = arc_malloc(sizeof(int) * 4096);
    if (dom && *dom) {
      char* d = under_to_space(dom);
      int k = arc_tok_encode(e->tok, d, tmp, 4096);
      arc_free(d);
      buf[n++] = CTX_DOM;
      for (int i = 0; i < k && i < 4; ++i) buf[n++] = tmp[i];
    }
    if (m->n_ses) {   /* render_session */
      int budget = m->ctx_ses_max < mx - n ? m->ctx_ses_max : mx - n;
      if (budget >= 3) {
        int o0 = n, len = 1;
        buf[n++] = CTX_SES;
        for (int i = 0; i < m->n_ses; ++i) {
          int k = arc_tok_encode(e->tok, m->ses_text[i], tmp, 4096);
          if (k < 0) k = 0;
          if (len + 1 + k > budget) break;
          buf[n++] = m->ses_mark[i];
          memcpy(buf + n, tmp, sizeof(int) * (size_t)k);
          n += k;
          len += 1 + k;
        }
        if (len <= 1) n = o0;
      }
    }
    if (nkb) {
      int kbud = m->ctx_kb_max < mx - n ? m->ctx_kb_max : mx - n;
      if (kbud > 2) {
        arc_ctx c = {NULL, kb, ty, nkb, NULL};
        n += arc_ctx_prompt_ex(e->tok, &c, "full", kbud, 0, kbud, 0, buf + n);
      }
    }
    if (prev && *prev) {
      int* ids = arc_malloc(sizeof(int) * 65536);
      int k = arc_tok_encode(e->tok, prev, ids, 65536);
      int bp = m->ctx_prev_max < mx - n - 1 ? m->ctx_prev_max : mx - n - 1;
      if (bp > 4) {
        buf[n++] = CTX_PREV;
        for (int i = k > bp ? k - bp : 0; i < k; ++i) buf[n++] = ids[i];
      }
      arc_free(ids);
    }
    if (n > mx) n = mx;
    if (n > cap) n = cap;
    memcpy(out, buf, sizeof(int) * (size_t)n);
    arc_free(buf);
    arc_free(tmp);
  }
  arc_free(prev);
  arc_free(kb);
  arc_free(ty);
  return n;
}

/* ------------------------------------------------------------------ front end and encoder */

/* LogMel (normalize NA): pre-emphasis 0.97, |STFT|^2 (n_fft 512, hop 160, window 400 centred, zero padding),
 * log(fb @ p + 2^-24); the first n // 160 frames. M [nf, 80] */
static float* log_mel(arc_nft2* m, const float* y, int n, int* nf) {
  int F = n / HOP;
  *nf = F;
  float* x = arc_malloc(sizeof(float) * (size_t)(n + NFFT));
  float* xp = x + NFFT / 2;
  memset(x, 0, sizeof(float) * (size_t)(n + NFFT));
  xp[0] = y[0];
  const float a = 0.97f;
  for (int i = 1; i < n; ++i) xp[i] = y[i] - a * y[i - 1];
  float* M = arc_malloc(sizeof(float) * (size_t)(F > 0 ? F : 1) * 80);
  float fr[NFFT], p[NFFT / 2 + 1];
  double spec[NFFT + 2], work[2 * NFFT + 4];
  for (int t = 0; t < F; ++t) {
    const float* s = x + (size_t)t * HOP;
    for (int k = 0; k < NFFT; ++k) fr[k] = s[k] * m->window[k];
    arc_rfft(m->fft, fr, spec, work);
    for (int k = 0; k <= NFFT / 2; ++k) {
      float re = (float)spec[2 * k], im = (float)spec[2 * k + 1];
      float mag = sqrtf(re * re + im * im);
      p[k] = mag * mag;
    }
    for (int b = 0; b < 80; ++b) {
      const float* w = m->fb + (size_t)b * 257;
      double acc = 0;
      for (int k = 0; k < 257; ++k) acc += (double)w[k] * p[k];
      M[(size_t)t * 80 + b] = logf((float)acc + 5.9604644775390625e-08f);
    }
  }
  arc_free(x);
  return M;
}

static int calc_len(int n) {
  for (int k = 0; k < 3; ++k) n = n / 2 + 1;
  return n > 0 ? n : 0;
}

/* OnnxEncoder.__call__ for one clip: y cut / zero-padded to n = max(80, len // 80 * 80) samples -> H [T, d] (every
 * encoder frame, incl. frame 0) and Htap [T, d]; memoised on the samples (one pass serves the RNNT, the decoder and
 * the pooled features of a clip up to 30 s). The returned arrays belong to the memo. */
static int encode(arc_nft2* m, const float* wav, long len, const float** H, const float** Ht, int* T) {
  int n = (int)(len / 80 * 80);
  if (n < 80) n = 80;
  float* y = arc_calloc(sizeof(float) * (size_t)n);
  memcpy(y, wav, sizeof(float) * (size_t)(len < n ? len : n));
  uint64_t key = fnv((const char*)y, sizeof(float) * (size_t)n) ^ (uint64_t)n;
  if (m->memo_H && m->memo_key == key && m->memo_n == n) {
    arc_free(y);
    *H = m->memo_H;
    *Ht = m->memo_Ht;
    *T = m->memo_T;
    return 0;
  }
  int nf;
  float* M = log_mel(m, y, n, &nf);
  arc_free(y);
  const arc_nft2_dims* d = &m->d;
  int D = d->d_model, Tf = calc_len(nf), cap = Tf + 64;
  float* Ho = arc_calloc(sizeof(float) * (size_t)cap * D);
  float* Hto = arc_calloc(sizeof(float) * (size_t)cap * D);
  int got = 0, rc = 0;
  arc_nft2_cache* c = arc_nft2_be_cache_new(m->be);
  int pre_n = d->pre;
  float* pre = arc_calloc(sizeof(float) * (size_t)(pre_n + 1) * 80);
  float* xb = arc_malloc(sizeof(float) * (size_t)(pre_n + d->chunk + d->first + 1) * 80);
  float* ob = arc_malloc(sizeof(float) * (size_t)64 * D);
  float* tb = arc_malloc(sizeof(float) * (size_t)64 * D);
  if (!c) rc = -1;
  for (int i = 0; !rc && i < nf;) {
    int first = i == 0 && d->first > 0;
    int nn = first ? d->first : d->chunk;
    if (i + nn > nf) nn = nf - i;
    int xn;
    if (first) {
      memcpy(xb, M, sizeof(float) * (size_t)nn * 80);
      xn = nn;
    } else {
      memcpy(xb, pre, sizeof(float) * (size_t)pre_n * 80);
      memcpy(xb + (size_t)pre_n * 80, M + (size_t)i * 80, sizeof(float) * (size_t)nn * 80);
      xn = pre_n + nn;
    }
    /* pre = concat(pre, x)[-pre_n:] */
    {
      int tot = pre_n + xn;
      float* cat = arc_malloc(sizeof(float) * (size_t)tot * 80);
      memcpy(cat, pre, sizeof(float) * (size_t)pre_n * 80);
      memcpy(cat + (size_t)pre_n * 80, xb, sizeof(float) * (size_t)xn * 80);
      memcpy(pre, cat + (size_t)(tot - pre_n) * 80, sizeof(float) * (size_t)pre_n * 80);
      arc_free(cat);
    }
    int k = 0;
    if (arc_nft2_be_chunk(m->be, c, first, xb, xn, ob, m->tap ? tb : NULL, 64, &k)) {
      rc = -1;
      break;
    }
    for (int r = 0; r < k && got < cap; ++r, ++got) {
      memcpy(Ho + (size_t)got * D, ob + (size_t)r * D, sizeof(float) * D);
      if (m->tap) memcpy(Hto + (size_t)got * D, tb + (size_t)r * D, sizeof(float) * D);
    }
    i += nn;
  }
  arc_nft2_be_cache_free(c);
  arc_free(pre);
  arc_free(xb);
  arc_free(ob);
  arc_free(tb);
  if (rc) {
    arc_free(M);
    arc_free(Ho);
    arc_free(Hto);
    return -1;
  }
  /* fit: pad with the last frame (never seen), cut to the full-sequence length */
  for (int r = got; r < Tf; ++r) {
    if (got) {
      memcpy(Ho + (size_t)r * D, Ho + (size_t)(got - 1) * D, sizeof(float) * D);
      memcpy(Hto + (size_t)r * D, Hto + (size_t)(got - 1) * D, sizeof(float) * D);
    }
  }
  int idx = m->dump_enc++;
  dumpf(m, "mel", idx, M, nf, 80);
  dumpf(m, "H", idx, Ho, Tf, D);
  if (m->tap) dumpf(m, "Htap", idx, Hto, Tf, D);
  arc_free(M);
  arc_free(m->memo_H);
  arc_free(m->memo_Ht);
  m->memo_H = Ho;
  m->memo_Ht = Hto;
  m->memo_T = Tf;
  m->memo_n = n;
  m->memo_key = key;
  *H = Ho;
  *Ht = Hto;
  *T = Tf;
  return 0;
}

/* ------------------------------------------------------------------ RNNT greedy (OnnxRNNT.greedy) */

static int argmax(const float* v, int n) {
  int b = 0;
  for (int i = 1; i < n; ++i)
    if (v[i] > v[b]) b = i;
  return b;
}

/* nft2.tok_stats: log p of token kk under the unbiased joint distribution, the runner-up non-blank token, its log p */
static void tok_stats(const float* row, int V1, int blank, int kk, float* out3) {
  double mx = row[0];
  for (int k = 1; k < V1; ++k)
    if (row[k] > mx) mx = row[k];
  double s = 0;
  for (int k = 0; k < V1; ++k) s += exp((double)row[k] - mx);
  double lse = mx + log(s);
  int k2 = kk == 0 ? 1 : 0;
  for (int k = 0; k < blank; ++k)
    if (k != kk && row[k] > row[k2]) k2 = k;
  out3[0] = (float)((double)row[kk] - lse);
  out3[1] = (float)k2;
  out3[2] = (float)((double)row[k2] - lse);
}

/* the greedy decode; tok_out (may be NULL): 3 floats per token (tok_stats), malloc'd */
static int rnnt_greedy(arc_nft2* m, const float* H, int T, int bias, int** ids_out, int** frames_out, float** tok_out) {
  const arc_nft2_dims* d = &m->d;
  int J = d->joint_hidden, V1 = d->vocab + 1, blank = d->blank;
  int cap = 64, n = 0, rc = 0;
  int* ids = arc_malloc(sizeof(int) * (size_t)cap);
  int* fr = arc_malloc(sizeof(int) * (size_t)cap);
  float* ts = arc_malloc(sizeof(float) * 3 * (size_t)cap);
  *ids_out = ids;
  *frames_out = fr;
  if (tok_out) *tok_out = ts;
  if (T <= 0) {
    if (!tok_out) arc_free(ts);
    return 0;
  }
  float* hc = arc_calloc(sizeof(float) * (size_t)2 * d->pred_layers * d->pred_hidden);
  float* g = arc_malloc(sizeof(float) * (size_t)J);
  float* Fe = arc_malloc(sizeof(float) * (size_t)T * J);
  enum { BLK = 16 };
  float* lg = arc_malloc(sizeof(float) * (size_t)BLK * V1);
  float* row = arc_malloc(sizeof(float) * (size_t)V1);
  float* biased = arc_malloc(sizeof(float) * (size_t)blank);
  boost_trie* tr = bias ? m->trie : NULL;
  int bs = 0;
  const float al = (float)m->alpha;
  if (arc_nft2_be_pred(m->be, blank, hc, g) || arc_nft2_be_joint_enc(m->be, H, T, Fe)) rc = -1;
  int t = 0;
  while (!rc && t < T) {
    /* the first frame from t whose best symbol is not blank */
    int found = -1;
    for (int t0 = t; t0 < T && found < 0; t0 += BLK) {
      int nb = T - t0 < BLK ? T - t0 : BLK;
      if (arc_nft2_be_joint(m->be, Fe + (size_t)t0 * J, nb, g, lg)) {
        rc = -1;
        break;
      }
      for (int r = 0; r < nb; ++r)
        if (argmax(lg + (size_t)r * V1, V1) != blank) {
          found = t0 + r;
          memcpy(row, lg + (size_t)r * V1, sizeof(float) * (size_t)V1);
          break;
        }
    }
    if (rc || found < 0) break;
    t = found;
    int sym = 0;
    for (;;) {
      int kk = argmax(row, V1);
      if (kk == blank) break;
      if (tr) {
        const float* sc = trie_scores(tr, bs);
        for (int k = 0; k < blank; ++k) biased[k] = row[k] + al * sc[k];
        kk = argmax(biased, blank);
        bs = trie_advance(tr, bs, kk);
      }
      if (n == cap) {
        cap *= 2;
        int* a = arc_malloc(sizeof(int) * (size_t)cap);
        int* b = arc_malloc(sizeof(int) * (size_t)cap);
        float* c = arc_malloc(sizeof(float) * 3 * (size_t)cap);
        memcpy(a, ids, sizeof(int) * (size_t)n);
        memcpy(b, fr, sizeof(int) * (size_t)n);
        memcpy(c, ts, sizeof(float) * 3 * (size_t)n);
        arc_free(ids);
        arc_free(fr);
        arc_free(ts);
        ids = a;
        fr = b;
        ts = c;
      }
      tok_stats(row, V1, blank, kk, ts + 3 * (size_t)n);
      ids[n] = kk;
      fr[n++] = t;
      if (arc_nft2_be_pred(m->be, kk, hc, g)) {
        rc = -1;
        break;
      }
      if (++sym >= d->max_sym) break;
      if (arc_nft2_be_joint(m->be, Fe + (size_t)t * J, 1, g, row)) {
        rc = -1;
        break;
      }
    }
    t += 1;
  }
  arc_free(hc);
  arc_free(g);
  arc_free(Fe);
  arc_free(lg);
  arc_free(row);
  arc_free(biased);
  *ids_out = ids;
  *frames_out = fr;
  if (tok_out) *tok_out = ts;
  else
    arc_free(ts);
  return rc ? -1 : n;
}

typedef struct {
  char* text;
  double start, end;
  int tag, tag_begin;
  int t0, t1;      /* its tokens [t0, t1) */
  int has_conf;
  float conf;
  char* alt;       /* the runner-up spelling or NULL */
  float alt_p;
  char** xalt;     /* other spellings from the n-best (nbest_alts), malloc'd */
  float* xalt_p;
  int n_xalt;
} rword;

/* Pieces.words: ids + frames -> words (frame f covers [(f - 1) * 80, f * 80) ms); ts (tok_stats per token, may be
 * NULL): conf and the runner-up spelling */
static int rnnt_words(const arc_nft2* m, const int* ids, const int* fr, int n, double t0, const float* ts, rword** out) {
  rword* w = arc_calloc(sizeof(rword) * (size_t)(n + 1));
  int nw = 0;
  const double fs = 0.08;
  for (int i = 0; i < n; ++i) {
    const char* p = ids[i] >= 0 && ids[i] < m->n_piece ? m->piece[ids[i]] : "";
    double st = t0 + fmax(0.0, (fr[i] - 1) * fs), en = t0 + fmax(fs, fr[i] * fs);
    if (p[0] == 1 || !nw) {
      while (*p == 1) p++;
      w[nw].text = arc_strdup(p);
      w[nw].start = st;
      w[nw].end = en;
      w[nw].t0 = i;
      w[nw].t1 = i + 1;
      nw++;
    } else {
      w[nw - 1].t1 = i + 1;
      size_t a = strlen(w[nw - 1].text), b = strlen(p);
      char* s = arc_malloc(a + b + 1);
      memcpy(s, w[nw - 1].text, a);
      memcpy(s + a, p, b + 1);
      arc_free(w[nw - 1].text);
      w[nw - 1].text = s;
      w[nw - 1].end = en;
    }
  }
  int k = 0;
  for (int i = 0; i < nw; ++i) {
    if (!w[i].text[0]) {
      arc_free(w[i].text);
      continue;
    }
    rword r = w[i];
    r.end = arc_pyround(fmax(r.end, r.start + 0.04), 3);
    r.start = arc_pyround(r.start, 3);
    r.tag = -1;
    r.tag_begin = 0;
    if (ts) {
      int j = r.t0;
      for (int q = r.t0 + 1; q < r.t1; ++q)
        if (ts[3 * q] < ts[3 * j]) j = q;
      r.has_conf = 1;
      r.conf = (float)arc_pyround(exp(ts[3 * j]), 4);
      int k2 = (int)ts[3 * j + 1];
      if (ts[3 * j + 2] >= log(0.05) && k2 >= 0 && k2 < m->n_piece && (m->piece[k2][0] == 1) == (j == r.t0)) {
        sbuf b = {0};
        sb_put(&b, "");
        for (int q = r.t0; q < r.t1; ++q) {
          const char* pc = q == j ? m->piece[k2] : (ids[q] >= 0 && ids[q] < m->n_piece ? m->piece[ids[q]] : "");
          if (q == r.t0)
            while (*pc == 1) pc++;
          sb_put(&b, pc);
        }
        char* alt = sb_take(&b);
        if (alt[0] && strcmp(alt, r.text)) {
          r.alt = alt;
          r.alt_p = (float)arc_pyround(exp(ts[3 * j + 2]), 4);
        } else
          arc_free(alt);
      }
    }
    w[k++] = r;
  }
  *out = w;
  return k;
}

/* ------------------------------------------------------------------ Arc adapters */

static void linear(arc_nft2* m, const float* x, int T, int K, const float* W, int O, const float* b, float* y) {
  arc_gemm_nt(m->e->moon->pool, T, O, K, x, K, W, K, y, O, b, 0);
}

/* mem_with_pos(heads, pos_emb, P): MemProj(P) + pos_emb[4i + pos_off] - pos_emb[i]; P [T, d] -> [T, 320] */
static float* mem_with_pos(arc_nft2* m, const float* P, int T) {
  int D = m->d.d_model;
  float* x = arc_malloc(sizeof(float) * (size_t)T * D);
  memcpy(x, P, sizeof(float) * (size_t)T * D);
  arc_layernorm(x, T, D, D, m->mem_ln_w, m->mem_ln_b, 1e-5f);
  float* h = arc_malloc(sizeof(float) * (size_t)T * 1024);
  linear(m, x, T, D, m->mem_fc1_w, 1024, m->mem_fc1_b, h);
  arc_gelu(h, (size_t)T * 1024);
  float* M = arc_malloc(sizeof(float) * (size_t)T * ARC_MOON_D);
  linear(m, h, T, 1024, m->mem_fc2_w, ARC_MOON_D, m->mem_fc2_b, M);
  arc_layernorm(M, T, ARC_MOON_D, ARC_MOON_D, m->mem_out_w, m->mem_out_b, 1e-5f);
  const float* W = m->e->moon->pos_emb;
  for (int i = 0; i < T; ++i) {
    int a = 4 * i + m->pos_off, b = i;
    if (a > 4095) a = 4095;
    if (b > 4095) b = 4095;
    for (int c = 0; c < ARC_MOON_D; ++c) {
      float corr = W[(size_t)a * ARC_MOON_D + c] - W[(size_t)b * ARC_MOON_D + c];
      M[(size_t)i * ARC_MOON_D + c] += corr;
    }
  }
  arc_free(x);
  arc_free(h);
  return M;
}

/* FrameProj.frame: LN then proj; P [T, d] -> [T, O] */
static float* frame_proj(arc_nft2* m, const float* P, int T, const float* lw, const float* lb, const float* W,
                         const float* b, int O) {
  int D = m->d.d_model;
  float* x = arc_malloc(sizeof(float) * (size_t)(T > 0 ? T : 1) * D);
  memcpy(x, P, sizeof(float) * (size_t)T * D);
  arc_layernorm(x, T, D, D, lw, lb, 1e-5f);
  float* y = arc_malloc(sizeof(float) * (size_t)(T > 0 ? T : 1) * O);
  if (T) linear(m, x, T, D, W, O, b, y);
  arc_free(x);
  return y;
}

/* server.frame_of_enc for 80 ms frames: ceil((g * 320 + 3920) / 16000 * 10 - 1e-9) - 1 */
static int frame_of_enc(long g) { return (int)ceil(((double)g * 320 + 3920) / 16000.0 * 10 - 1e-9) - 1; }

/* pool_range(Z, n_ms, k0 = 0, k1): mean and max of the 80 ms rows over each 100 ms frame (4 sub-frames of 20 ms per
 * row), zero where none -> [k1, 2D] */
static float* pool_range(const float* Z, int T, int D, int n_ms, int k1) {
  float* out = arc_calloc(sizeof(float) * (size_t)(k1 > 0 ? k1 : 1) * 2 * D);
  if (k1 <= 0 || T <= 0) return out;
  float* mx = arc_malloc(sizeof(float) * (size_t)k1 * D);
  int* cnt = arc_calloc(sizeof(int) * (size_t)k1);
  for (size_t i = 0; i < (size_t)k1 * D; ++i) mx[i] = -1e30f;
  long g1 = n_ms < 5L * k1 + 5 ? n_ms : 5L * k1 + 5;
  for (long g = 0; g < g1; ++g) {
    int f = frame_of_enc(g);
    if (f < 0 || f >= k1) continue;
    long r = g / 4 < T - 1 ? g / 4 : T - 1;
    const float* e = Z + (size_t)r * D;
    float* s = out + (size_t)f * 2 * D;
    float* x = mx + (size_t)f * D;
    for (int c = 0; c < D; ++c) {
      s[c] += e[c];
      if (e[c] > x[c]) x[c] = e[c];
    }
    cnt[f]++;
  }
  for (int f = 0; f < k1; ++f) {
    float* s = out + (size_t)f * 2 * D;
    if (!cnt[f]) {
      memset(s, 0, sizeof(float) * 2 * D);
      continue;
    }
    float c = (float)cnt[f];
    for (int k = 0; k < D; ++k) {
      s[k] = s[k] / c;
      s[D + k] = mx[(size_t)f * D + k];
    }
  }
  arc_free(mx);
  arc_free(cnt);
  return out;
}

/* Arc._feats for nft2 (ServerFeatures over 30 s chunks with 4 s of context, flushed): X [N, 640] (arc head),
 * Xt [N, 640] (tap pooled) */
static int arc_feats(arc_nft2* m, const float* wav, long N, float** Xo, float** Xto, int* No) {
  int D = m->d.d_model, DA = m->d_arc, DT = m->d_tap;
  long CH = lround(m->chunk_s * SR / ENC_STRIDE) * ENC_STRIDE, CTX = lround(m->context_s * SR / ENC_STRIDE) * ENC_STRIDE;
  float *Z = NULL, *Zt = NULL;
  int nZ = 0;
  for (long c = 0; c < N; c += CH) {
    long s0 = c - CTX > 0 ? c - CTX : 0, s1 = c + CH + CTX < N ? c + CH + CTX : N;
    const float *H, *Ht;
    int T;
    if (encode(m, wav + s0, s1 - s0, &H, &Ht, &T)) {
      arc_free(Z);
      arc_free(Zt);
      return -1;
    }
    int nv = T - 1;   /* drop0 */
    long a = (c - s0) / ENC_STRIDE;
    long hi = c + CH < N ? c + CH : N;
    long b = a + (hi - c + ENC_STRIDE - 1) / ENC_STRIDE;
    if (b > nv) b = nv;
    int k = b > a ? (int)(b - a) : 0;
    float* z = frame_proj(m, H + (size_t)(1 + a) * D, k, m->arc_ln_w, m->arc_ln_b, m->arc_proj_w, m->arc_proj_b, DA);
    float* zt = frame_proj(m, Ht + (size_t)(1 + a) * D, k, m->tap_ln_w, m->tap_ln_b, m->tap_proj_w, m->tap_proj_b, DT);
    float* nz = arc_malloc(sizeof(float) * (size_t)(nZ + k + 1) * DA);
    float* nzt = arc_malloc(sizeof(float) * (size_t)(nZ + k + 1) * DT);
    if (nZ) {
      memcpy(nz, Z, sizeof(float) * (size_t)nZ * DA);
      memcpy(nzt, Zt, sizeof(float) * (size_t)nZ * DT);
    }
    memcpy(nz + (size_t)nZ * DA, z, sizeof(float) * (size_t)k * DA);
    memcpy(nzt + (size_t)nZ * DT, zt, sizeof(float) * (size_t)k * DT);
    arc_free(Z);
    arc_free(Zt);
    arc_free(z);
    arc_free(zt);
    Z = nz;
    Zt = nzt;
    nZ += k;
  }
  int n_ms = (int)(N / 320) + 1;
  int k1 = frame_of_enc(n_ms - 1) + 1;
  if (!nZ || k1 <= 0) {
    arc_free(Z);
    arc_free(Zt);
    *Xo = arc_calloc(sizeof(float) * DIN);
    *Xto = arc_calloc(sizeof(float) * DIN);
    *No = 0;
    return 0;
  }
  float* pa = pool_range(Z, nZ, DA, n_ms, k1);    /* [k1, 2 DA] */
  float* pt = pool_range(Zt, nZ, DT, n_ms, k1);   /* [k1, 640] */
  arc_free(Z);
  arc_free(Zt);
  /* arc.head: LN(2 DA) -> Linear -> GELU -> Linear(640) */
  arc_layernorm(pa, k1, 2 * DA, 2 * DA, m->arc_mlp_ln_w, m->arc_mlp_ln_b, 1e-5f);
  float* h = arc_malloc(sizeof(float) * (size_t)k1 * 1024);
  linear(m, pa, k1, 2 * DA, m->arc_mlp1_w, 1024, m->arc_mlp1_b, h);
  arc_gelu(h, (size_t)k1 * 1024);
  float* X = arc_malloc(sizeof(float) * (size_t)k1 * DIN);
  linear(m, h, k1, 1024, m->arc_mlp3_w, DIN, m->arc_mlp3_b, X);
  arc_free(h);
  arc_free(pa);
  int j0 = frame_of_enc(0);
  for (int j = 0; j < k1 && j < j0; ++j) {
    memset(X + (size_t)j * DIN, 0, sizeof(float) * DIN);
    memset(pt + (size_t)j * DIN, 0, sizeof(float) * DIN);
  }
  *Xo = X;
  *Xto = pt;
  *No = k1;
  return 0;
}

/* ------------------------------------------------------------------ decoder tags on the RNNT words */

typedef struct {
  int tag, a, b;   /* tag id 32600.., words [a, b) */
} span3;

typedef struct {
  uint32_t* A;   /* ' '.join(keys) */
  uint32_t* B;
  int la, lb;
  int *aw, *bw, *cm;   /* word of each char, B char aligned to each A char (-1) */
  uint32_t** ak;       /* keys */
  int* akn;
  uint32_t** bk;
  int* bkn;
  int na, nb;
} charalign;

static void keys_of(char** w, int n, uint32_t*** k, int** kn) {
  *k = arc_malloc(sizeof(uint32_t*) * (size_t)(n + 1));
  *kn = arc_malloc(sizeof(int) * (size_t)(n + 1));
  for (int i = 0; i < n; ++i) {
    (*k)[i] = arc_malloc(sizeof(uint32_t) * wkey_cap(w[i]));
    (*kn)[i] = wkey_cps(w[i], (*k)[i]);
  }
}

static void join_keys(uint32_t** k, const int* kn, int n, uint32_t** S, int** wi, int* len) {
  int L = 0;
  for (int i = 0; i < n; ++i) L += kn[i] + 1;
  *S = arc_malloc(sizeof(uint32_t) * (size_t)(L + 1));
  *wi = arc_malloc(sizeof(int) * (size_t)(L + 1));
  int p = 0;
  for (int i = 0; i < n; ++i) {
    for (int c = 0; c < kn[i]; ++c) {
      (*S)[p] = k[i][c];
      (*wi)[p++] = i;
    }
    (*S)[p] = ' ';
    (*wi)[p++] = i;
  }
  *len = L > 0 ? L - 1 : 0;   /* no space after the last */
}

static void char_align(charalign* c, char** src, int ns, char** dst, int nd) {
  memset(c, 0, sizeof *c);
  c->na = ns;
  c->nb = nd;
  keys_of(src, ns, &c->ak, &c->akn);
  keys_of(dst, nd, &c->bk, &c->bkn);
  join_keys(c->ak, c->akn, ns, &c->A, &c->aw, &c->la);
  join_keys(c->bk, c->bkn, nd, &c->B, &c->bw, &c->lb);
  c->cm = arc_malloc(sizeof(int) * (size_t)(c->la + 1));
  for (int i = 0; i < c->la; ++i) c->cm[i] = -1;
  int nb;
  smblock* bl = sm_blocks(c->A, c->la, c->B, c->lb, &nb);
  int i = 0, j = 0;
  for (int x = 0; x < nb; ++x) {   /* get_opcodes */
    int ai = bl[x].a, bj = bl[x].b, sz = bl[x].n;
    if (i < ai && j < bj)   /* replace: proportional */
      for (int k = 0; k < ai - i; ++k) {
        int off = (int)(((long)k * (bj - j)) / (ai - i));
        c->cm[i + k] = j + (bj - j - 1 < off ? bj - j - 1 : off);
      }
    for (int k = 0; k < sz; ++k) c->cm[ai + k] = bj + k;
    i = ai + sz;
    j = bj + sz;
  }
  arc_free(bl);
}

static void char_align_free(charalign* c) {
  for (int i = 0; i < c->na; ++i) arc_free(c->ak[i]);
  for (int i = 0; i < c->nb; ++i) arc_free(c->bk[i]);
  arc_free(c->ak);
  arc_free(c->akn);
  arc_free(c->bk);
  arc_free(c->bkn);
  arc_free(c->A);
  arc_free(c->B);
  arc_free(c->aw);
  arc_free(c->bw);
  arc_free(c->cm);
}

/* ''.join(keys[a:b]) into a fresh array */
static uint32_t* join_range(uint32_t** k, const int* kn, int a, int b, int* n) {
  int L = 0;
  for (int i = a; i < b; ++i) L += kn[i];
  uint32_t* s = arc_malloc(sizeof(uint32_t) * (size_t)(L + 1));
  int p = 0;
  for (int i = a; i < b; ++i) {
    memcpy(s + p, k[i], sizeof(uint32_t) * (size_t)kn[i]);
    p += kn[i];
  }
  *n = L;
  return s;
}

/* map_spans_char(src_words, spans, dst_words, min_ratio=0.5, cover=0.5) */
static int map_spans_char(const charalign* c, const span3* sp, int nsp, span3* out) {
  int no = 0;
  int* hit = arc_calloc(sizeof(int) * (size_t)(c->nb + 1));
  for (int s = 0; s < nsp; ++s) {
    int a = sp[s].a, b = sp[s].b, ncs = 0;
    memset(hit, 0, sizeof(int) * (size_t)(c->nb + 1));
    for (int ch = 0; ch < c->la; ++ch) {
      if (!(a <= c->aw[ch] && c->aw[ch] < b) || c->A[ch] == ' ') continue;
      ncs++;
      int dd = c->cm[ch];
      if (dd >= 0 && c->B[dd] != ' ') hit[c->bw[dd]]++;
    }
    int j1 = -1, j2 = -1;
    for (int j = 0; j < c->nb; ++j) {
      if (!hit[j]) continue;
      double lenj = c->bkn[j] > 1 ? c->bkn[j] : 1;
      if (hit[j] >= 0.5 * lenj || hit[j] >= 0.5 * ncs) {
        if (j1 < 0) j1 = j;
        j2 = j + 1;
      }
    }
    if (j1 < 0) continue;
    int n1, n2;
    uint32_t* s1 = join_range(c->ak, c->akn, a, b, &n1);
    uint32_t* s2 = join_range(c->bk, c->bkn, j1, j2, &n2);
    int same = n1 == n2 && !memcmp(s1, s2, sizeof(uint32_t) * (size_t)n1);
    int keep = same || sm_ratio(s1, n1, s2, n2) >= 0.5;
    arc_free(s1);
    arc_free(s2);
    if (!keep) continue;
    if (!no || j1 >= out[no - 1].b) out[no++] = (span3){sp[s].tag, j1, j2};
  }
  arc_free(hit);
  return no;
}

static int keys_eq(uint32_t** k, const int* kn, int a1, int b1, int a2, int b2) {
  if (b1 - a1 != b2 - a2) return 0;
  for (int i = 0; i < b1 - a1; ++i)
    if (kn[a1 + i] != kn[a2 + i] || memcmp(k[a1 + i], k[a2 + i], sizeof(uint32_t) * (size_t)kn[a1 + i])) return 0;
  return 1;
}

static int is_space_tag(int t) { return t >= 32636 && t <= 32639; }

/* unify_types: one entity type per surface (a space type wins, else the most frequent, first on a tie) */
static void unify_types(span3* sp, int n, uint32_t** k, const int* kn) {
  int* pick = arc_malloc(sizeof(int) * (size_t)(n + 1));
  for (int i = 0; i < n; ++i) {
    pick[i] = sp[i].tag;
    /* the group of i: its first member decides; scan all members with the same key */
    int first = -1;
    for (int j = 0; j < n && first < 0; ++j)
      if (keys_eq(k, kn, sp[j].a, sp[j].b, sp[i].a, sp[i].b)) first = j;
    int distinct = 0, any_space = 0;
    for (int j = 0; j < n; ++j)
      if (keys_eq(k, kn, sp[j].a, sp[j].b, sp[i].a, sp[i].b)) {
        if (sp[j].tag != sp[first].tag) distinct = 1;
        any_space |= is_space_tag(sp[j].tag);
      }
    if (!distinct) continue;
    /* Counter(space tags or all).most_common(1): count, ties in first-seen order */
    int best = -1, bc = 0;
    for (int j = 0; j < n; ++j) {
      if (!keys_eq(k, kn, sp[j].a, sp[j].b, sp[i].a, sp[i].b)) continue;
      if (any_space && !is_space_tag(sp[j].tag)) continue;
      int seen = 0;
      for (int q = 0; q < j && !seen; ++q)
        seen = sp[q].tag == sp[j].tag && keys_eq(k, kn, sp[q].a, sp[q].b, sp[i].a, sp[i].b) &&
               (!any_space || is_space_tag(sp[q].tag));
      if (seen) continue;
      int cnt = 0;
      for (int q = 0; q < n; ++q)
        cnt += sp[q].tag == sp[j].tag && keys_eq(k, kn, sp[q].a, sp[q].b, sp[i].a, sp[i].b);
      if (cnt > bc) bc = cnt, best = sp[j].tag;
    }
    if (best >= 0) pick[i] = best;
  }
  for (int i = 0; i < n; ++i) sp[i].tag = pick[i];
  arc_free(pick);
}

/* heard_by_decoder(src, dst)(i, j): the decoder keys aligned to RNNT words [i, j), joined */
static uint32_t* heard_by_decoder(const charalign* c, int i, int j, int* n) {
  int lo = -1, hi = -1;
  for (int ch = 0; ch < c->la; ++ch) {
    int dd = c->cm[ch];
    if (dd < 0 || c->A[ch] == ' ' || c->B[dd] == ' ') continue;
    int bw = c->bw[dd];
    if (bw < i || bw >= j) continue;
    int s = c->aw[ch];
    if (lo < 0 || s < lo) lo = s;
    if (s > hi) hi = s;
  }
  if (lo < 0) {
    *n = 0;
    return arc_malloc(sizeof(uint32_t));
  }
  return join_range(c->ak, c->akn, lo, hi + 1, n);
}

static int cmp_span_start(const void* x, const void* y) {
  const span3 *a = x, *b = y;
  return (a->a > b->a) - (a->a < b->a);
}

/* list_spans: names-list entries spelled out in the RNNT words, longest first */
static int list_spans(arc_nft2* m, span3* sp, int n, int cap, const charalign* c) {
  uint32_t** k = c->bk;
  const int* kn = c->bkn;
  int nd = c->nb;
  unsigned char* used = arc_calloc((size_t)nd + 1);
  int* order = arc_malloc(sizeof(int) * (size_t)(m->n_lk + 1));
  for (int i = 0; i < m->n_lk; ++i) order[i] = i;
  for (int a = 1; a < m->n_lk; ++a) {   /* stable sort by -len(words) */
    int t = order[a], b = a - 1;
    while (b >= 0 && m->lk[order[b]].nw < m->lk[t].nw) {
      order[b + 1] = order[b];
      b--;
    }
    order[b + 1] = t;
  }
  for (int oi = 0; oi < m->n_lk; ++oi) {
    const list_key* L = &m->lk[order[oi]];
    int nw = L->nw;
    int tagid = arc_tag_id(L->tag);
    for (int i = 0; i + nw <= nd; ++i) {
      /* tuple(keys[i:i+n]) == k */
      int p = 0, eq = 1;
      for (int w = 0; w < nw && eq; ++w) {
        int len = 0;
        while (p + len < L->n && L->k[p + len] != 0) len++;
        eq = kn[i + w] == len && !memcmp(k[i + w], L->k + p, sizeof(uint32_t) * (size_t)len);
        p += len + 1;
      }
      if (!eq) continue;
      int u = 0;
      for (int w = i; w < i + nw; ++w) u |= used[w];
      if (u) continue;
      int nov = 0, nent = 0;
      for (int s = 0; s < n; ++s)
        if (sp[s].a < i + nw && sp[s].b > i) {
          nov++;
          const char* tn = arc_tag_name(sp[s].tag);
          nent += tn && !strncmp(tn, "ENTITY_", 7);
        }
      int okk = 0;
      if (!nent && m->list_add && !nov && L->chars >= m->list_min_chars &&
          m->list_add_long_chars > 0 && L->chars >= m->list_add_long_chars)
        okk = 1;   /* a long exact list match: not made up by the bias, no decoder check */
      else if (!nent && m->list_add && !nov && L->chars >= m->list_min_chars) {
        int hn;
        uint32_t* h = heard_by_decoder(c, i, i + nw, &hn);
        uint32_t* kk = arc_malloc(sizeof(uint32_t) * (size_t)(L->chars + 1));
        int q = 0;
        for (int z = 0; z < L->n; ++z)
          if (L->k[z]) kk[q++] = L->k[z];
        okk = sm_ratio(h, hn, kk, q) >= m->list_add_ratio;
        arc_free(h);
        arc_free(kk);
      }
      if (nent || okk) {
        if (nent) {   /* drop every overlapping span */
          int w = 0;
          for (int s = 0; s < n; ++s)
            if (!(sp[s].a < i + nw && sp[s].b > i)) sp[w++] = sp[s];
          n = w;
        }
        if (n < cap) sp[n++] = (span3){tagid, i, i + nw};
        for (int w = i; w < i + nw; ++w) used[w] = 1;
      }
    }
  }
  /* sorted by start, stable */
  for (int a = 1; a < n; ++a) {
    span3 t = sp[a];
    int b = a - 1;
    while (b >= 0 && sp[b].a > t.a) {
      sp[b + 1] = sp[b];
      b--;
    }
    sp[b + 1] = t;
  }
  arc_free(used);
  arc_free(order);
  return n;
}

/* case_from: the decoder's casing / punctuation onto RNNT words whose key matches (word-level difflib) */
static void case_from(rword* w, int nw, char** dec, int nd) {
  char** ka = arc_malloc(sizeof(char*) * (size_t)(nw + 1));
  char** kb = arc_malloc(sizeof(char*) * (size_t)(nd + 1));
  uint32_t* A = arc_malloc(sizeof(uint32_t) * (size_t)(nw + 1));
  uint32_t* B = arc_malloc(sizeof(uint32_t) * (size_t)(nd + 1));
  char** vocab = arc_malloc(sizeof(char*) * (size_t)(nw + nd + 1));
  int nv = 0;
  for (int i = 0; i < nw + nd; ++i) {
    char* k = lkey(i < nw ? w[i].text : dec[i - nw]);
    int id = -1;
    for (int v = 0; v < nv && id < 0; ++v)
      if (!strcmp(vocab[v], k)) id = v;
    if (id < 0) vocab[id = nv++] = k;
    else
      arc_free(k);
    if (i < nw) {
      A[i] = (uint32_t)id;
      ka[i] = vocab[id];
    } else {
      B[i - nw] = (uint32_t)id;
      kb[i - nw] = vocab[id];
    }
  }
  int nb;
  smblock* bl = sm_blocks(A, nw, B, nd, &nb);
  for (int x = 0; x < nb; ++x)
    for (int k = 0; k < bl[x].n; ++k)
      if (ka[bl[x].a + k][0]) {
        arc_free(w[bl[x].a + k].text);
        w[bl[x].a + k].text = arc_strdup(dec[bl[x].b + k]);
      }
  arc_free(bl);
  for (int v = 0; v < nv; ++v) arc_free(vocab[v]);
  arc_free(vocab);
  arc_free(ka);
  arc_free(kb);
  arc_free(A);
  arc_free(B);
}

/* ------------------------------------------------------------------ Arc Tagger on the finals */

static char* tag_words(arc_nft2* m, const arc_utt_build* u);

int arc_nft2_set_tagger(arc_nft2* m, const char* model_dir, const char* domain, int int8, char* err, size_t errlen) {
  if (err && errlen) err[0] = 0;
  if (!m) return -1;
#ifdef ARC_WITH_ARCTAG
  if (m->tagger) arctag_destroy((arctag_t*)m->tagger);
  m->tagger = NULL;
  if (!model_dir) return 0;
  arctag_options opt;
  arctag_options_init(&opt);
  opt.precision = int8 ? ARCTAG_INT8 : ARCTAG_FP32;
  opt.backend = ARCTAG_BACKEND_NATIVE;
  arctag_t* h = arctag_create_ex(model_dir, &opt, err, errlen);
  if (!h) return -1;
  if (arctag_set_option(h, "domain", domain ? domain : "kids")) {
    if (err && errlen) snprintf(err, errlen, "arc tagger: unknown domain %s", domain ? domain : "kids");
    arctag_destroy(h);
    return -1;
  }
  m->tagger = h;
  return 0;
#else
  (void)domain, (void)int8;
  if (!model_dir) return 0;
  if (err && errlen) snprintf(err, errlen, "libarc was built without ARC_WITH_ARCTAG (make ARC_WITH_ARCTAG=/path/to/arc-tagger-c)");
  return -1;
#endif
}

#ifdef ARC_WITH_ARCTAG
static void json_esc(sbuf* b, const char* s) {
  sb_put(b, "\"");
  for (; *s; ++s) {
    unsigned char c = (unsigned char)*s;
    char t[8];
    if (c == '"' || c == '\\') {
      t[0] = '\\';
      t[1] = (char)c;
      sb_putn(b, t, 2);
    } else if (c < 0x20) {
      snprintf(t, sizeof t, "\\u%04x", c);
      sb_put(b, t);
    } else
      sb_putn(b, (const char*)&c, 1);
  }
  sb_put(b, "\"");
}
#endif

/* the utterance's words -> arctag_tag_words (word objects with times and confidences, the utterance ends at the last) */
static char* tag_words(arc_nft2* m, const arc_utt_build* u) {
#ifdef ARC_WITH_ARCTAG
  if (!m->tagger || !u->n_aw) return NULL;
  sbuf b = {0};
  sb_put(&b, "[");
  char t[96];
  for (int i = 0; i < u->n_aw; ++i) {
    if (i) sb_put(&b, ", ");
    sb_put(&b, "{\"w\": ");
    json_esc(&b, u->aw[i].w);
    snprintf(t, sizeof t, ", \"t0\": %.3f, \"t1\": %.3f", u->aw[i].start, u->aw[i].end);
    sb_put(&b, t);
    if (u->aw[i].has_conf) {
      snprintf(t, sizeof t, ", \"conf\": %.4f", u->aw[i].conf);
      sb_put(&b, t);
    }
    sb_put(&b, i == u->n_aw - 1 ? ", \"final\": true, \"eou\": true}" : ", \"final\": true}");
  }
  sb_put(&b, "]");
  char* in = sb_take(&b);
  char* out = arctag_tag_words((arctag_t*)m->tagger, in);
  arc_free(in);
  if (!out) return NULL;
  char* r = arc_strdup(out);
  arctag_free_str(out);
  return r;
#else
  (void)m, (void)u;
  return NULL;
#endif
}

/* ------------------------------------------------------------------ n-best (an RNNT beam for alternatives) */

static void nbest_alts(arc_nft2* m, const float* H, int T, int bias, rword* w, int nw, arc_utt_build* u);

/* ------------------------------------------------------------------ one utterance (Nft2Finals.decode) */

/* y [len]: the audio the encoder reads; n_dec: the utterance's own length (without a stream's tail), which sets the
 * decoder's length limit as tr._gen(..., len(y)) does */
static int decode_utt(arc_nft2* m, const float* y, long len, long n_dec, double t0, int f0, int f1, int bias,
                      int guard, arc_utt_build* u) {
  arc_engine* e = m->e;
  int ui = m->dump_utt++;
  const float *H, *Ht;
  int T;
  if (encode(m, y, len, &H, &Ht, &T)) return -1;
  int D = m->d.d_model;
  /* RNNT words */
  int *ids, *frs;
  float* tst = NULL;
  int nid = rnnt_greedy(m, H, T, bias && m->trie, &ids, &frs, &tst);
  if (nid < 0) {
    arc_free(ids);
    arc_free(frs);
    arc_free(tst);
    return -1;
  }
  dumpi(m, "rnnt_ids", ui, ids, nid);
  dumpi(m, "rnnt_frames", ui, frs, nid);
  rword* w;
  int nw = rnnt_words(m, ids, frs, nid, t0, tst, &w);
  arc_free(ids);
  arc_free(frs);
  arc_free(tst);
  /* nft2_post.guard: no-speech gate, low-confidence repeats and tails (on the utterance's own audio) */
  int n_dropped = 0;
  if (guard && nw) {
    const char** tx = arc_malloc(sizeof(char*) * (size_t)nw);
    double* st = arc_malloc(sizeof(double) * (size_t)nw);
    float* cf = arc_malloc(sizeof(float) * (size_t)nw);
    int* keep = arc_malloc(sizeof(int) * (size_t)nw);
    for (int i = 0; i < nw; ++i) {
      tx[i] = w[i].text;
      st[i] = w[i].start;
      cf[i] = w[i].has_conf ? w[i].conf : -1.0f;
    }
    arc_post_guard(tx, st, cf, nw, y, n_dec < len ? n_dec : len, t0, &m->guard_cfg, keep, NULL);
    int k = 0;
    for (int i = 0; i < nw; ++i) {
      if (keep[i]) w[k++] = w[i];
      else {
        arc_free(w[i].text);
        arc_free(w[i].alt);
        n_dropped++;
      }
    }
    nw = k;
    arc_free(tx);
    arc_free(st);
    arc_free(cf);
    arc_free(keep);
  }
  /* the Arc decoder on the tap memory (drop0) with the context prompt */
  int nv = T - 1 > 1 ? T - 1 : 1;
  float* Mm = mem_with_pos(m, H + (T > 1 ? D : 0), nv);
  dumpf(m, "M", ui, Mm, nv, ARC_MOON_D);
  int P[512];
  int np = prompt_ids(m, f0 / 10.0, P, 512);
  dumpi(m, "prompt", ui, P, np);
  int ml = 1 + (int)(e->len_mult * (e->len_base + e->tokens_per_s * (double)n_dec / SR));
  int max_len = ml + np > 2 + np ? ml + np : 2 + np;
  enum { CAP = 2048 };
  int* dids = arc_malloc(sizeof(int) * CAP);
  arc_moon_procs pr = e->procs;
  pr.bias = NULL;
  int nd = arc_moon_decode(e->moon, Mm, nv, P, np, max_len, 1, &pr, dids, CAP);
  arc_free(Mm);
  if (nd < 0) {
    arc_free(dids);
    for (int i = 0; i < nw; ++i) arc_free(w[i].text);
    arc_free(w);
    return -1;
  }
  dumpi(m, "dec_ids", ui, dids, nd);
  char* raw = arc_tok_decode(e->tok, dids, nd);
  char* txt = arc_loopfix(raw, 2);
  /* tagged_spans(raw): readout -> raw words; then onto the loop-fixed words */
  char** rw;
  int nrw;
  arc_span* sp0;
  int nsp0;
  arc_readout(e->tok, dids, nd, &rw, &nrw, &sp0, &nsp0);
  arc_free(dids);
  char** raww;
  int nraw = arc_split_words(raw, &raww);
  arc_span* s1 = arc_malloc(sizeof(arc_span) * (size_t)(nsp0 + 1));
  int n1 = arc_map_spans(rw, nrw, sp0, nsp0, raww, nraw, s1);
  char** txw;
  int ntx = arc_split_words(txt, &txw);
  arc_span* s2 = arc_malloc(sizeof(arc_span) * (size_t)(n1 + 1));
  int n2 = n1;
  if (n1 && strcmp(txt, raw)) n2 = arc_map_spans(raww, nraw, s1, n1, txw, ntx, s2);
  else
    memcpy(s2, s1, sizeof(arc_span) * (size_t)n1);
  if (txt[0]) arc_hist_add(e, f0 / 10.0, (f1 + 1) / 10.0, txt);
  /* decoder spans -> RNNT words; one type per surface; the names list */
  char** dst = arc_malloc(sizeof(char*) * (size_t)(nw + 1));
  for (int i = 0; i < nw; ++i) dst[i] = w[i].text;
  charalign ca;
  char_align(&ca, txw, ntx, dst, nw);
  span3* wsp = arc_malloc(sizeof(span3) * (size_t)(n2 + m->n_lk + nw + 1));
  span3* src = arc_malloc(sizeof(span3) * (size_t)(n2 + 1));
  for (int i = 0; i < n2; ++i) src[i] = (span3){s2[i].tag, s2[i].a, s2[i].b};
  int nws = n2 ? map_spans_char(&ca, src, n2, wsp) : 0;
  if (m->unify) unify_types(wsp, nws, ca.bk, ca.bkn);
  if (m->n_lk && m->list_on) nws = list_spans(m, wsp, nws, n2 + m->n_lk + nw, &ca);
  char_align_free(&ca);
  arc_free(dst);
  arc_free(src);
  if (m->case_from && txt[0]) case_from(w, nw, txw, ntx);
  /* annotate */
  for (int s = 0; s < nws; ++s)
    for (int j = wsp[s].a > 0 ? wsp[s].a : 0; j < wsp[s].b && j < nw; ++j) {
      w[j].tag = wsp[s].tag;
      w[j].tag_begin = j == wsp[s].a;
    }
  if (m->cur_nbest > 1) nbest_alts(m, H, T, bias && m->trie, w, nw, u);
  u->f0 = f0;
  u->f1 = f1;
  u->start = (float)arc_pyround(f0 / 10.0, 1);
  u->end = (float)arc_pyround((f1 + 1) / 10.0, 1);
  sbuf tb = {0};
  sb_put(&tb, "");
  u->aw = arc_calloc(sizeof(arc_wordb) * (size_t)(nw + 1));
  u->words = arc_malloc(sizeof(char*) * (size_t)(nw + 1));
  double csum = 0;
  int ncs = 0;
  for (int i = 0; i < nw; ++i) {
    if (i) sb_put(&tb, " ");
    sb_put(&tb, w[i].text);
    u->aw[i] = (arc_wordb){.w = arc_strdup(w[i].text), .start = w[i].start, .end = w[i].end, .tag = w[i].tag, .tag_begin = w[i].tag_begin};
    u->aw[i].has_conf = w[i].has_conf;
    u->aw[i].conf = w[i].conf;
    if (w[i].has_conf) csum += w[i].conf, ncs++;
    if (w[i].alt) {   /* the runner-up spelling first, then the n-best's (nbest_alts) */
      int na = 1 + (w[i].n_xalt > 0 ? w[i].n_xalt : 0);
      u->aw[i].alts = arc_malloc(sizeof(char*) * (size_t)na);
      u->aw[i].alt_p = arc_malloc(sizeof(float) * (size_t)na);
      u->aw[i].alts[0] = w[i].alt;
      u->aw[i].alt_p[0] = w[i].alt_p;
      u->aw[i].n_alts = 1;
    } else if (w[i].n_xalt > 0) {
      u->aw[i].alts = arc_malloc(sizeof(char*) * (size_t)w[i].n_xalt);
      u->aw[i].alt_p = arc_malloc(sizeof(float) * (size_t)w[i].n_xalt);
    }
    for (int q = 0; q < w[i].n_xalt; ++q) {
      u->aw[i].alts[u->aw[i].n_alts] = w[i].xalt[q];
      u->aw[i].alt_p[u->aw[i].n_alts++] = w[i].xalt_p[q];
    }
    arc_free(w[i].xalt);
    arc_free(w[i].xalt_p);
    u->words[i] = w[i].text;
  }
  u->n_aw = nw;
  u->n_words = nw;
  u->text = sb_take(&tb);
  /* Nft2Finals._post: confidence, disfluency tags + clean_text, quantities */
  u->has_post = 1;
  u->conf = ncs ? (float)arc_pyround(csum / ncs, 4) : -1.0f;
  u->n_dropped = n_dropped;
  u->leak = -1.0f;
  {
    const char** tx = arc_malloc(sizeof(char*) * (size_t)(nw + 1));
    const char** tg = arc_malloc(sizeof(char*) * (size_t)(nw + 1));
    const char** df = arc_malloc(sizeof(char*) * (size_t)(nw + 1));
    for (int i = 0; i < nw; ++i) {
      tx[i] = u->aw[i].w;
      tg[i] = u->aw[i].tag >= 0 ? arc_tag_name(u->aw[i].tag) : NULL;
    }
    arc_post_disfluency(tx, nw, df);
    sbuf cb = {0};
    sb_put(&cb, "");
    int first = 1;
    for (int i = 0; i < nw; ++i) {
      u->aw[i].disfl = df[i];
      if (df[i]) continue;
      if (!first) sb_put(&cb, " ");
      sb_put(&cb, tx[i]);
      first = 0;
    }
    u->clean_text = sb_take(&cb);
    u->n_qty = arc_post_quantities(tx, tg, nw, &u->qty);
    u->tagger_json = tag_words(m, u);
    arc_free(tx);
    arc_free(tg);
    arc_free(df);
  }
  u->spans = arc_malloc(sizeof(arc_span) * (size_t)(nws + 1));
  for (int s = 0; s < nws; ++s) u->spans[s] = (arc_span){wsp[s].tag, wsp[s].a, wsp[s].b};
  u->n_spans = nws;
  u->decoder_text = txt;
  arc_free(w);
  arc_free(wsp);
  arc_free(raw);
  arc_free(s1);
  arc_free(s2);
  arc_free(sp0);
  arc_free_words(rw, nrw);
  arc_free_words(raww, nraw);
  arc_free_words(txw, ntx);
  return 0;
}

/* ------------------------------------------------------------------ whole clips */

typedef struct {
  int f0, f1;
  long a, b;   /* samples decoded */
} seg_t;

/* Nft2Finals.transcribe_offline's utterances: the whole clip up to whole_max_s, else the segmenter's with the pad */
static int segments(arc_nft2* m, const float* pcm, size_t n, seg_t** out) {
  int cap = 16, ns = 0;
  seg_t* s = arc_malloc(sizeof(seg_t) * (size_t)cap);
  if ((double)n <= m->whole_max_s * SR) {
    if (n >= 800) s[ns++] = (seg_t){0, (int)((n + FR - 1) / FR) - 1 > 0 ? (int)((n + FR - 1) / FR) - 1 : 0, 0, (long)n};
    *out = s;
    return ns;
  }
  arc_seg* sg = arc_seg_new(&m->e->seg);
  int nf = (int)((n + FR - 1) / FR), f0, f1;
  const char* why;
  long pad = (long)(m->edge_pad_s * SR);
  for (int f = 0; f <= nf; ++f) {
    int got;
    if (f < nf) {
      size_t a = (size_t)f * FR, b = a + FR < n ? a + FR : n;
      got = arc_seg_feed(sg, f, arc_frame_db(pcm + a, (int)(b - a)), &f0, &f1, &why);
    } else
      got = arc_seg_flush(sg, &f0, &f1, &why);
    if (!got) continue;
    if (ns == cap) {
      cap *= 2;
      seg_t* ns2 = arc_malloc(sizeof(seg_t) * (size_t)cap);
      memcpy(ns2, s, sizeof(seg_t) * (size_t)ns);
      arc_free(s);
      s = ns2;
    }
    long a = (long)f0 * FR - pad, b = (long)(f1 + 1) * FR + pad;
    s[ns++] = (seg_t){f0, f1, a > 0 ? a : 0, b < (long)n ? b : (long)n};
  }
  arc_seg_free(sg);
  *out = s;
  return ns;
}

static const char* opt_domain(const arc_nft2_options* o) { return o && o->domain ? o->domain : "assistant"; }
static int opt_guard(const arc_nft2* m, const arc_nft2_options* o) { return m->guard_on && !(o && o->no_guard); }
static int opt_nbest(const arc_nft2* m, const arc_nft2_options* o) { return o && o->nbest ? o->nbest : m->nbest; }
static int opt_almost(const arc_nft2* m, const arc_nft2_options* o) { return (o && o->almost) || m->almost_on; }

static int split_ws(const char* s, char*** out);
static int cmp_str(const void* a, const void* b) { return strcmp(*(char* const*)a, *(char* const*)b); }

/* nft2_post.common_words: the vocabulary's whole-word pieces, lower case, sorted (built once) */
static void common_build(arc_nft2* m) {
  if (m->common) return;
  m->common = arc_malloc(sizeof(char*) * (size_t)(m->n_piece + 1));
  m->n_common = 0;
  for (int i = 0; i < m->n_piece; ++i) {
    const char* p = m->piece[i];
    if (p[0] != 1 || !p[1]) continue;
    char* x = arc_strdup(p + 1);
    for (char* c = x; *c; ++c) *c = (char)tolower((unsigned char)*c);
    m->common[m->n_common++] = x;
  }
  qsort(m->common, (size_t)m->n_common, sizeof(char*), cmp_str);
}

static int is_common(arc_nft2* m, const char* word) {
  common_build(m);
  char* k = arc_post_wk(word);
  int r = k[0] && bsearch(&k, m->common, (size_t)m->n_common, sizeof(char*), cmp_str) != NULL;
  arc_free(k);
  return r;
}

/* nft2_post.almost_names: unlinked heard words that sound like a list entry -> state "almost" (candidate + p); an
 * abstain / not_in_list record that sounds like one becomes "almost". Linked records are untouched. */
static void almost_names(arc_nft2* m, arc_result_build* R) {
  if (!m->n_kb) return;
  int nf = 0;
  for (int i = 0; i < m->n_kb; ++i) nf += 1 + m->kb[i].n_aliases;
  arc_form* fm = arc_malloc(sizeof(arc_form) * (size_t)(nf + 1));
  nf = 0;
  for (int i = 0; i < m->n_kb; ++i) {
    if (!m->kb[i].name[0]) continue;
    fm[nf++] = (arc_form){m->kb[i].name, i};
    for (int a = 0; a < m->kb[i].n_aliases; ++a) {
      const char* al = m->kb[i].aliases[a];
      int blank = 1;
      for (const char* c = al; *c; ++c) blank &= isspace((unsigned char)*c) != 0;
      if (!blank) fm[nf++] = (arc_form){al, i};
    }
  }
  const arc_almost_cfg* c = &m->almost_cfg;
  for (int k = 0; k < R->n_names; ++k) {
    arc_name* r = &R->names[k];
    if (!r->state || !r->heard || (strcmp(r->state, "abstain") && strcmp(r->state, "not_in_list"))) continue;
    /* not only stop / common words (_all_common), and the words under it unsure (_unsure) */
    {
      char** hw;
      int nhw = split_ws(r->heard, &hw), allc = 1;
      for (int j = 0; j < nhw; ++j) {
        char* x = arc_post_wk(hw[j]);
        if (x[0] && !arc_post_is_stop(x) && !is_common(m, hw[j])) allc = 0;
        arc_free(x);
        arc_free(hw[j]);
      }
      arc_free(hw);
      if (allc) continue;
      int nconf = 0;
      double minc = 1e9;
      for (int i = 0; i < R->n; ++i)
        for (int j = 0; j < R->u[i].n_aw; ++j) {
          const arc_wordb* w = &R->u[i].aw[j];
          if (!(w->start < r->end && w->end > r->start) || !w->has_conf) continue;
          char* x = arc_post_wk(w->w);
          if (!arc_post_is_stop(x)) {
            nconf++;
            if (w->conf < minc) minc = w->conf;
          }
          arc_free(x);
        }
      if (nconf && !(minc < c->max_conf)) continue;
    }
    double p;
    int bi = arc_almost_best(r->heard, fm, nf, c, &p);
    if (bi >= 0 && p >= c->min_p) {
      r->state = "almost";
      r->candidate = m->kb[fm[bi].kb].name;
      r->candidate_kb_index = fm[bi].kb;
      r->candidate_p = (float)arc_pyround(p, 4);
    }
  }
  int cap = R->n_names + 8, na = R->n_names;
  arc_name* all = arc_arena_alloc(R->arena, sizeof(arc_name) * (size_t)cap);
  memcpy(all, R->names, sizeof(arc_name) * (size_t)R->n_names);
  for (int i = 0; i < R->n; ++i) {
    arc_utt_build* u = &R->u[i];
    int n = u->n_aw;
    if (!n) continue;
    const char** tx = arc_malloc(sizeof(char*) * (size_t)n);
    int* taken = arc_calloc(sizeof(int) * (size_t)n);
    float* cf = arc_malloc(sizeof(float) * (size_t)n);
    int* cm = arc_malloc(sizeof(int) * (size_t)n);
    for (int j = 0; j < n; ++j) {
      tx[j] = u->aw[j].w;
      cf[j] = u->aw[j].has_conf ? u->aw[j].conf : -1.0f;
      cm[j] = is_common(m, u->aw[j].w);
      for (int k = 0; k < R->n_names; ++k) {
        const arc_name* r = &R->names[k];
        if (r->state && (!strcmp(r->state, "linked") || !strcmp(r->state, "almost")) && u->aw[j].start < r->end &&
            u->aw[j].end > r->start)
          taken[j] = 1;
      }
    }
    arc_almost_hit* h;
    int nh = arc_post_almost(tx, cf, cm, n, taken, fm, nf, c, &h);
    for (int q = 0; q < nh; ++q) {
      if (na == cap) {
        cap *= 2;
        arc_name* nn = arc_arena_alloc(R->arena, sizeof(arc_name) * (size_t)cap);
        memcpy(nn, all, sizeof(arc_name) * (size_t)na);
        all = nn;
      }
      arc_name* o = &all[na++];
      memset(o, 0, sizeof *o);
      int a = h[q].word_begin, b = h[q].word_end;
      o->start = (float)u->aw[a].start;
      o->end = (float)u->aw[b - 1].end;
      o->tag = "entity.mention.place";
      sbuf hb = {0};
      sb_put(&hb, "");
      for (int j = a; j < b; ++j) {
        if (j > a) sb_put(&hb, " ");
        sb_put(&hb, u->aw[j].w);
      }
      char* heard = sb_take(&hb);
      o->heard = arc_arena_strdup(R->arena, heard);
      arc_free(heard);
      o->entity = NULL;
      o->kb_index = -1;
      o->state = "almost";
      o->p = (float)h[q].p;
      o->model_entity = NULL;
      o->model_p = -1;
      o->src = "sound_alike";
      o->via = "sound_alike";
      o->snap_score = -1;
      o->span_p = -1;
      o->show = 1;
      arc_word* ws = arc_arena_alloc(R->arena, sizeof(arc_word) * (size_t)(b - a + 1));
      for (int j = a; j < b; ++j) {
        arc_word* x = &ws[j - a];
        x->text = arc_arena_strdup(R->arena, u->aw[j].w);
        x->start = (float)u->aw[j].start;
        x->end = (float)u->aw[j].end;
        x->tag = u->aw[j].tag >= 0 ? arc_tag_name(u->aw[j].tag) : NULL;
        x->tag_begin = u->aw[j].tag_begin;
        x->conf = u->aw[j].has_conf ? u->aw[j].conf : -1.0f;
        x->disfl = u->aw[j].disfl;
      }
      o->words = ws;
      o->n_words = b - a;
      o->candidate = m->kb[h[q].kb].name;
      o->candidate_kb_index = h[q].kb;
      o->candidate_p = (float)h[q].p;
    }
    arc_free(h);
    arc_free(tx);
    arc_free(taken);
    arc_free(cf);
    arc_free(cm);
  }
  /* sorted by (start, end), stable */
  for (int i = 1; i < na; ++i) {
    arc_name t = all[i];
    int j = i - 1;
    while (j >= 0 && (all[j].start > t.start || (all[j].start == t.start && all[j].end > t.end))) {
      all[j + 1] = all[j];
      j--;
    }
    all[j + 1] = t;
  }
  R->names = all;
  R->n_names = na;
  arc_free(fm);
}

/* the engine over the clip with the utterances' words: names, numbers, events, device-directed head */
static int run_engine(arc_nft2* m, const float* pcm, size_t n, const arc_nft2_options* o, arc_result_build* R) {
  float *X, *Xt;
  int N;
  if (arc_feats(m, pcm, (long)n, &X, &Xt, &N)) return -1;
  dumpf(m, "X", 0, X, N, DIN);
  dumpf(m, "Xt", 0, Xt, N, DIN);
  int rc = 0;
  if (N > 0) {
    float* Xh = arc_malloc(sizeof(float) * (size_t)N * DIN);   /* x: via float16 (the trunk's input) */
    for (size_t i = 0; i < (size_t)N * DIN; ++i) Xh[i] = arc_round_half(X[i]);
    arc_options eo = {0};
    eo.domain = opt_domain(o);
    eo.events = !(o && o->no_events);
    eo.names = 1;
    eo.numbers = 1;
    rc = arc_engine_run_feats(m->e, Xh, X, eo.events ? Xt : NULL, N, N, &eo, R);
    arc_free(Xh);
    if (!rc && opt_almost(m, o)) almost_names(m, R);
    float thr = (float)arc_json_num(arc_json_get(arc_json_get(m->e->cfg, "device_directed"), "threshold"), 0.5);
    for (int i = 0; i < R->n; ++i)
      if (R->u[i].dir_head >= 0) {
        R->u[i].dir_head = (float)arc_pyround(R->u[i].dir_head, 4);
        R->u[i].directed = R->u[i].dir_head >= thr;
      }
  }
  arc_free(X);
  arc_free(Xt);
  return rc;
}

arc_result* arc_nft2_transcribe(arc_nft2* m, const float* pcm, size_t n, const arc_nft2_options* o) {
  double t0 = arc_time_ms();
  g_err[0] = 0;
  arc_seterr("%s", "");
  if (!m || (!pcm && n)) {
    arc_seterr("arc_nft2_transcribe: bad arguments");
    return NULL;
  }
  m->dump_enc = m->dump_utt = 0;
  m->cur_nbest = opt_nbest(m, o);
  seg_t* sg;
  int ns = segments(m, pcm, n, &sg);
  arc_result_build* R = arc_result_new(ns);
  int bias = !(o && o->no_bias);
  for (int i = 0; i < ns; ++i)
    if (decode_utt(m, pcm + sg[i].a, sg[i].b - sg[i].a, sg[i].b - sg[i].a, sg[i].a / (double)SR, sg[i].f0, sg[i].f1, bias,
                   opt_guard(m, o), &R->u[i])) {
      arc_free(sg);
      arc_result_free(arc_result_finish(R, 0, 0));
      return NULL;
    }
  arc_free(sg);
  if (n >= 80 && run_engine(m, pcm, n, o, R)) {
    arc_result_free(arc_result_finish(R, 0, 0));
    return NULL;
  }
  return arc_result_finish(R, (float)((double)n / SR), (float)(arc_time_ms() - t0));
}

/* ------------------------------------------------------------------ streams */

/* live decoding of the open utterance (nft2_live.Live): stable log-mel frames -> encoder chunks with carried caches ->
 * the greedy RNNT with carried state; nothing is decoded twice */
typedef struct {
  int on;
  long a;                  /* its first sample */
  int utt;                 /* the index it will get */
  int nmel;                /* log-mel frames fed to the encoder */
  float* mel;              /* pending stable frames [nmp][80] */
  int nmp, capmp;
  float* pre;              /* the encoder's pre-encode cache frames */
  arc_nft2_cache* c;
  int t_enc;               /* encoder frames decoded */
  float *g, *hc;
  int bs;
  int *ids, *frs;
  float* ts;
  int n, cap;
  int sent_words;          /* words of the last PARTIAL */
  char* sent_text;
  char** ent;              /* ENTITY keys already sent */
  int nent, capent;
  int barge;               /* BARGE_IN sent */
} live_t;

struct arc_nft2_stream {
  arc_nft2* m;
  arc_nft2_options o;
  char domain[64];
  float* pcm;
  size_t n, cap;
  arc_seg* seg;
  int nf;          /* frames fed to the segmenter */
  seg_t* utt;      /* closed utterances */
  arc_utt_build* ub;
  int nu, capu, sent;
  int n_out;       /* utterances closed (incl. dropped ones): the next one's index */
  live_t lv;
  arc_echo* echo;
  arc_nft2_event* ev;
  int nev, capev;
  char** evs;      /* strings of the events */
  int nevs, capevs;
};

static void ev_clear(arc_nft2_stream* s);
static void live_reset(live_t* L);
static int live_step(arc_nft2_stream* s);
static void ev_final(arc_nft2_stream* s, const arc_utt_build* u, int idx, const char* kind);

arc_nft2_stream* arc_nft2_stream_open(arc_nft2* m, const arc_nft2_options* o) {
  if (!m) return NULL;
  arc_nft2_stream* s = arc_calloc(sizeof *s);
  s->m = m;
  if (o) s->o = *o;
  snprintf(s->domain, sizeof s->domain, "%s", opt_domain(o));
  s->o.domain = s->domain;
  s->seg = arc_seg_new(&m->e->seg);
  m->dump_enc = m->dump_utt = 0;
  return s;
}

static void stream_utt_free(arc_utt_build* u) {
  if (!u->aw && !u->text) return;
  arc_utt_post_free(u);
  for (int k = 0; k < u->n_alts; ++k) arc_free(u->alts[k]);
  arc_free(u->alts);
  arc_free(u->alt_scores);
  arc_free(u->text);
  arc_free(u->decoder_text);
  arc_free(u->spans);
  arc_free_words(u->words, u->n_words);
  for (int k = 0; k < u->n_aw; ++k) arc_free(u->aw[k].w);
  arc_free(u->aw);
}

void arc_nft2_stream_close(arc_nft2_stream* s) {
  if (!s) return;
  ev_clear(s);
  arc_free(s->ev);
  arc_free(s->evs);
  live_reset(&s->lv);
  arc_echo_free(s->echo);
  for (int i = 0; i < s->nu; ++i) stream_utt_free(&s->ub[i]);
  arc_free(s->ub);
  arc_free(s->utt);
  arc_free(s->pcm);
  arc_seg_free(s->seg);
  arc_free(s);
}

/* a deep copy of an utterance build (the stream keeps its own; the result takes one) */
static void utt_copy(arc_utt_build* d, const arc_utt_build* u) {
  *d = *u;
  d->text = arc_strdup(u->text);
  d->decoder_text = arc_strdup(u->decoder_text);
  d->spans = arc_malloc(sizeof(arc_span) * (size_t)(u->n_spans + 1));
  memcpy(d->spans, u->spans, sizeof(arc_span) * (size_t)u->n_spans);
  d->words = arc_malloc(sizeof(char*) * (size_t)(u->n_words + 1));
  for (int k = 0; k < u->n_words; ++k) d->words[k] = arc_strdup(u->words[k]);
  d->aw = arc_malloc(sizeof(arc_wordb) * (size_t)(u->n_aw + 1));
  for (int k = 0; k < u->n_aw; ++k) {
    d->aw[k] = u->aw[k];
    d->aw[k].w = arc_strdup(u->aw[k].w);
    d->aw[k].alts = arc_malloc(sizeof(char*) * (size_t)(u->aw[k].n_alts + 1));
    d->aw[k].alt_p = arc_malloc(sizeof(float) * (size_t)(u->aw[k].n_alts + 1));
    for (int j = 0; j < u->aw[k].n_alts; ++j) {
      d->aw[k].alts[j] = arc_strdup(u->aw[k].alts[j]);
      d->aw[k].alt_p[j] = u->aw[k].alt_p[j];
    }
  }
  d->alts = arc_malloc(sizeof(char*) * (size_t)(u->n_alts + 1));
  d->alt_scores = arc_malloc(sizeof(float) * (size_t)(u->n_alts + 1));
  for (int k = 0; k < u->n_alts; ++k) {
    d->alts[k] = arc_strdup(u->alts[k]);
    d->alt_scores[k] = u->alt_scores[k];
  }
  d->clean_text = u->clean_text ? arc_strdup(u->clean_text) : NULL;
  d->tagger_json = u->tagger_json ? arc_strdup(u->tagger_json) : NULL;
  d->qty = arc_malloc(sizeof(arc_qty) * (size_t)(u->n_qty + 1));
  for (int k = 0; k < u->n_qty; ++k) {
    d->qty[k] = u->qty[k];
    d->qty[k].text = arc_strdup(u->qty[k].text);
    d->qty[k].written = arc_strdup(u->qty[k].written);
  }
}

static void stream_utt_free(arc_utt_build* u);

static int stream_close_utt(arc_nft2_stream* s, int f0, int f1) {
  arc_nft2* m = s->m;
  long pad = (long)(m->edge_pad_s * SR);
  long a = (long)f0 * FR - pad, b = (long)(f1 + 1) * FR + pad;
  if (a < 0) a = 0;
  if (b > (long)s->n) b = (long)s->n;
  if (s->nu == s->capu) {
    s->capu = s->capu ? s->capu * 2 : 8;
    seg_t* nu = arc_malloc(sizeof(seg_t) * (size_t)s->capu);
    arc_utt_build* nb = arc_calloc(sizeof(arc_utt_build) * (size_t)s->capu);
    if (s->nu) {
      memcpy(nu, s->utt, sizeof(seg_t) * (size_t)s->nu);
      memcpy(nb, s->ub, sizeof(arc_utt_build) * (size_t)s->nu);
    }
    arc_free(s->utt);
    arc_free(s->ub);
    s->utt = nu;
    s->ub = nb;
  }
  arc_utt_build* u = &s->ub[s->nu];
  memset(u, 0, sizeof *u);
  u->has_post = 0;
  u->leak = -1.0f;
  u->dd = u->cc = u->dir_head = u->intent_p = u->act_p = -1.0f;
  u->directed = -1;
  if (b - a < 800) {   /* too short to decode */
    live_reset(&s->lv);
    return 0;
  }
  /* Nft2Finals._tail: silence after the cut so the input runs tail_pad_s past the last frame (the RNNT emits a word's
   * last pieces a few hundred ms after it is spoken); offline decoding has no tail */
  long need = (long)nearbyint((((f1 + 1) / 10.0) + m->tail_pad_s - a / (double)SR) * SR) - (b - a);
  float* y = s->pcm + a;
  float* yp = NULL;
  if (need > 0) {
    yp = arc_calloc(sizeof(float) * (size_t)(b - a + need));
    memcpy(yp, y, sizeof(float) * (size_t)(b - a));
    y = yp;
  }
  float* yr = NULL;
  if (s->echo && s->o.echo_decode) {   /* the echo-cancelled audio (complete up to one frame, 32 ms, before the mic) */
    size_t nr;
    const float* r = arc_echo_residual(s->echo, &nr);
    yr = arc_calloc(sizeof(float) * (size_t)(b - a + (need > 0 ? need : 0)));
    for (long i = a; i < b; ++i) yr[i - a] = (size_t)i < nr ? r[i] : s->pcm[i];
    y = yr;
  } else if (need > 0) {
    yp = arc_calloc(sizeof(float) * (size_t)(b - a + need));
    memcpy(yp, y, sizeof(float) * (size_t)(b - a));
    y = yp;
  }
  m->cur_nbest = opt_nbest(m, &s->o);
  int rc = decode_utt(m, y, b - a + (need > 0 ? need : 0), b - a, a / (double)SR, f0, f1, !s->o.no_bias, opt_guard(m, &s->o), u);
  arc_free(yp);
  arc_free(yr);
  if (rc) return -1;
  int idx = s->n_out++;
  const char* kind = NULL;
  if (s->echo) {
    double lk = arc_echo_leak(s->echo, f0 / 10.0, (f1 + 1) / 10.0);
    u->leak = (float)lk;
    double thr = s->o.playback_threshold > 0 ? s->o.playback_threshold : arc_echo_threshold(s->echo);
    u->playback = lk >= thr;
    if (u->playback) kind = s->o.playback_drop ? "dropped" : "playback";
  }
  ev_final(s, u, idx, kind);
  live_reset(&s->lv);
  if (kind && !strcmp(kind, "dropped")) {
    stream_utt_free(u);
    memset(u, 0, sizeof *u);
    return 0;
  }
  s->utt[s->nu++] = (seg_t){f0, f1, a, b};
  return 0;
}

static arc_result* stream_new_utts(arc_nft2_stream* s) {
  arc_result_build* R = arc_result_new(s->nu - s->sent);
  for (int i = s->sent; i < s->nu; ++i) utt_copy(&R->u[i - s->sent], &s->ub[i]);
  s->sent = s->nu;
  return arc_result_finish(R, (float)((double)s->n / SR), 0);
}

arc_result* arc_nft2_stream_feed(arc_nft2_stream* s, const float* pcm, size_t n) {
  if (!s || (!pcm && n)) return NULL;
  arc_seterr("%s", "");
  if (s->n + n > s->cap) {
    size_t nc = (s->n + n) * 2 + SR;
    float* p = arc_malloc(sizeof(float) * nc);
    if (!p) return NULL;
    if (s->n) memcpy(p, s->pcm, sizeof(float) * s->n);
    arc_free(s->pcm);
    s->pcm = p;
    s->cap = nc;
  }
  memcpy(s->pcm + s->n, pcm, sizeof(float) * n);
  s->n += n;
  ev_clear(s);
  if (s->echo && arc_echo_feed_mic(s->echo, pcm, n)) return NULL;
  /* whole 100 ms frames to the segmenter; an utterance closes 0.4 s into a pause, so its 0.15 s pad is here */
  while ((size_t)(s->nf + 1) * FR <= s->n) {
    int f0, f1;
    const char* why;
    if (arc_seg_feed(s->seg, s->nf, arc_frame_db(s->pcm + (size_t)s->nf * FR, FR), &f0, &f1, &why) &&
        stream_close_utt(s, f0, f1))
      return NULL;
    s->nf++;
  }
  if (live_step(s)) return NULL;
  return stream_new_utts(s);
}

arc_result* arc_nft2_stream_flush(arc_nft2_stream* s) {
  if (!s) return NULL;
  double t0 = arc_time_ms();
  arc_seterr("%s", "");
  ev_clear(s);
  int f0, f1;
  const char* why;
  if (s->n > (size_t)s->nf * FR) {   /* the last partial frame */
    size_t a = (size_t)s->nf * FR;
    if (arc_seg_feed(s->seg, s->nf, arc_frame_db(s->pcm + a, (int)(s->n - a)), &f0, &f1, &why) &&
        stream_close_utt(s, f0, f1))
      return NULL;
    s->nf++;
  }
  if (arc_seg_flush(s->seg, &f0, &f1, &why) && stream_close_utt(s, f0, f1)) return NULL;
  live_reset(&s->lv);
  s->sent = s->nu;
  arc_result_build* R = arc_result_new(s->nu);
  for (int i = 0; i < s->nu; ++i) utt_copy(&R->u[i], &s->ub[i]);
  if (s->n >= 80 && run_engine(s->m, s->pcm, s->n, &s->o, R)) {
    arc_result_free(arc_result_finish(R, 0, 0));
    return NULL;
  }
  return arc_result_finish(R, (float)((double)s->n / SR), (float)(arc_time_ms() - t0));
}

/* ------------------------------------------------------------------ live decoding, events, echo (nft2_live.py) */

static char* ev_str(arc_nft2_stream* s, const char* x) {
  if (!x) return NULL;
  if (s->nevs == s->capevs) {
    s->capevs = s->capevs ? s->capevs * 2 : 16;
    char** p = arc_malloc(sizeof(char*) * (size_t)s->capevs);
    if (s->nevs) memcpy(p, s->evs, sizeof(char*) * (size_t)s->nevs);
    arc_free(s->evs);
    s->evs = p;
  }
  return s->evs[s->nevs++] = arc_strdup(x);
}

static arc_nft2_event* ev_push(arc_nft2_stream* s, int type, float t, int utt) {
  if (s->nev == s->capev) {
    s->capev = s->capev ? s->capev * 2 : 16;
    arc_nft2_event* p = arc_malloc(sizeof(arc_nft2_event) * (size_t)s->capev);
    if (s->nev) memcpy(p, s->ev, sizeof(arc_nft2_event) * (size_t)s->nev);
    arc_free(s->ev);
    s->ev = p;
  }
  arc_nft2_event* e = &s->ev[s->nev++];
  memset(e, 0, sizeof *e);
  e->type = type;
  e->t = t;
  e->utt = utt;
  e->kb_index = -1;
  e->p = -1.0f;
  return e;
}

static void ev_clear(arc_nft2_stream* s) {
  for (int i = 0; i < s->nevs; ++i) arc_free(s->evs[i]);
  s->nevs = 0;
  s->nev = 0;
}

const arc_nft2_event* arc_nft2_stream_events(arc_nft2_stream* s, int* n) {
  if (n) *n = s ? s->nev : 0;
  return s ? s->ev : NULL;
}

static void live_reset(live_t* L) {
  arc_free(L->mel);
  arc_free(L->pre);
  arc_nft2_be_cache_free(L->c);
  arc_free(L->g);
  arc_free(L->hc);
  arc_free(L->ids);
  arc_free(L->frs);
  arc_free(L->ts);
  arc_free(L->sent_text);
  for (int i = 0; i < L->nent; ++i) arc_free(L->ent[i]);
  arc_free(L->ent);
  memset(L, 0, sizeof *L);
}

static char* words_text(char* const* w, int n) {
  sbuf b = {0};
  sb_put(&b, "");
  for (int i = 0; i < n; ++i) {
    if (i) sb_put(&b, " ");
    sb_put(&b, w[i]);
  }
  return sb_take(&b);
}

/* nft2_post.eot_prob on an utterance's words */
static float eot_of(char* const* w, int n, double pause_s) { return (float)arc_post_eot((const char* const*)w, n, pause_s); }

static void ev_final(arc_nft2_stream* s, const arc_utt_build* u, int idx, const char* kind) {
  float t = (float)((double)s->n / SR);
  arc_nft2_event* e = ev_push(s, ARC_EV_FINAL, t, idx);
  e->text = ev_str(s, u->text ? u->text : "");
  e->kind = kind;
  e->p = u->leak;
  e->start = u->start;
  e->end = u->end;
  if (s->o.live) {
    char** w = arc_malloc(sizeof(char*) * (size_t)(u->n_aw + 1));
    for (int i = 0; i < u->n_aw; ++i) w[i] = u->aw[i].w;
    float p = eot_of(w, u->n_aw, s->m->e->seg.pause_s);
    arc_free(w);
    e = ev_push(s, ARC_EV_EOT, t, idx);
    e->p = (float)arc_pyround(p, 4);
  }
}

/* the live RNNT over new encoder frames H [k][D] (rnnt_greedy's loop with its state carried) */
static int live_rnnt(arc_nft2* m, live_t* L, const float* H, int k, int bias) {
  const arc_nft2_dims* d = &m->d;
  int J = d->joint_hidden, V1 = d->vocab + 1, blank = d->blank, rc = 0;
  boost_trie* tr = bias ? m->trie : NULL;
  const float al = (float)m->alpha;
  float* Fe = arc_malloc(sizeof(float) * (size_t)k * J);
  float* row = arc_malloc(sizeof(float) * (size_t)V1);
  float* biased = arc_malloc(sizeof(float) * (size_t)blank);
  if (arc_nft2_be_joint_enc(m->be, H, k, Fe)) rc = -1;
  for (int t = 0; !rc && t < k; ++t) {
    if (arc_nft2_be_joint(m->be, Fe + (size_t)t * J, 1, L->g, row)) {
      rc = -1;
      break;
    }
    for (int sym = 0;;) {
      int kk = argmax(row, V1);
      if (kk == blank) break;
      if (tr) {
        const float* sc = trie_scores(tr, L->bs);
        for (int q = 0; q < blank; ++q) biased[q] = row[q] + al * sc[q];
        kk = argmax(biased, blank);
        L->bs = trie_advance(tr, L->bs, kk);
      }
      if (L->n == L->cap) {
        L->cap = L->cap ? L->cap * 2 : 64;
        int* a = arc_malloc(sizeof(int) * (size_t)L->cap);
        int* b = arc_malloc(sizeof(int) * (size_t)L->cap);
        float* c = arc_malloc(sizeof(float) * 3 * (size_t)L->cap);
        if (L->n) {
          memcpy(a, L->ids, sizeof(int) * (size_t)L->n);
          memcpy(b, L->frs, sizeof(int) * (size_t)L->n);
          memcpy(c, L->ts, sizeof(float) * 3 * (size_t)L->n);
        }
        arc_free(L->ids);
        arc_free(L->frs);
        arc_free(L->ts);
        L->ids = a;
        L->frs = b;
        L->ts = c;
      }
      tok_stats(row, V1, blank, kk, L->ts + 3 * (size_t)L->n);
      L->ids[L->n] = kk;
      L->frs[L->n++] = L->t_enc + t;
      if (arc_nft2_be_pred(m->be, kk, L->hc, L->g)) {
        rc = -1;
        break;
      }
      if (++sym >= d->max_sym) break;
      if (arc_nft2_be_joint(m->be, Fe + (size_t)t * J, 1, L->g, row)) {
        rc = -1;
        break;
      }
    }
  }
  L->t_enc += k;
  arc_free(Fe);
  arc_free(row);
  arc_free(biased);
  return rc;
}

/* stable log-mel frames of the open utterance -> encoder chunks -> RNNT */
static int live_encode(arc_nft2_stream* s) {
  arc_nft2* m = s->m;
  live_t* L = &s->lv;
  const arc_nft2_dims* d = &m->d;
  long avail = (long)s->n - L->a;
  int stable = avail >= NFFT / 2 ? (int)((avail - NFFT / 2) / HOP) + 1 : 0;
  if (stable > (int)(avail / HOP)) stable = (int)(avail / HOP);
  int have = L->nmel + L->nmp;
  if (stable > have) {   /* frames [have, stable): log_mel over the samples from 2 frames before (no edge effect) */
    long st = have >= 2 ? (long)(have - 2) * HOP : 0;
    int off = have >= 2 ? 2 : have;
    int nf;
    float* M = log_mel(m, s->pcm + L->a + st, (int)(avail - st), &nf);
    int add = stable - have;
    if (off + add > nf) add = nf - off;
    if (add > 0) {
      if (L->nmp + add > L->capmp) {
        L->capmp = (L->nmp + add) * 2;
        float* q = arc_malloc(sizeof(float) * (size_t)L->capmp * 80);
        if (L->nmp) memcpy(q, L->mel, sizeof(float) * (size_t)L->nmp * 80);
        arc_free(L->mel);
        L->mel = q;
      }
      memcpy(L->mel + (size_t)L->nmp * 80, M + (size_t)off * 80, sizeof(float) * (size_t)add * 80);
      L->nmp += add;
    }
    arc_free(M);
  }
  int D = d->d_model, pre_n = d->pre;
  float* ob = NULL;
  float* xb = NULL;
  int rc = 0;
  for (;;) {
    int first = L->nmel == 0 && d->first > 0;
    int nn = first ? d->first : d->chunk;
    if (L->nmp < nn) break;
    if (!ob) {
      ob = arc_malloc(sizeof(float) * (size_t)64 * D);
      xb = arc_malloc(sizeof(float) * (size_t)(pre_n + d->chunk + d->first + 1) * 80);
    }
    int xn;
    if (first) {
      memcpy(xb, L->mel, sizeof(float) * (size_t)nn * 80);
      xn = nn;
    } else {
      memcpy(xb, L->pre, sizeof(float) * (size_t)pre_n * 80);
      memcpy(xb + (size_t)pre_n * 80, L->mel, sizeof(float) * (size_t)nn * 80);
      xn = pre_n + nn;
    }
    {
      int tot = pre_n + xn;
      float* cat = arc_malloc(sizeof(float) * (size_t)tot * 80);
      memcpy(cat, L->pre, sizeof(float) * (size_t)pre_n * 80);
      memcpy(cat + (size_t)pre_n * 80, xb, sizeof(float) * (size_t)xn * 80);
      memcpy(L->pre, cat + (size_t)(tot - pre_n) * 80, sizeof(float) * (size_t)pre_n * 80);
      arc_free(cat);
    }
    int k = 0;
    if (arc_nft2_be_chunk(m->be, L->c, first, xb, xn, ob, NULL, 64, &k)) {
      rc = -1;
      break;
    }
    memmove(L->mel, L->mel + (size_t)nn * 80, sizeof(float) * (size_t)(L->nmp - nn) * 80);
    L->nmp -= nn;
    L->nmel += nn;
    if (k > 0 && live_rnnt(m, L, ob, k, !s->o.no_bias)) {
      rc = -1;
      break;
    }
  }
  arc_free(ob);
  arc_free(xb);
  return rc;
}

static int ent_seen(live_t* L, const char* key) {
  for (int i = 0; i < L->nent; ++i)
    if (!strcmp(L->ent[i], key)) return 1;
  if (L->nent == L->capent) {
    L->capent = L->capent ? L->capent * 2 : 8;
    char** p = arc_malloc(sizeof(char*) * (size_t)L->capent);
    if (L->nent) memcpy(p, L->ent, sizeof(char*) * (size_t)L->nent);
    arc_free(L->ent);
    L->ent = p;
  }
  L->ent[L->nent++] = arc_strdup(key);
  return 0;
}

/* nft2_live.entities: list spellings (word keys), sound-alikes and complete quantities among the stable words */
static void live_entities(arc_nft2_stream* s, rword* w, int nst, float t) {
  arc_nft2* m = s->m;
  live_t* L = &s->lv;
  if (nst <= 0) return;
  char** wk = arc_malloc(sizeof(char*) * (size_t)nst);
  const char** tx = arc_malloc(sizeof(char*) * (size_t)nst);
  int* taken = arc_calloc(sizeof(int) * (size_t)nst);
  for (int i = 0; i < nst; ++i) {
    wk[i] = arc_post_wk(w[i].text);
    tx[i] = w[i].text;
  }
  char key[512];
  /* list spellings: the longest form at each position */
  for (int i = 0; i < nst; ++i) {
    int best_len = 0, best_kb = -1;
    for (int e = 0; e < m->n_kb; ++e)
      for (int a = -1; a < m->kb[e].n_aliases; ++a) {
        const char* f = a < 0 ? m->kb[e].name : m->kb[e].aliases[a];
        if (!f || !f[0]) continue;
        char* nm = arc_nft2_norm(f);
        char** fw;
        int nfw = arc_split_words(nm, &fw);
        int ok = nfw > 0 && i + nfw <= nst;
        for (int j = 0; ok && j < nfw; ++j) {
          char* x = arc_post_wk(fw[j]);
          ok = x[0] && !strcmp(x, wk[i + j]);
          arc_free(x);
        }
        if (ok && nfw > best_len) {
          best_len = nfw;
          best_kb = e;
        }
        arc_free_words(fw, nfw);
        arc_free(nm);
      }
    if (best_kb < 0) continue;
    for (int j = i; j < i + best_len; ++j) taken[j] = 1;
    snprintf(key, sizeof key, "name:%d:%d:%d", i, i + best_len, best_kb);
    if (!ent_seen(L, key)) {
      sbuf b = {0};
      sb_put(&b, "");
      for (int j = i; j < i + best_len; ++j) {
        if (j > i) sb_put(&b, " ");
        sb_put(&b, tx[j]);
      }
      char* heard = sb_take(&b);
      arc_nft2_event* e = ev_push(s, ARC_EV_ENTITY, t, L->utt);
      e->text = ev_str(s, heard);
      arc_free(heard);
      e->kind = "name";
      e->entity = ev_str(s, m->kb[best_kb].name);
      e->kb_index = best_kb;
      e->p = 1.0f;
      e->start = (float)w[i].start;
      e->end = (float)w[i + best_len - 1].end;
    }
    i += best_len - 1;
  }
  /* sound-alikes */
  if (m->n_kb) {
    int nf = 0;
    for (int e = 0; e < m->n_kb; ++e) nf += 1 + m->kb[e].n_aliases;
    arc_form* fm = arc_malloc(sizeof(arc_form) * (size_t)(nf + 1));
    nf = 0;
    for (int e = 0; e < m->n_kb; ++e) {
      if (!m->kb[e].name[0]) continue;
      fm[nf++] = (arc_form){m->kb[e].name, e};
      for (int a = 0; a < m->kb[e].n_aliases; ++a) {
        const char* al = m->kb[e].aliases[a];
        int blank = 1;
        for (const char* c = al; *c; ++c) blank &= isspace((unsigned char)*c) != 0;
        if (!blank) fm[nf++] = (arc_form){al, e};
      }
    }
    float* cf = arc_malloc(sizeof(float) * (size_t)nst);
    int* cm = arc_malloc(sizeof(int) * (size_t)nst);
    for (int i = 0; i < nst; ++i) {
      cf[i] = w[i].has_conf ? w[i].conf : -1.0f;
      cm[i] = is_common(m, w[i].text);
    }
    arc_almost_hit* h;
    int nh = arc_post_almost(tx, cf, cm, nst, taken, fm, nf, &m->almost_cfg, &h);
    arc_free(cf);
    arc_free(cm);
    for (int q = 0; q < nh; ++q) {
      for (int j = h[q].word_begin; j < h[q].word_end; ++j) taken[j] = 1;
      snprintf(key, sizeof key, "almost:%d:%d:%d", h[q].word_begin, h[q].word_end, h[q].kb);
      if (ent_seen(L, key)) continue;
      sbuf b = {0};
      sb_put(&b, "");
      for (int j = h[q].word_begin; j < h[q].word_end; ++j) {
        if (j > h[q].word_begin) sb_put(&b, " ");
        sb_put(&b, tx[j]);
      }
      char* heard = sb_take(&b);
      arc_nft2_event* e = ev_push(s, ARC_EV_ENTITY, t, L->utt);
      e->text = ev_str(s, heard);
      arc_free(heard);
      e->kind = "almost";
      e->entity = ev_str(s, m->kb[h[q].kb].name);
      e->kb_index = h[q].kb;
      e->p = (float)h[q].p;
      e->start = (float)w[h[q].word_begin].start;
      e->end = (float)w[h[q].word_end - 1].end;
    }
    arc_free(h);
    arc_free(fm);
  }
  /* quantities that end before the last stable word (a number may still grow) */
  arc_qty* qs;
  const char** tg = arc_calloc(sizeof(char*) * (size_t)nst);
  for (int i = 0; i < nst; ++i) tg[i] = taken[i] ? "ENTITY_NAME" : NULL;   /* not inside a name */
  int nq = arc_post_quantities(tx, tg, nst, &qs);
  arc_free(tg);
  for (int q = 0; q < nq; ++q) {
    if (qs[q].word_end >= nst) continue;
    snprintf(key, sizeof key, "qty:%d:%d", qs[q].word_begin, qs[q].word_end);
    if (ent_seen(L, key)) continue;
    arc_nft2_event* e = ev_push(s, ARC_EV_ENTITY, t, L->utt);
    e->text = ev_str(s, qs[q].text);
    e->kind = "quantity";
    e->entity = ev_str(s, qs[q].written);
    e->p = 1.0f;
    e->qtype = qs[q].type;
    e->value = qs[q].value;
    e->unit = qs[q].unit;
    e->relative = qs[q].relative;
    e->start = (float)w[qs[q].word_begin].start;
    e->end = (float)w[qs[q].word_end - 1].end;
  }
  arc_post_qty_free(qs, nq);
  for (int i = 0; i < nst; ++i) arc_free(wk[i]);
  arc_free(wk);
  arc_free(tx);
  arc_free(taken);
}

/* after the segmenter: start / advance live decoding of the open utterance; PARTIAL, ENTITY, EOT, BARGE_IN events */
static int live_step(arc_nft2_stream* s) {
  arc_nft2* m = s->m;
  live_t* L = &s->lv;
  int f_open = arc_seg_open(s->seg);
  float t = (float)((double)s->n / SR);
  if (f_open < 0) return 0;
  if (s->echo && !L->barge && arc_echo_barge_in(s->echo, f_open / 10.0, s->nf / 10.0, 0.2, 0.5)) {
    L->barge = 1;
    arc_nft2_event* e = ev_push(s, ARC_EV_BARGE_IN, t, s->n_out);
    double t0 = s->nf / 10.0 - 0.3 > f_open / 10.0 ? s->nf / 10.0 - 0.3 : f_open / 10.0;
    double lk = arc_echo_leak(s->echo, t0, s->nf / 10.0);   /* the window the rule read */
    e->p = (float)arc_pyround(1.0 - lk, 4);
    e->start = (float)(f_open / 10.0);
    e->end = (float)(s->nf / 10.0);
  }
  if (!s->o.live) return 0;
  if (!L->on) {
    int barge = L->barge;
    live_reset(L);
    L->barge = barge;
    L->on = 1;
    L->utt = s->n_out;
    long a = (long)f_open * FR - (long)(m->edge_pad_s * SR);
    L->a = a < 0 ? 0 : a;
    L->c = arc_nft2_be_cache_new(m->be);
    L->pre = arc_calloc(sizeof(float) * (size_t)(m->d.pre + 1) * 80);
    L->g = arc_malloc(sizeof(float) * (size_t)m->d.joint_hidden);
    L->hc = arc_calloc(sizeof(float) * (size_t)2 * m->d.pred_layers * m->d.pred_hidden);
    if (!L->c || arc_nft2_be_pred(m->be, m->d.blank, L->hc, L->g)) return -1;
  }
  int n0 = L->n;
  if (live_encode(s)) return -1;
  rword* w;
  int nw = rnnt_words(m, L->ids, L->frs, L->n, L->a / (double)SR, L->ts, &w);
  char** tx = arc_malloc(sizeof(char*) * (size_t)(nw + 1));
  for (int i = 0; i < nw; ++i) tx[i] = w[i].text;
  char* text = words_text(tx, nw);
  if (L->n != n0 && (!L->sent_text || strcmp(text, L->sent_text))) {
    arc_nft2_event* e = ev_push(s, ARC_EV_PARTIAL, t, L->utt);
    e->text = ev_str(s, text);
    e->n_stable = nw > 0 ? nw - 1 : 0;
    e->start = nw ? (float)w[0].start : 0.0f;
    e->end = nw ? (float)w[nw - 1].end : 0.0f;
    arc_free(L->sent_text);
    L->sent_text = arc_strdup(text);
    L->sent_words = nw;
    live_entities(s, w, nw - 1, t);
  }
  if (nw) {
    arc_nft2_event* e = ev_push(s, ARC_EV_EOT, t, L->utt);
    e->p = (float)arc_pyround(eot_of(tx, nw, arc_seg_quiet_run(s->seg) / 10.0), 4);
  }
  arc_free(text);
  arc_free(tx);
  for (int i = 0; i < nw; ++i) {
    arc_free(w[i].text);
    arc_free(w[i].alt);
  }
  arc_free(w);
  return 0;
}

int arc_nft2_stream_feed_ref(arc_nft2_stream* s, const float* pcm, size_t n, double t) {
  if (!s || (!pcm && n)) return -1;
  if (!s->echo) {
    s->echo = arc_echo_new(NULL);
    if (!s->echo || (s->n && arc_echo_feed_mic(s->echo, s->pcm, s->n))) return -1;
  }
  return arc_echo_feed_ref(s->echo, pcm, n, t);
}

int arc_nft2_stream_set_names(arc_nft2_stream* s, const arc_name_entry* names, int n) {
  if (!s || arc_nft2_set_names(s->m, names, n)) return -1;
  s->lv.bs = 0;   /* the open utterance's bias state restarts on the new list */
  return 0;
}

arc_result* arc_nft2_stream_end_turn(arc_nft2_stream* s) {
  if (!s) return NULL;
  arc_seterr("%s", "");
  ev_clear(s);
  int f0, f1;
  const char* why;
  if (arc_seg_flush(s->seg, &f0, &f1, &why) && stream_close_utt(s, f0, f1)) return NULL;
  live_reset(&s->lv);
  return stream_new_utts(s);
}

/* ------------------------------------------------------------------ n-best (rnnt_beam.py) */

typedef struct {
  int* ids;
  int n;
  double s;
  float *g, *hc;
  int b;
} bhyp;

static void bhyp_free(bhyp* h) {
  arc_free(h->ids);
  arc_free(h->g);
  arc_free(h->hc);
  memset(h, 0, sizeof *h);
}

/* Python tuple order on token ids */
static int ids_cmp(const bhyp* a, const bhyp* b) {
  int n = a->n < b->n ? a->n : b->n;
  for (int i = 0; i < n; ++i)
    if (a->ids[i] != b->ids[i]) return a->ids[i] < b->ids[i] ? -1 : 1;
  return (a->n > b->n) - (a->n < b->n);
}

static int bhyp_order(const void* x, const void* y) {
  const bhyp *a = x, *b = y;
  if (a->s != b->s) return a->s > b->s ? -1 : 1;
  return ids_cmp(a, b);
}

typedef struct {
  double s;
  int i, k;
} bcand;

static int bcand_order(const void* x, const void* y) {
  const bcand *a = x, *b = y;
  if (a->s != b->s) return a->s > b->s ? -1 : 1;
  if (a->i != b->i) return a->i < b->i ? -1 : 1;
  return (a->k > b->k) - (a->k < b->k);
}

/* rnnt_beam.beam: *out (malloc'd, bhyp_free each) best first; returns the count, -1 on error */
static int rnnt_beam(arc_nft2* m, const float* H, int T, int bias, int width, bhyp** out) {
  const arc_nft2_dims* d = &m->d;
  int J = d->joint_hidden, V1 = d->vocab + 1, blank = d->blank, max_sym = 2;
  size_t nhc = (size_t)2 * d->pred_layers * d->pred_hidden;
  *out = NULL;
  if (T <= 0 || width < 2) return 0;
  boost_trie* tr = bias ? m->trie : NULL;
  const float al = (float)m->alpha;
  float* Fe = arc_malloc(sizeof(float) * (size_t)T * J);
  float* row = arc_malloc(sizeof(float) * (size_t)V1);
  double* lp = arc_malloc(sizeof(double) * (size_t)V1);
  double* sc = arc_malloc(sizeof(double) * (size_t)blank);
  int* order = arc_malloc(sizeof(int) * (size_t)blank);
  bhyp* hy = arc_calloc(sizeof(bhyp) * (size_t)width);
  bhyp* A = arc_calloc(sizeof(bhyp) * (size_t)width);
  bhyp* done = arc_calloc(sizeof(bhyp) * (size_t)(width * (max_sym + 1) + 1));
  bcand* cand = arc_malloc(sizeof(bcand) * (size_t)width * width);
  int nh = 1, rc = 0;
  hy[0].ids = arc_malloc(sizeof(int));
  hy[0].g = arc_malloc(sizeof(float) * (size_t)J);
  hy[0].hc = arc_calloc(sizeof(float) * nhc);
  if (arc_nft2_be_pred(m->be, blank, hy[0].hc, hy[0].g) || arc_nft2_be_joint_enc(m->be, H, T, Fe)) rc = -1;
  for (int t = 0; !rc && t < T; ++t) {
    int na = nh, nd = 0;
    for (int i = 0; i < na; ++i) A[i] = hy[i];   /* A owns the beam now */
    memset(hy, 0, sizeof(bhyp) * (size_t)width);
    for (int step = 0; !rc && step <= max_sym && na; ++step) {
      int nc = 0;
      for (int i = 0; i < na && !rc; ++i) {
        bhyp* a = &A[i];
        if (arc_nft2_be_joint(m->be, Fe + (size_t)t * J, 1, a->g, row)) {
          rc = -1;
          break;
        }
        double mx = row[0], sum = 0;
        for (int k = 1; k < V1; ++k)
          if (row[k] > mx) mx = row[k];
        for (int k = 0; k < V1; ++k) sum += exp((double)row[k] - mx);
        double lse = mx + log(sum);
        for (int k = 0; k < V1; ++k) lp[k] = (double)row[k] - lse;
        double sb = a->s + lp[blank];
        int f = -1;
        for (int q = 0; q < nd; ++q)
          if (!ids_cmp(&done[q], a)) f = q;
        if (f < 0 || sb > done[f].s) {
          if (f < 0) f = nd++;
          else
            bhyp_free(&done[f]);
          done[f].n = a->n;
          done[f].ids = arc_malloc(sizeof(int) * (size_t)(a->n + 1));
          memcpy(done[f].ids, a->ids, sizeof(int) * (size_t)a->n);
          done[f].g = arc_malloc(sizeof(float) * (size_t)J);
          memcpy(done[f].g, a->g, sizeof(float) * (size_t)J);
          done[f].hc = arc_malloc(sizeof(float) * nhc);
          memcpy(done[f].hc, a->hc, sizeof(float) * nhc);
          done[f].b = a->b;
          done[f].s = sb;
        }
        if (step < max_sym) {
          const float* ts = tr ? trie_scores(tr, a->b) : NULL;
          for (int k = 0; k < blank; ++k) {
            sc[k] = lp[k] + (ts ? (double)(float)(al * ts[k]) : 0.0);
            order[k] = k;
          }
          /* the width best (stable: lower id first on ties) */
          for (int r = 0; r < width && r < blank; ++r) {
            int bi = r;
            for (int k = r + 1; k < blank; ++k)
              if (sc[order[k]] > sc[order[bi]] || (sc[order[k]] == sc[order[bi]] && order[k] < order[bi])) bi = k;
            int tmp = order[r];
            order[r] = order[bi];
            order[bi] = tmp;
            cand[nc++] = (bcand){a->s + sc[order[r]], i, order[r]};
          }
        }
      }
      if (rc || !nc) break;
      qsort(cand, (size_t)nc, sizeof(bcand), bcand_order);
      if (nc > width) nc = width;
      bhyp* A2 = arc_calloc(sizeof(bhyp) * (size_t)width);
      for (int c = 0; c < nc && !rc; ++c) {
        const bhyp* p = &A[cand[c].i];
        bhyp* h = &A2[c];
        h->n = p->n + 1;
        h->ids = arc_malloc(sizeof(int) * (size_t)h->n);
        memcpy(h->ids, p->ids, sizeof(int) * (size_t)p->n);
        h->ids[p->n] = cand[c].k;
        h->s = cand[c].s;
        h->g = arc_malloc(sizeof(float) * (size_t)J);
        h->hc = arc_malloc(sizeof(float) * nhc);
        memcpy(h->hc, p->hc, sizeof(float) * nhc);
        h->b = tr ? trie_advance(tr, p->b, cand[c].k) : 0;
        if (arc_nft2_be_pred(m->be, cand[c].k, h->hc, h->g)) rc = -1;
      }
      for (int i = 0; i < na; ++i) bhyp_free(&A[i]);
      memcpy(A, A2, sizeof(bhyp) * (size_t)width);
      arc_free(A2);
      na = nc;
    }
    for (int i = 0; i < na; ++i) bhyp_free(&A[i]);
    qsort(done, (size_t)nd, sizeof(bhyp), bhyp_order);
    nh = nd < width ? nd : width;
    for (int i = 0; i < nh; ++i) hy[i] = done[i];
    for (int i = nh; i < nd; ++i) bhyp_free(&done[i]);
    memset(done, 0, sizeof(bhyp) * (size_t)nd);
  }
  arc_free(Fe);
  arc_free(row);
  arc_free(lp);
  arc_free(sc);
  arc_free(order);
  arc_free(A);
  arc_free(done);
  arc_free(cand);
  if (rc) {
    for (int i = 0; i < width; ++i) bhyp_free(&hy[i]);
    arc_free(hy);
    return -1;
  }
  *out = hy;
  return nh;
}

/* ids -> text (pieces joined, "▁" = space, single spaces), malloc'd */
static char* beam_text(const arc_nft2* m, const int* ids, int n) {
  sbuf b = {0};
  sb_put(&b, "");
  int sp = 1;
  for (int i = 0; i < n; ++i) {
    const char* p = ids[i] >= 0 && ids[i] < m->n_piece ? m->piece[ids[i]] : "";
    for (; *p; ++p) {
      int ws = *p == 1 || isspace((unsigned char)*p);
      if (ws) sp = 1;
      else {
        if (sp && b.n) sb_putn(&b, " ", 1);
        sp = 0;
        sb_putn(&b, p, 1);
      }
    }
  }
  return sb_take(&b);
}

/* whitespace-separated words of s -> *out (malloc'd strings), count */
static int split_ws(const char* s, char*** out) {
  int n = 0, cap = 8;
  char** w = arc_malloc(sizeof(char*) * (size_t)cap);
  while (*s) {
    while (*s && isspace((unsigned char)*s)) s++;
    if (!*s) break;
    const char* e = s;
    while (*e && !isspace((unsigned char)*e)) e++;
    if (n == cap) {
      cap *= 2;
      char** x = arc_malloc(sizeof(char*) * (size_t)cap);
      memcpy(x, w, sizeof(char*) * (size_t)n);
      arc_free(w);
      w = x;
    }
    w[n] = arc_malloc((size_t)(e - s) + 1);
    memcpy(w[n], s, (size_t)(e - s));
    w[n++][e - s] = 0;
    s = e;
  }
  *out = w;
  return n;
}

/* rnnt_beam._key: per word lower case letters / digits / apostrophes, single spaces; malloc'd */
static char* lower_dup(const char* s) {
  char** t;
  int n = split_ws(s, &t);
  sbuf b = {0};
  sb_put(&b, "");
  for (int i = 0; i < n; ++i) {
    char* k = arc_post_wk(t[i]);
    if (k[0]) {
      if (b.n) sb_put(&b, " ");
      sb_put(&b, k);
    }
    arc_free(k);
    arc_free(t[i]);
  }
  arc_free(t);
  return sb_take(&b);
}

/* rnnt_beam.utt_alts + word_alts over the final words w (after the guard and casing) */
static void nbest_alts(arc_nft2* m, const float* H, int T, int bias, rword* w, int nw, arc_utt_build* u) {
  bhyp* hy;
  int nh = rnnt_beam(m, H, T, bias, m->cur_nbest, &hy);
  if (nh <= 0) return;
  double mx = hy[0].s, z = 0;
  for (int i = 0; i < nh; ++i) z += exp(hy[i].s - mx);
  /* the greedy text (seen) */
  sbuf gb = {0};
  sb_put(&gb, "");
  for (int i = 0; i < nw; ++i) {
    if (i) sb_put(&gb, " ");
    sb_put(&gb, w[i].text);
  }
  char* gt = sb_take(&gb);
  char** seen = arc_malloc(sizeof(char*) * (size_t)(nh + 1));
  int ns = 0;
  seen[ns++] = lower_dup(gt);
  arc_free(gt);
  u->alts = arc_malloc(sizeof(char*) * (size_t)(nh + 1));
  u->alt_scores = arc_malloc(sizeof(float) * (size_t)(nh + 1));
  u->n_alts = 0;
  /* word keys of the greedy words */
  char** wl = arc_malloc(sizeof(char*) * (size_t)(nw + 1));
  for (int i = 0; i < nw; ++i) wl[i] = lower_dup(w[i].text);
  for (int h = 0; h < nh; ++h) {
    double p = arc_pyround(exp(hy[h].s - mx) / z, 4);
    char* t = beam_text(m, hy[h].ids, hy[h].n);
    char* tl = lower_dup(t);
    int dup = 0;
    for (int q = 0; q < ns; ++q) dup |= !strcmp(seen[q], tl);
    if (dup) {
      arc_free(t);
      arc_free(tl);
      continue;
    }
    seen[ns++] = tl;
    u->alts[u->n_alts] = t;
    u->alt_scores[u->n_alts++] = (float)p;
    /* word_alts: the alternative's words against the greedy words (string keys -> ids for sm_blocks) */
    char** bw;
    int nb = split_ws(t, &bw);
    uint32_t* ka = arc_malloc(sizeof(uint32_t) * (size_t)(nw + 1));
    uint32_t* kb = arc_malloc(sizeof(uint32_t) * (size_t)(nb + 1));
    for (int i = 0; i < nw; ++i) {
      ka[i] = (uint32_t)i;
      for (int j = 0; j < i; ++j)
        if (!strcmp(wl[j], wl[i])) {
          ka[i] = ka[j];
          break;
        }
    }
    for (int j = 0; j < nb; ++j) {
      char* l = lower_dup(bw[j]);
      kb[j] = (uint32_t)(nw + j);
      for (int i = 0; i < nw; ++i)
        if (!strcmp(wl[i], l)) {
          kb[j] = ka[i];
          break;
        }
      if (kb[j] == (uint32_t)(nw + j))
        for (int q = 0; q < j; ++q) {
          char* lq = lower_dup(bw[q]);
          int same = !strcmp(lq, l);
          arc_free(lq);
          if (same) {
            kb[j] = kb[q];
            break;
          }
        }
      arc_free(l);
    }
    int nbk;
    smblock* bl = sm_blocks(ka, nw, kb, nb, &nbk);
    int i = 0, j = 0;
    for (int x = 0; x < nbk; ++x) {   /* get_opcodes: 'replace' runs */
      int ai = bl[x].a, bj = bl[x].b, sz = bl[x].n;
      if (i < ai && j < bj && ai - i == 1 && bj - j >= 1 && bj - j <= 3) {
        rword* r = &w[i];
        sbuf ab = {0};
        sb_put(&ab, "");
        for (int q = j; q < bj; ++q) {
          if (q > j) sb_put(&ab, " ");
          sb_put(&ab, bw[q]);
        }
        char* at = sb_take(&ab);
        char* al = lower_dup(at);
        int skip = !strcmp(al, wl[i]);
        if (!skip && r->alt) {
          char* x2 = lower_dup(r->alt);
          skip = !strcmp(x2, al);
          arc_free(x2);
        }
        for (int q = 0; !skip && q < r->n_xalt; ++q) {
          char* x2 = lower_dup(r->xalt[q]);
          skip = !strcmp(x2, al);
          arc_free(x2);
        }
        arc_free(al);
        if (skip) arc_free(at);
        else {
          char** xa = arc_malloc(sizeof(char*) * (size_t)(r->n_xalt + 1));
          float* xp = arc_malloc(sizeof(float) * (size_t)(r->n_xalt + 1));
          if (r->n_xalt) {
            memcpy(xa, r->xalt, sizeof(char*) * (size_t)r->n_xalt);
            memcpy(xp, r->xalt_p, sizeof(float) * (size_t)r->n_xalt);
          }
          arc_free(r->xalt);
          arc_free(r->xalt_p);
          xa[r->n_xalt] = at;
          xp[r->n_xalt++] = (float)p;
          r->xalt = xa;
          r->xalt_p = xp;
        }
      }
      i = ai + sz;
      j = bj + sz;
    }
    arc_free(bl);
    arc_free(ka);
    arc_free(kb);
    for (int q = 0; q < nb; ++q) arc_free(bw[q]);
    arc_free(bw);
  }
  for (int q = 0; q < ns; ++q) arc_free(seen[q]);
  arc_free(seen);
  for (int i = 0; i < nw; ++i) arc_free(wl[i]);
  arc_free(wl);
  for (int i = 0; i < nh; ++i) bhyp_free(&hy[i]);
  arc_free(hy);
}
