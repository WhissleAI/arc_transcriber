/* The Moonshine BPE tokenizer. See arc_tok.h.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include "arc_tok.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arc_base.h"

/* string -> int open-addressing table */
typedef struct {
  char** key;
  int* val;
  uint32_t cap;
} smap;

static uint32_t hash_str(const char* s, size_t n) {
  uint32_t h = 2166136261u;
  for (size_t i = 0; i < n; ++i) {
    h ^= (unsigned char)s[i];
    h *= 16777619u;
  }
  return h;
}

static int smap_init(smap* m, uint32_t n) {
  m->cap = 1;
  while (m->cap < n * 2) m->cap <<= 1;
  m->key = arc_calloc(sizeof(char*) * m->cap);
  m->val = arc_calloc(sizeof(int) * m->cap);
  return m->key && m->val ? 0 : -1;
}

static void smap_put(smap* m, const char* k, int v) {
  uint32_t i = hash_str(k, strlen(k)) & (m->cap - 1);
  while (m->key[i] && strcmp(m->key[i], k)) i = (i + 1) & (m->cap - 1);
  if (!m->key[i]) m->key[i] = arc_strdup(k);
  m->val[i] = v;
}

static int smap_get(const smap* m, const char* k, size_t n) {
  uint32_t i = hash_str(k, n) & (m->cap - 1);
  while (m->key[i]) {
    if (strlen(m->key[i]) == n && !memcmp(m->key[i], k, n)) return m->val[i];
    i = (i + 1) & (m->cap - 1);
  }
  return -1;
}

static void smap_free(smap* m) {
  if (m->key)
    for (uint32_t i = 0; i < m->cap; ++i) arc_free(m->key[i]);
  arc_free(m->key);
  arc_free(m->val);
}

/* (left id, right id) -> rank and merged id */
typedef struct {
  uint64_t* key;
  int* rank;
  int* merged;
  uint32_t cap;
} pmap;

static uint32_t hash64(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  return (uint32_t)x;
}

struct arc_tok {
  int n;                /* vocabulary incl. added tokens */
  char** piece;         /* id -> piece */
  unsigned char* special;
  smap vocab;
  pmap merges;
  int unk;
  int byte_id[256];     /* <0xXX> pieces */
};

void arc_tok_free(arc_tok* t) {
  if (!t) return;
  if (t->piece)
    for (int i = 0; i < t->n; ++i) arc_free(t->piece[i]);
  arc_free(t->piece);
  arc_free(t->special);
  smap_free(&t->vocab);
  arc_free(t->merges.key);
  arc_free(t->merges.rank);
  arc_free(t->merges.merged);
  arc_free(t);
}

arc_tok* arc_tok_load(const char* path) {
  char* text = NULL;
  size_t len = 0;
  arc_json* j = NULL;
  arc_tok* t = arc_calloc(sizeof *t);
  if (!t) return NULL;
  if (arc_read_file(path, &text, &len)) goto fail;
  if (!(j = arc_json_parse(text, len))) goto fail;
  const arc_json* model = arc_json_get(j, "model");
  const arc_json* vocab = arc_json_get(model, "vocab");
  const arc_json* merges = arc_json_get(model, "merges");
  const arc_json* added = arc_json_get(j, "added_tokens");
  if (!vocab || vocab->type != ARC_JOBJ || !merges || merges->type != ARC_JARR) {
    arc_seterr("%s: not a BPE tokenizer.json", path);
    goto fail;
  }
  int maxid = -1;
  for (int i = 0; i < vocab->n; ++i) {
    int id = (int)vocab->items[i].num;
    if (id > maxid) maxid = id;
  }
  if (added)
    for (int i = 0; i < added->n; ++i) {
      int id = (int)arc_json_num(arc_json_get(&added->items[i], "id"), -1);
      if (id > maxid) maxid = id;
    }
  t->n = maxid + 1;
  t->piece = arc_calloc(sizeof(char*) * (size_t)t->n);
  t->special = arc_calloc((size_t)t->n);
  if (!t->piece || !t->special || smap_init(&t->vocab, (uint32_t)vocab->n + 1024)) goto fail;
  for (int i = 0; i < vocab->n; ++i) {
    int id = (int)vocab->items[i].num;
    t->piece[id] = arc_strdup(vocab->keys[i]);
    smap_put(&t->vocab, vocab->keys[i], id);
  }
  if (added)
    for (int i = 0; i < added->n; ++i) {
      const arc_json* a = &added->items[i];
      int id = (int)arc_json_num(arc_json_get(a, "id"), -1);
      const char* c = arc_json_str(arc_json_get(a, "content"), "");
      if (id < 0) continue;
      if (!t->piece[id]) t->piece[id] = arc_strdup(c);
      t->special[id] = arc_json_num(arc_json_get(a, "special"), 0) != 0;
    }
  t->unk = smap_get(&t->vocab, "<unk>", 5);
  for (int b = 0; b < 256; ++b) {
    char s[8];
    snprintf(s, sizeof s, "<0x%02X>", b);
    t->byte_id[b] = smap_get(&t->vocab, s, strlen(s));
  }
  /* merges: [[a, b], ...] in rank order */
  uint32_t cap = 1;
  while (cap < (uint32_t)merges->n * 2) cap <<= 1;
  t->merges.cap = cap;
  t->merges.key = arc_malloc(sizeof(uint64_t) * cap);
  t->merges.rank = arc_malloc(sizeof(int) * cap);
  t->merges.merged = arc_malloc(sizeof(int) * cap);
  if (!t->merges.key || !t->merges.rank || !t->merges.merged) goto fail;
  for (uint32_t i = 0; i < cap; ++i) t->merges.key[i] = UINT64_MAX;
  char buf[1024];
  for (int r = 0; r < merges->n; ++r) {
    const arc_json* p = &merges->items[r];
    const char *a, *b;
    if (p->type == ARC_JARR && p->n == 2) {
      a = p->items[0].str;
      b = p->items[1].str;
    } else if (p->type == ARC_JSTR) {   /* older "a b" form */
      const char* sp = strchr(p->str, ' ');
      if (!sp) continue;
      size_t la = (size_t)(sp - p->str);
      if (la >= sizeof buf) continue;
      memcpy(buf, p->str, la);
      buf[la] = 0;
      a = buf;
      b = sp + 1;
    } else continue;
    int ia = smap_get(&t->vocab, a, strlen(a)), ib = smap_get(&t->vocab, b, strlen(b));
    char cat[1024];
    snprintf(cat, sizeof cat, "%s%s", a, b);
    int im = smap_get(&t->vocab, cat, strlen(cat));
    if (ia < 0 || ib < 0 || im < 0) continue;
    uint64_t key = (uint64_t)(uint32_t)ia << 32 | (uint32_t)ib;
    uint32_t h = hash64(key) & (cap - 1);
    while (t->merges.key[h] != UINT64_MAX && t->merges.key[h] != key) h = (h + 1) & (cap - 1);
    if (t->merges.key[h] == key) continue;   /* the first (lowest) rank wins */
    t->merges.key[h] = key;
    t->merges.rank[h] = r;
    t->merges.merged[h] = im;
  }
  arc_json_free(j);
  arc_free(text);
  return t;
fail:
  arc_json_free(j);
  arc_free(text);
  arc_tok_free(t);
  return NULL;
}

static int merge_of(const arc_tok* t, int a, int b, int* merged) {
  uint64_t key = (uint64_t)(uint32_t)a << 32 | (uint32_t)b;
  uint32_t h = hash64(key) & (t->merges.cap - 1);
  while (t->merges.key[h] != UINT64_MAX) {
    if (t->merges.key[h] == key) {
      *merged = t->merges.merged[h];
      return t->merges.rank[h];
    }
    h = (h + 1) & (t->merges.cap - 1);
  }
  return -1;
}

static int utf8_len(unsigned char c) {
  if (c < 0x80) return 1;
  if ((c >> 5) == 6) return 2;
  if ((c >> 4) == 14) return 3;
  if ((c >> 3) == 30) return 4;
  return 1;
}

int arc_tok_encode(const arc_tok* t, const char* text, int* ids, int cap) {
  size_t n = strlen(text);
  if (!n) return 0;
  /* normalizer: prepend "▁", every space -> "▁" */
  char* s = arc_malloc(n * 3 + 4);
  int* sym = arc_malloc(sizeof(int) * (n * 3 + 8));
  if (!s || !sym) {
    arc_free(s);
    arc_free(sym);
    return -1;
  }
  size_t k = 0;
  memcpy(s, "\xE2\x96\x81", 3);
  k = 3;
  for (size_t i = 0; i < n; ++i) {
    if (text[i] == ' ') {
      memcpy(s + k, "\xE2\x96\x81", 3);
      k += 3;
    } else s[k++] = text[i];
  }
  s[k] = 0;
  /* one symbol per character; unknown characters fall back to their bytes */
  int ns = 0;
  for (size_t i = 0; i < k;) {
    int l = utf8_len((unsigned char)s[i]);
    if (i + (size_t)l > k) l = (int)(k - i);
    int id = smap_get(&t->vocab, s + i, (size_t)l);
    if (id >= 0) sym[ns++] = id;
    else
      for (int b = 0; b < l; ++b) {
        int bid = t->byte_id[(unsigned char)s[i + b]];
        if (bid >= 0) sym[ns++] = bid;
        else if (!(ns && sym[ns - 1] == t->unk)) sym[ns++] = t->unk;   /* fuse_unk */
      }
    i += (size_t)l;
  }
  /* merges: always the lowest-ranked pair, leftmost first */
  while (ns > 1) {
    int best = -1, br = 0x7fffffff, bm = -1;
    for (int i = 0; i + 1 < ns; ++i) {
      int mg, r = merge_of(t, sym[i], sym[i + 1], &mg);
      if (r >= 0 && r < br) {
        br = r;
        best = i;
        bm = mg;
      }
    }
    if (best < 0) break;
    sym[best] = bm;
    memmove(sym + best + 1, sym + best + 2, sizeof(int) * (size_t)(ns - best - 2));
    ns--;
  }
  for (int i = 0; i < ns && i < cap; ++i) ids[i] = sym[i];
  arc_free(s);
  arc_free(sym);
  return ns;
}

const char* arc_tok_piece(const arc_tok* t, int id) { return id >= 0 && id < t->n ? t->piece[id] : NULL; }
int arc_tok_is_special(const arc_tok* t, int id) { return id >= 0 && id < t->n && t->special[id]; }
int arc_tok_size(const arc_tok* t) { return t->n; }

static int byte_piece(const char* p) {
  if (!p || strlen(p) != 6 || strncmp(p, "<0x", 3) || p[5] != '>') return -1;
  int v = 0;
  for (int i = 3; i < 5; ++i) {
    char c = p[i];
    v = v * 16 + (c >= '0' && c <= '9' ? c - '0' : c >= 'A' && c <= 'F' ? c - 'A' + 10 : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -100);
  }
  return v >= 0 && v < 256 ? v : -1;
}

/* is s[0..n) valid UTF-8? */
static int valid_utf8(const unsigned char* s, size_t n) {
  size_t i = 0;
  while (i < n) {
    int l = utf8_len(s[i]);
    if (l == 1 && s[i] >= 0x80) return 0;
    if (i + (size_t)l > n) return 0;
    for (int j = 1; j < l; ++j)
      if ((s[i + j] & 0xC0) != 0x80) return 0;
    i += (size_t)l;
  }
  return 1;
}

char* arc_tok_decode(const arc_tok* t, const int* ids, int n) {
  size_t cap = 64;
  for (int i = 0; i < n; ++i) cap += t->piece[ids[i] >= 0 && ids[i] < t->n ? ids[i] : 0] ? strlen(t->piece[ids[i]]) + 4 : 4;
  char* out = arc_malloc(cap * 3);
  unsigned char bytes[256];
  if (!out) return NULL;
  size_t k = 0;
  int nb = 0;
  for (int i = 0; i <= n; ++i) {
    const char* p = NULL;
    int b = -1;
    if (i < n) {
      int id = ids[i];
      if (id < 0 || id >= t->n || t->special[id] || !t->piece[id]) continue;
      p = t->piece[id];
      b = byte_piece(p);
    }
    if (b >= 0 && nb < (int)sizeof bytes) {
      bytes[nb++] = (unsigned char)b;
      continue;
    }
    if (nb) {   /* a run of byte tokens: as UTF-8 if valid, else one U+FFFD per byte */
      if (valid_utf8(bytes, (size_t)nb)) {
        memcpy(out + k, bytes, (size_t)nb);
        k += (size_t)nb;
      } else
        for (int j = 0; j < nb; ++j) {
          memcpy(out + k, "\xEF\xBF\xBD", 3);
          k += 3;
        }
      nb = 0;
    }
    if (!p) continue;
    for (const char* c = p; *c;) {   /* "▁" -> " " */
      if (!strncmp(c, "\xE2\x96\x81", 3)) {
        out[k++] = ' ';
        c += 3;
      } else out[k++] = *c++;
    }
  }
  out[k] = 0;
  /* Strip(" ", 1, 0) on the fused string, then Python's .strip() */
  char* a = out;
  if (*a == ' ') a++;
  while (*a == ' ' || *a == '\t' || *a == '\n' || *a == '\r') a++;
  size_t l = strlen(a);
  while (l && (a[l - 1] == ' ' || a[l - 1] == '\t' || a[l - 1] == '\n' || a[l - 1] == '\r')) a[--l] = 0;
  memmove(out, a, l + 1);
  return out;
}
