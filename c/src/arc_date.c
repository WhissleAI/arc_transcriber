/* Spoken dates: a C port of arc_audio written.py (is_month, trail, is_num_word, num_runs, toks_of, parse_date,
 * date_iso) and api.py (extend_date). Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include "arc_date.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arc_align.h"
#include "arc_base.h"
#include "arc_num.h"

static int isdig(int c) { return c >= '0' && c <= '9'; }
static int eq(const char* a, const char* b) { return a && b && !strcmp(a, b); }

/* written.MONTH_ID: 1..12, 0 when absent */
static int month_id(const char* c) {
  static const char* const M[] = {"january", "february", "march",     "april",   "may",      "june",
                                  "july",    "august",   "september", "october", "november", "december"};
  for (int i = 0; i < 12; ++i)
    if (eq(c, M[i])) return i + 1;
  return eq(c, "sept") ? 9 : 0;
}

/* written.core = numvalue.lw; free() it */
static char* core(const char* w) { return arc_num_lw(w); }
static int core_is(const char* w, const char* s) {
  char* c = core(w);
  int r = eq(c, s);
  free(c);
  return r;
}

int arc_is_month(const char* w) {
  size_t a = 0;
  while (w[a] && strchr("\"'([", w[a])) a++;
  const char* r = a && w[a] ? w + a : w;   /* lead(w) and w[len(lead(w)):] or w */
  char* c = core(r);
  int m = month_id(c) && r[0] >= 'A' && r[0] <= 'Z';
  free(c);
  return m;
}

/* written.trail: the length of the trailing punctuation, keeping the dot of "p.m." / "a.m." with the word (0: none) */
static int is_wordch(unsigned char c) { return isdig(c) || ((c | 32) >= 'a' && (c | 32) <= 'z') || c == '_' || c >= 0x80; }
static int trail_len(const char* w) {
  size_t n = strlen(w), s = n;
  while (s && strchr(",;:?!.", w[s - 1])) s--;
  if (s == n) return 0;
  int t = (int)(n - s);
  /* \b[ap]\.m\.[,;:?!.]*$ (case-insensitive): the run starts with the dot after "m" */
  if (w[s] == '.' && s >= 3 && ((w[s - 3] | 32) == 'a' || (w[s - 3] | 32) == 'p') && w[s - 2] == '.' &&
      (w[s - 1] | 32) == 'm' && (s == 3 || !is_wordch((unsigned char)w[s - 4])))
    t--;
  return t;
}
static int ends_punct(const char* w) {   /* w.rstrip().endswith(('.', ',', '?', '!')) */
  size_t n = strlen(w);
  while (n && (w[n - 1] == ' ' || (w[n - 1] >= 9 && w[n - 1] <= 13))) n--;
  return n && strchr(".,?!", w[n - 1]);
}

/* written.is_num_word on a core word */
static int is_num_word(const char* c) {
  if (!c || !c[0]) return 0;
  const unsigned char* p = (const unsigned char*)c;
  if (*p == '$' || *p == '#') p++;
  else if ((p[0] == 0xC2 && (p[1] == 0xA3 || p[1] == 0xA5))) p += 2;                    /* £ ¥ */
  else if (p[0] == 0xE2 && p[1] == 0x82 && (p[2] == 0xAC || p[2] == 0xB9)) p += 3;      /* € ₹ */
  if (isdig(*p)) {   /* [$£€¥₹#]?\d[\d,]*(\.\d+)?(st|nd|rd|th|s|%|k|m)? */
    p++;
    while (isdig(*p) || *p == ',') p++;
    if (*p == '.' && isdig(p[1])) {
      p++;
      while (isdig(*p)) p++;
    }
    if (!*p) return 1;
    const char* s = (const char*)p;
    if (eq(s, "st") || eq(s, "nd") || eq(s, "rd") || eq(s, "th") || eq(s, "s") || eq(s, "%") || eq(s, "k") ||
        eq(s, "m"))
      return 1;
  }
  p = (const unsigned char*)c;   /* \d+[:\-]\d+ */
  if (isdig(*p)) {
    while (isdig(*p)) p++;
    if ((*p == ':' || *p == '-') && isdig(p[1])) {
      p++;
      while (isdig(*p)) p++;
      if (!*p) return 1;
    }
  }
  /* all(p in NUMW for p in c.split('-')) */
  char buf[128];
  const char* s = c;
  for (;;) {
    const char* e = strchr(s, '-');
    size_t l = e ? (size_t)(e - s) : strlen(s);
    if (l >= sizeof buf) return 0;
    memcpy(buf, s, l);
    buf[l] = 0;
    if (!arc_num_numw(buf)) return 0;
    if (!e) return 1;
    s = e + 1;
  }
}

static int is_sc(const char* c) { return eq(c, "hundred") || eq(c, "thousand") || eq(c, "million") || eq(c, "billion"); }

/* written.num_runs(texts) (lo 0, hi n, no extra): maximal runs [i, j) of number words, written to runs[2k], runs[2k+1] */
static int num_runs(const char* const* texts, char* const* C, int n, int* runs) {
  int nr = 0, i = 0;
  while (i < n) {
    const char* c = C[i];
    const char* nxt = i + 1 < n ? C[i + 1] : "";
    if (!(is_num_word(c) || ((eq(c, "a") || eq(c, "an")) && is_sc(nxt)) ||
          ((eq(c, "minus") || eq(c, "negative")) && is_num_word(nxt)))) {
      i++;
      continue;
    }
    int j = i + 1;
    while (j < n && !trail_len(texts[j - 1])) {
      c = C[j];
      const char* n1 = j + 1 < n ? C[j + 1] : "";
      const char* n2 = j + 2 < n ? C[j + 2] : "";
      if (is_num_word(c)) j += 1;
      else if ((eq(c, "point") || eq(c, "oh") || eq(c, "o")) && is_num_word(n1) && !trail_len(texts[j])) j += 2;
      else if (eq(c, "and") && (eq(n1, "a") || eq(n1, "an")) && (eq(n2, "half") || eq(n2, "quarter")) &&
               !trail_len(texts[j]) && !trail_len(texts[j + 1]))
        j += 3;
      else if (eq(c, "and") && is_num_word(n1) && !trail_len(texts[j]) && is_sc(C[j - 1])) j += 2;
      else break;
    }
    runs[2 * nr] = i;
    runs[2 * nr + 1] = j;
    nr++;
    i = j;
  }
  return nr;
}

/* written.toks_of: the value tokens of the words ('<none>' and 'UNITW' dropped); *tok malloc'd (free() it), -1 on none */
static int toks_of(const char* const* words, int n, const char*** tok, int* ordf) {
  const char** t = NULL;
  *ordf = 0;
  int nt = arc_num_word_tokens(words, n, &t, NULL, ordf), k = 0;
  if (nt < 0) {
    *tok = NULL;
    return 0;
  }
  for (int q = 0; q < nt; ++q)
    if (!eq(t[q], "<none>") && !eq(t[q], "UNITW")) t[k++] = t[q];
  *tok = t;
  return k;
}

static int days_in(int y, int m) {
  static const int D[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  if (m == 2 && ((y % 4 == 0 && y % 100 != 0) || y % 400 == 0)) return 29;
  return D[m - 1];
}

int arc_parse_date(const char* const* ws, int n, arc_ymd* out) {
  if (n <= 0) return 0;
  char** C = arc_malloc(sizeof(char*) * (size_t)n);
  int* runs = arc_malloc(sizeof(int) * (size_t)(2 * n + 2));
  int ok = 0, nm = 0, mk = -1;
  for (int k = 0; k < n; ++k) {
    C[k] = core(ws[k]);
    if (month_id(C[k])) {
      if (!nm) mk = k;
      nm++;
    }
  }
  arc_ymd d = {0, mk >= 0 ? month_id(C[mk]) : 0, 0};
  int nr = nm > 1 ? 0 : num_runs(ws, C, n, runs);
  if (nm > 1 || !nr || nr > 2) goto done;
  for (int r = 0; r < nr; ++r) {
    int i = runs[2 * r], j = runs[2 * r + 1], ordf = 0;
    const char** tok;
    int nt = toks_of(ws + i, j - i, &tok, &ordf);
    double v;
    if (ordf && !d.day) {
      int vok = arc_num_compose_int(tok, nt, &v);
      free(tok);
      if (!vok || !(v >= 1 && v <= 31)) goto done;
      d.day = (int)v;
      continue;
    }
    int big = 0;
    for (int q = 0; q < nt; ++q) big |= eq(tok[q], "thousand") || eq(tok[q], "hundred");
    if (arc_num_compose_int(tok, nt, &v) && v == (double)(long long)v && v >= 1 && v <= 31 && !d.day && mk >= 0 &&
        j - i <= 2 && !big) {
      free(tok);
      d.day = (int)v;
      continue;
    }
    arc_numval y = arc_num_compose(tok, nt, "date");
    free(tok);
    if (!y.ok || y.has_v2 || y.v != (double)(long long)y.v || !(y.v >= 1000 && y.v <= 2999) || d.year) goto done;
    d.year = (int)y.v;
  }
  for (int k = 0; k < n; ++k) {   /* other words: only 'the', 'of', 'on', 'in' */
    if (k == mk) continue;
    int in_run = 0;
    for (int r = 0; r < nr; ++r) in_run |= runs[2 * r] <= k && k < runs[2 * r + 1];
    if (in_run) continue;
    if (!(eq(C[k], "the") || eq(C[k], "of") || eq(C[k], "on") || eq(C[k], "in") || eq(C[k], ""))) goto done;
  }
  if (!d.day && !d.year) goto done;
  if (d.day && d.month && d.day > days_in(d.year ? d.year : 2000, d.month)) goto done;
  *out = d;
  ok = 1;
done:
  for (int k = 0; k < n; ++k) free(C[k]);
  arc_free(C);
  arc_free(runs);
  return ok;
}

int arc_date_iso(const arc_ymd* d, char* out) {
  out[0] = 0;
  if (!d) return 0;
  int y = d->year, m = d->month, dd = d->day;
  if (y && m && dd) snprintf(out, 16, "%04d-%02d-%02d", y, m, dd);
  else if (m && dd) snprintf(out, 16, "--%02d-%02d", m, dd);
  else if (y && m) snprintf(out, 16, "%04d-%02d", y, m);
  else if (y && !dd) snprintf(out, 16, "%04d", y);
  else return 0;
  return 1;
}

int arc_extend_date(const arc_tword* W, int n, int i, int j, char** heard, double* start, double* end, arc_ymd* d) {
  if (i > 0 && arc_is_month(W[i - 1].w) && W[i].start - W[i - 1].end < 0.6 && !ends_punct(W[i - 1].w)) i--;
  if (j + 2 < n && core_is(W[j + 1].w, "of") && arc_is_month(W[j + 2].w) && !ends_punct(W[j].w)) j += 2;
  int k0 = -1;
  for (int k = i; k <= j && k0 < 0; ++k) {
    const char* w = W[k].w;
    int dig = 0;
    for (const char* p = w; *p; ++p) dig |= isdig((unsigned char)*p);
    char* c = core(w);
    if (arc_is_month(w) || arc_num_numw(c) || dig) k0 = k;
    free(c);
  }
  if (k0 < 0) return 0;
  i = k0;
  int m = j - i + 1;
  const char** ws = arc_malloc(sizeof(char*) * (size_t)m);
  size_t len = 1;
  for (int k = 0; k < m; ++k) {
    ws[k] = W[i + k].w;
    len += strlen(ws[k]) + 1;
  }
  arc_ymd v;
  if (!arc_parse_date(ws, m, &v)) {
    arc_free(ws);
    return 0;
  }
  char* h = arc_malloc(len);
  h[0] = 0;
  for (int k = 0; k < m; ++k) {
    if (k) strcat(h, " ");
    strcat(h, ws[k]);
  }
  arc_free(ws);
  arc_free(*heard);
  *heard = h;
  *start = arc_pyround(*start < W[i].start ? *start : W[i].start, 3);
  *end = arc_pyround(*end > W[j].end ? *end : W[j].end, 3);
  *d = v;
  return 1;
}
