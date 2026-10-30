/* C port of arc_audio numlib.py / numvalue.py / textlayer.py (number values from words and value-head tokens).
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include "arc_num.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arc_base.h"

/* ------------------------------------------------------------------ vocabularies (numlib.py) */

static const char* NWT[ARC_NWT_N] = {
    "<none>", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
    "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion", "point",
    "and", "a", "half", "quarter", "dozen", "SEP", "OH", "NEG", "UNITW"};
#define UNITS_W (NWT + 1)   /* zero .. nineteen */
#define TENS_W (NWT + 21)   /* twenty .. ninety */

static const struct { const char *o, *c; } ORD[] = {
    {"first", "one"}, {"second", "two"}, {"third", "three"}, {"fourth", "four"}, {"fifth", "five"}, {"sixth", "six"},
    {"seventh", "seven"}, {"eighth", "eight"}, {"ninth", "nine"}, {"tenth", "ten"}, {"eleventh", "eleven"},
    {"twelfth", "twelve"}, {"thirteenth", "thirteen"}, {"fourteenth", "fourteen"}, {"fifteenth", "fifteen"},
    {"sixteenth", "sixteen"}, {"seventeenth", "seventeen"}, {"eighteenth", "eighteen"}, {"nineteenth", "nineteen"},
    {"twentieth", "twenty"}, {"thirtieth", "thirty"}, {"fortieth", "forty"}, {"fiftieth", "fifty"},
    {"sixtieth", "sixty"}, {"seventieth", "seventy"}, {"eightieth", "eighty"}, {"ninetieth", "ninety"},
    {"hundredth", "hundred"}, {"thousandth", "thousand"}, {"millionth", "million"}};
static const struct { const char *a, *t; } ALIAS[] = {
    {"oh", "OH"}, {"o", "OH"}, {"nil", "zero"}, {"nought", "zero"}, {"naught", "zero"}, {"love", "zero"},
    {"to", "SEP"}, {"for", "SEP"}, {"dash", "SEP"}, {"-", "SEP"}, {"minus", "NEG"}, {"negative", "NEG"},
    {"an", "a"}, {"couple", NULL}};

/* ------------------------------------------------------------------ tables (numvalue.py) */

static const char* UNITS[] = {
    "second", "minute", "hour", "day", "week", "month", "year", "decade", "am", "pm", "oclock", "USD", "EUR", "GBP",
    "JPY", "CAD", "AUD", "INR", "XXX", "mile", "foot", "inch", "ft_in", "yard", "meter", "kilometer", "centimeter",
    "kilogram", "pound_wt", "ounce", "gallon", "liter", "degree", "mph", "kph", "acre", "points", "rebounds",
    "assists", "yards", "runs", "wickets", "overs", "saves", "shots", "goals", "kills", "steals", "blocks", "sacks",
    "touchdowns", "catches", "threes", "turnovers", "fouls", "boundaries", "sixes", "fours", "hits", "strikeouts",
    "ppg", "times", "people", "age", "page", "exhibit", "section", "chapter", "channel", "docket", "rule", "line",
    "flight", "paragraph", "article", "volume", "question", "route", "room", "case", "item", "number", "other"};
#define N_UNITS ((int)(sizeof UNITS / sizeof *UNITS))

static const struct { const char *w, *u; } UWORD[] = {
    {"second", "second"}, {"seconds", "second"}, {"sec", "second"}, {"secs", "second"}, {"minute", "minute"},
    {"minutes", "minute"}, {"mins", "minute"}, {"min", "minute"}, {"hour", "hour"}, {"hours", "hour"},
    {"hrs", "hour"}, {"day", "day"}, {"days", "day"}, {"week", "week"}, {"weeks", "week"}, {"month", "month"},
    {"months", "month"}, {"year", "year"}, {"years", "year"}, {"decade", "decade"}, {"decades", "decade"},
    {"fortnight", "week"}, {"am", "am"}, {"a.m.", "am"}, {"a.m", "am"}, {"pm", "pm"}, {"p.m.", "pm"}, {"p.m", "pm"},
    {"o'clock", "oclock"}, {"oclock", "oclock"}, {"dollars", "USD"}, {"dollar", "USD"}, {"bucks", "USD"},
    {"$", "USD"}, {"cents", "USD"}, {"euros", "EUR"}, {"euro", "EUR"}, {"\xe2\x82\xac", "EUR"}, {"pounds", "GBP"},
    {"quid", "GBP"}, {"\xc2\xa3", "GBP"}, {"yen", "JPY"}, {"rupees", "INR"}, {"miles", "mile"}, {"mile", "mile"},
    {"feet", "foot"}, {"foot", "foot"}, {"inches", "inch"}, {"inch", "inch"}, {"yard", "yard"}, {"meters", "meter"},
    {"metres", "meter"}, {"meter", "meter"}, {"kilometers", "kilometer"}, {"kilometres", "kilometer"},
    {"km", "kilometer"}, {"centimeters", "centimeter"}, {"kilos", "kilogram"}, {"kg", "kilogram"},
    {"kilograms", "kilogram"}, {"ounces", "ounce"}, {"gallons", "gallon"}, {"gallon", "gallon"}, {"liters", "liter"},
    {"litres", "liter"}, {"degrees", "degree"}, {"degree", "degree"}, {"mph", "mph"}, {"acres", "acre"},
    {"times", "times"}, {"people", "people"}, {"old", "age"}, {"points", "points"}, {"rebounds", "rebounds"},
    {"assists", "assists"}, {"yards", "yards"}, {"runs", "runs"}, {"wickets", "wickets"}, {"overs", "overs"},
    {"saves", "saves"}, {"shots", "shots"}, {"goals", "goals"}, {"kills", "kills"}, {"steals", "steals"},
    {"blocks", "blocks"}, {"sacks", "sacks"}, {"touchdowns", "touchdowns"}, {"catches", "catches"},
    {"threes", "threes"}, {"turnovers", "turnovers"}, {"fouls", "fouls"}, {"boundaries", "boundaries"},
    {"sixes", "sixes"}, {"fours", "fours"}, {"hits", "hits"}, {"strikeouts", "strikeouts"}, {"ppg", "ppg"},
    {"run", "runs"}, {"point", "points"}};
static const struct { const char *w, *u; } REFW[] = {
    {"page", "page"}, {"pages", "page"}, {"exhibit", "exhibit"}, {"section", "section"}, {"chapter", "chapter"},
    {"channel", "channel"}, {"docket", "docket"}, {"rule", "rule"}, {"line", "line"}, {"flight", "flight"},
    {"paragraph", "paragraph"}, {"article", "article"}, {"volume", "volume"}, {"question", "question"},
    {"route", "route"}, {"interstate", "route"}, {"highway", "route"}, {"room", "room"}, {"case", "case"},
    {"item", "item"}, {"number", "number"}, {"no.", "number"}};

#define LEN(a) ((int)(sizeof a / sizeof *a))

static int eq(const char* a, const char* b) { return a && b && !strcmp(a, b); }
static int in_list(const char* s, const char* const* l, int n) {
  for (int i = 0; i < n; ++i)
    if (eq(s, l[i])) return 1;
  return 0;
}
#define IN(s, ...) in_list((s), (const char* const[]){__VA_ARGS__}, LEN(((const char* const[]){__VA_ARGS__})))

const char* arc_nwt_token(int k) { return k >= 0 && k < ARC_NWT_N ? NWT[k] : NULL; }

/* VAL.get(t): 0..19, 20..90; -1 when absent */
static int val_of(const char* t) {
  if (!t) return -1;
  for (int i = 0; i < 20; ++i)
    if (!strcmp(t, UNITS_W[i])) return i;
  for (int i = 0; i < 8; ++i)
    if (!strcmp(t, TENS_W[i])) return 20 + 10 * i;
  return -1;
}
static long long sc_of(const char* t) {
  if (eq(t, "hundred")) return 100;
  if (eq(t, "thousand")) return 1000;
  if (eq(t, "million")) return 1000000;
  if (eq(t, "billion")) return 1000000000;
  return 0;
}
static int is_tens(const char* t) { return in_list(t, TENS_W, 8); }
static const char* nwt_static(const char* t) {   /* the vocabulary's own copy of a token string */
  for (int i = 0; i < ARC_NWT_N; ++i)
    if (eq(t, NWT[i])) return NWT[i];
  return t;
}
static const char* uword(const char* w) {
  for (int i = 0; i < LEN(UWORD); ++i)
    if (eq(w, UWORD[i].w)) return UWORD[i].u;
  return NULL;
}
static int unit_id(const char* u) {
  for (int i = 0; i < N_UNITS; ++i)
    if (eq(u, UNITS[i])) return i;
  return -1;
}

/* ------------------------------------------------------------------ Python numbers (int or float) */

typedef struct {
  double d;
  int isint;
} pn;
static pn PI(long long i) { return (pn){(double)i, 1}; }
static pn PF(double d) { return (pn){d, 0}; }
static pn padd(pn a, pn b) { return (pn){a.d + b.d, a.isint && b.isint}; }
static pn pmul(pn a, pn b) { return (pn){a.d * b.d, a.isint && b.isint}; }
static int is_integer(pn a) { return a.isint || (isfinite(a.d) && a.d == floor(a.d)); }
static double round3(double x) {   /* Python round(x, 3) on a float: correctly rounded */
  char b[512];
  snprintf(b, sizeof b, "%.3f", x);
  return arc_strtod(b, NULL);
}
static arc_numval mkval(int ok, pn v, int has2, pn v2) {
  arc_numval r = {0};
  if (!ok) return r;
  r.ok = 1;
  r.v = v.d;
  r.v_int = v.isint;
  r.has_v2 = has2;
  if (has2) {
    r.v2 = v2.d;
    r.v2_int = v2.isint;
  }
  return r;
}

/* ------------------------------------------------------------------ token lists */

typedef struct {
  const char** t;
  int* wi;
  int n, cap;
} tlist;
static void tl_push(tlist* l, int wi, const char* t) {
  if (l->n == l->cap) {
    l->cap = l->cap ? l->cap * 2 : 16;
    l->t = realloc(l->t, sizeof(char*) * (size_t)l->cap);
    l->wi = realloc(l->wi, sizeof(int) * (size_t)l->cap);
  }
  l->t[l->n] = t;
  l->wi[l->n++] = wi;
}
static void tl_free(tlist* l) {
  free(l->t);
  free(l->wi);
}

/* numlib.int_words */
static void int_words(unsigned long long n, tlist* l, int wi) {
  if (n < 20) {
    tl_push(l, wi, UNITS_W[n]);
    return;
  }
  if (n < 100) {
    tl_push(l, wi, TENS_W[n / 10 - 2]);
    if (n % 10) tl_push(l, wi, UNITS_W[n % 10]);
    return;
  }
  static const unsigned long long V[4] = {1000000000ULL, 1000000ULL, 1000ULL, 100ULL};
  static const char* S[4] = {"billion", "million", "thousand", "hundred"};
  for (int i = 0; i < 4; ++i)
    if (n >= V[i]) {
      int_words(n / V[i], l, wi);
      tl_push(l, wi, nwt_static(S[i]));
      if (n % V[i]) int_words(n % V[i], l, wi);
      return;
    }
}
static unsigned long long digits_val(const char* s, int n) {   /* int(s) with ',' removed */
  unsigned long long v = 0;
  for (int i = 0; i < n; ++i)
    if (s[i] >= '0' && s[i] <= '9') v = v * 10 + (unsigned long long)(s[i] - '0');
  return v;
}

/* ------------------------------------------------------------------ word normalisation */

static int isdig(int c) { return c >= '0' && c <= '9'; }
static char* lower_dup(const char* s) {
  size_t n = strlen(s);
  char* o = malloc(n + 1);
  for (size_t i = 0; i <= n; ++i) o[i] = (s[i] >= 'A' && s[i] <= 'Z') ? (char)(s[i] + 32) : s[i];
  return o;
}
static void replace_rsquo(char* s) {   /* ’ -> ' */
  char* o = s;
  for (char* p = s; *p;) {
    if ((unsigned char)p[0] == 0xE2 && (unsigned char)p[1] == 0x80 && (unsigned char)p[2] == 0x99) {
      *o++ = '\'';
      p += 3;
    } else *o++ = *p++;
  }
  *o = 0;
}
static void strip_set(char* s, const char* set) {   /* str.strip(set) */
  size_t n = strlen(s), a = 0;
  while (a < n && strchr(set, s[a])) a++;
  while (n > a && strchr(set, s[n - 1])) n--;
  memmove(s, s + a, n - a);
  s[n - a] = 0;
}
static int is_ws(int c) { return c == ' ' || (c >= 9 && c <= 13) || (c >= 0x1c && c <= 0x1f); }
/* the word key tokens_of / word_tokens read: lower, strip .,?!;:"'(), ’ -> ', a trailing 's dropped */
static char* word_key(const char* w) {
  char* s = lower_dup(w);
  strip_set(s, ".,?!;:\"'()");
  replace_rsquo(s);
  size_t n = strlen(s);
  if (n >= 2 && s[n - 2] == '\'' && s[n - 1] == 's') s[n - 2] = 0;
  return s;
}
/* numvalue.lw */
static char* lw(const char* w) {
  char* s = lower_dup(w);
  size_t n = strlen(s), a = 0;
  while (a < n && strchr("\"'([", s[a])) a++;
  while (n > a && strchr("\"'),.?!;:]", s[n - 1])) n--;
  memmove(s, s + a, n - a);
  s[n - a] = 0;
  replace_rsquo(s);
  return s;
}
static int ends_comma(const char* w) {   /* w.rstrip().endswith(',') */
  size_t n = strlen(w);
  while (n && is_ws((unsigned char)w[n - 1])) n--;
  return n && w[n - 1] == ',';
}
static int strip_lower_is(const char* w, const char* a, const char* b) {   /* w.strip().lower() in (a, b) */
  size_t n = strlen(w), s = 0;
  while (s < n && is_ws((unsigned char)w[s])) s++;
  while (n > s && is_ws((unsigned char)w[n - 1])) n--;
  char buf[32];
  if (n - s >= sizeof buf) return 0;
  for (size_t i = s; i < n; ++i) buf[i - s] = (w[i] >= 'A' && w[i] <= 'Z') ? (char)(w[i] + 32) : w[i];
  buf[n - s] = 0;
  return !strcmp(buf, a) || !strcmp(buf, b);
}

/* the number regex '[$£€#]?(\d[\d,]*)(\.\d+)?(st|nd|rd|th|s|%|k|m)?' (fullmatch) */
typedef struct {
  const char* d;
  int dn;
  const char* fr;   /* after the '.' */
  int fn;
  const char* suf;
} numre;
static int match_num(const char* s, numre* m) {
  const char* p = s;
  if (*p == '$' || *p == '#') p++;
  else if (!strncmp(p, "\xc2\xa3", 2)) p += 2;
  else if (!strncmp(p, "\xe2\x82\xac", 3)) p += 3;
  if (!isdig(*p)) return 0;
  m->d = p;
  while (isdig(*p) || *p == ',') p++;
  m->dn = (int)(p - m->d);
  m->fn = 0;
  m->fr = NULL;
  if (*p == '.' && isdig(p[1])) {
    m->fr = ++p;
    while (isdig(*p)) p++;
    m->fn = (int)(p - m->fr);
  }
  static const char* SUF[] = {"", "st", "nd", "rd", "th", "s", "%", "k", "m"};
  for (int i = 0; i < LEN(SUF); ++i)
    if (!strcmp(p, SUF[i])) {
      m->suf = SUF[i];
      return 1;
    }
  return 0;
}
/* '(\d+)[:\-](\d+)' (fullmatch) */
static int match_pair(const char* s, const char** a, int* an, const char** b, int* bn) {
  const char* p = s;
  if (!isdig(*p)) return 0;
  *a = p;
  while (isdig(*p)) p++;
  *an = (int)(p - *a);
  if (*p != ':' && *p != '-') return 0;
  p++;
  if (!isdig(*p)) return 0;
  *b = p;
  while (isdig(*p)) p++;
  *bn = (int)(p - *b);
  return *p == 0;
}

/* the tokens of one word key (shared body of tokens_of / word_tokens) */
static void key_tokens(const char* w0, int wi, tlist* out, int* ordf) {
  numre m;
  const char *a, *b;
  int an, bn;
  if (match_num(w0, &m)) {
    int_words(digits_val(m.d, m.dn), out, wi);
    if (m.fr) {
      tl_push(out, wi, nwt_static("point"));
      for (int i = 0; i < m.fn; ++i) tl_push(out, wi, UNITS_W[m.fr[i] - '0']);
    }
    if (IN(m.suf, "st", "nd", "rd", "th")) *ordf = 1;
    if (!strcmp(m.suf, "k")) tl_push(out, wi, nwt_static("thousand"));
    if (!strcmp(m.suf, "m")) tl_push(out, wi, nwt_static("million"));
    return;
  }
  if (match_pair(w0, &a, &an, &b, &bn)) {
    int_words(digits_val(a, an), out, wi);
    tl_push(out, wi, nwt_static("SEP"));
    int_words(digits_val(b, bn), out, wi);
    return;
  }
  /* w0.split('-') */
  const char* p = w0;
  char part[256];
  for (;;) {
    const char* q = strchr(p, '-');
    size_t n = q ? (size_t)(q - p) : strlen(p);
    if (n >= sizeof part) n = sizeof part - 1;
    memcpy(part, p, n);
    part[n] = 0;
    int done = 0;
    for (int i = 0; i < LEN(ORD) && !done; ++i)
      if (!strcmp(part, ORD[i].o)) {
        tl_push(out, wi, nwt_static(ORD[i].c));
        *ordf = 1;
        done = 1;
      }
    if (!done && (val_of(part) >= 0 || sc_of(part) || IN(part, "point", "and", "a", "half", "quarter", "dozen"))) {
      tl_push(out, wi, nwt_static(part));
      done = 1;
    }
    for (int i = 0; i < LEN(ALIAS) && !done; ++i)
      if (!strcmp(part, ALIAS[i].a)) {
        if (ALIAS[i].t) tl_push(out, wi, nwt_static(ALIAS[i].t));
        done = 1;
      }
    if (!done) tl_push(out, wi, nwt_static("UNITW"));
    if (!q) break;
    p = q + 1;
  }
}

static int numtok_vsc(const char* t) { return val_of(t) >= 0 || sc_of(t); }
static int ahead_ok(const char* t) { return numtok_vsc(t) || IN(t, "half", "quarter", "dozen"); }

/* numlib.tokens_of */
int arc_num_tokens_of(const char* const* words, int n, const char*** toks, int* ordf_out) {
  tlist out = {0};
  int ordf = 0;
  for (int wi = 0; wi < n; ++wi) {
    const char* w = words[wi];
    if (out.n && !eq(out.t[out.n - 1], "SEP") && wi > 0 && ends_comma(words[wi - 1]) &&
        !strip_lower_is(w, "and", "point"))
      tl_push(&out, wi, nwt_static("SEP"));
    char* w0 = word_key(w);
    if (w0[0]) key_tokens(w0, wi, &out, &ordf);
    free(w0);
  }
  for (int i = 0; i < out.n; ++i) {
    const char* t = out.t[i];
    const char* nxt = i + 1 < out.n ? out.t[i + 1] : NULL;
    if (eq(t, "a") && !ahead_ok(nxt)) out.t[i] = nwt_static("UNITW");
    if (ordf && IN(t, "half", "quarter") && i > 0 && val_of(out.t[i - 1]) >= 0) out.t[i] = nwt_static("UNITW");
  }
  int any = 0;
  for (int i = 0; i < out.n; ++i)
    if (numtok_vsc(out.t[i]) || IN(out.t[i], "half", "dozen")) any = 1;
  if (ordf_out) *ordf_out = ordf;
  if (!any) {
    tl_free(&out);
    if (toks) *toks = NULL;
    return -1;
  }
  free(out.wi);
  if (toks) *toks = out.t;
  else free(out.t);
  return out.n;
}

/* numvalue.word_tokens */
int arc_num_word_tokens(const char* const* words, int n, const char*** toks, int** wis, int* ordf_out) {
  tlist out = {0};
  int ordf = 0;
  for (int wi = 0; wi < n; ++wi) {
    const char* w = words[wi];
    int n0 = out.n;
    if (out.n && !eq(out.t[out.n - 1], "SEP") && wi > 0 && ends_comma(words[wi - 1]) &&
        !strip_lower_is(w, "and", "point"))
      tl_push(&out, wi, nwt_static("SEP"));
    char* w0 = word_key(w);
    if (w0[0]) key_tokens(w0, wi, &out, &ordf);
    free(w0);
    if (out.n == n0) tl_push(&out, wi, nwt_static("<none>"));
  }
#define NUMTOK(t) (numtok_vsc(t) || IN((t), "half", "quarter", "dozen", "OH"))
  for (int i = 0; i < out.n; ++i)
    if (eq(out.t[i], "two") && i > 0 && NUMTOK(out.t[i - 1])) {
      char* l = lw(words[out.wi[i]]);
      if (IN(l, "second", "seconds")) out.t[i] = nwt_static("UNITW");
      free(l);
    }
  for (int i = 0; i < out.n; ++i)
    if (eq(out.t[i], "SEP")) {
      int any = 0;
      for (int k = i + 1; k < out.n && !any; ++k) any = NUMTOK(out.t[k]);
      if (!any) out.t[i] = nwt_static("UNITW");
    }
#undef NUMTOK
  const char** snap = malloc(sizeof(char*) * (size_t)(out.n + 1));
  int* real = malloc(sizeof(int) * (size_t)(out.n + 1));
  int nr = 0;
  for (int i = 0; i < out.n; ++i) {
    snap[i] = out.t[i];
    if (!eq(out.t[i], "<none>")) real[nr++] = i;
  }
  for (int k = 0; k < nr; ++k) {
    int i = real[k];
    const char* t = snap[i];
    const char* nxt = k + 1 < nr ? snap[real[k + 1]] : NULL;
    const char* prv = k ? snap[real[k - 1]] : NULL;
    if (eq(t, "a") && !ahead_ok(nxt)) out.t[i] = nwt_static("UNITW");
    if (ordf && IN(t, "half", "quarter") && k > 0 && val_of(prv) >= 0) out.t[i] = nwt_static("UNITW");
  }
  free(snap);
  free(real);
  if (ordf_out) *ordf_out = ordf;
  *toks = out.t;
  if (wis) *wis = out.wi;
  else free(out.wi);
  return out.n;
}

int arc_num_word(const char* w) {
  const char** t;
  int n = arc_num_tokens_of(&w, 1, &t, NULL);
  if (n < 0) return 0;
  free(t);
  return 1;
}

int arc_num_connector(const char* w) {
  const char** t;
  int n = arc_num_tokens_of(&w, 1, &t, NULL);
  if (n < 0) return 1;
  int all = 1;
  for (int i = 0; i < n; ++i)
    if (!IN(t[i], "and", "SEP", "a", "point", "NEG")) all = 0;
  free(t);
  return all;
}

/* textlayer.has_num */
int arc_num_has_num(const char* w) {
  const char** t;
  int n = arc_num_word_tokens(&w, 1, &t, NULL, NULL), any = 0;
  for (int i = 0; i < n; ++i)
    if (numtok_vsc(t[i]) || eq(t[i], "OH")) any = 1;
  free(t);
  return any;
}

/* ------------------------------------------------------------------ compose (numlib.py) */

/* numlib._compose_int */
static int compose_int(const char* const* tok, int n, pn* out) {
  long long total = 0, cur = 0;
  int seen = 0;
  for (int i = 0; i < n; ++i) {
    const char* t = tok[i];
    if (eq(t, "and")) continue;
    if (eq(t, "a")) {
      if (i + 1 < n && IN(tok[i + 1], "half", "quarter")) continue;
      cur += 1;
      seen = 1;
      continue;
    }
    if (eq(t, "quarter")) {
      *out = seen ? PF((double)(total + cur) + 0.25) : PF(0.25);
      return 1;
    }
    int v = val_of(t);
    long long s;
    if (v >= 0) {
      if (cur % 100 && v < 100 && cur % 10 == 0 && v < 10 && cur % 100 >= 20) cur += v;
      else if (cur % 100 && v >= 10) return 0;
      else if (cur % 10 && v < 10 && cur >= 1) return 0;
      else cur += v;
      seen = 1;
    } else if ((s = sc_of(t))) {
      if (s == 100) cur = (cur ? cur : 1) * 100;
      else {
        total += (cur ? cur : 1) * s;
        cur = 0;
      }
      seen = 1;
    } else if (eq(t, "dozen")) {
      cur = (cur ? cur : 1) * 12;
      seen = 1;
    } else if (eq(t, "half")) {
      *out = seen ? PF((double)(total + cur) + 0.5) : PF(0.5);
      return 1;
    } else return 0;
  }
  if (!seen) return 0;
  *out = PI(total + cur);
  return 1;
}

/* numlib._groups: the count (g filled), or -1 for None */
static int groups(const char* const* tok, int n, long long* g) {
  int ng = 0, c0 = 0, cn = 0;   /* the current group: tok[c0 .. c0+cn) (groups are consecutive slices) */
  int starts[1024], lens[1024];
  if (n > 1024) return -1;
  for (int i = 0; i < n; ++i) {
    const char* t = tok[i];
    if (eq(t, "OH")) {
      if (cn) {
        starts[ng] = c0;
        lens[ng++] = cn;
      }
      c0 = i;
      cn = 1;
      continue;
    }
    if (cn && !(cn == 1 && eq(tok[c0], "OH"))) {
      int v = val_of(t);
      const char* last = tok[c0 + cn - 1];
      if (is_tens(t) || v >= 10 || (in_list(last, UNITS_W + 1, 9) && v >= 0 && !is_tens(last))) {
        starts[ng] = c0;
        lens[ng++] = cn;
        cn = 0;
      }
    }
    if (!cn) c0 = i;
    cn++;
  }
  if (cn) {
    starts[ng] = c0;
    lens[ng++] = cn;
  }
  for (int k = 0; k < ng; ++k) {
    const char* const* x = tok + starts[k];
    pn v;
    if (eq(x[0], "OH")) {
      if (lens[k] > 1) {
        if (!compose_int(x + 1, lens[k] - 1, &v)) return -1;
      } else v = PI(0);
    } else if (!compose_int(x, lens[k], &v)) return -1;
    g[k] = (long long)v.d;   /* int(v) */
  }
  return ng;
}

/* numlib.compose: 1 and *v (and *v2 when *has2) for a value, 0 for None */
static int compose(const char* const* tok_in, int n, const char* kind, pn* v, pn* v2, int* has2) {
  *has2 = 0;
  if (n <= 0) return 0;
  const char** t2 = malloc(sizeof(char*) * (size_t)(n + 2));
  const char** t3 = malloc(sizeof(char*) * (size_t)(n + 2));
  int m = 0, ok = 0;
  for (int i = 0; i < n; ++i) {
    if (eq(tok_in[i], "UNITW")) {
      if (m && !eq(t2[m - 1], "SEP")) t2[m++] = "|";
      continue;
    }
    t2[m++] = tok_in[i];
  }
  while (m && eq(t2[m - 1], "|")) m--;
  int bar = -1;
  for (int i = 0; i < m && bar < 0; ++i)
    if (eq(t2[i], "|")) bar = i;
  int k = 0;
  if (bar >= 0) {
    int nrest = 0;
    for (int i = bar + 1; i < m; ++i) nrest += !eq(t2[i], "|");
    for (int i = 0; i < bar; ++i) t3[k++] = t2[i];
    if (nrest && IN(kind, "score", "ratio_odds", "clock")) {
      t3[k++] = "SEP";
      for (int i = bar + 1; i < m; ++i)
        if (!eq(t2[i], "|")) t3[k++] = t2[i];
    }
  } else
    for (int i = 0; i < m; ++i) t3[k++] = t2[i];
  /* tok = [t for t in t2 if t != '|'] */
  m = 0;
  for (int i = 0; i < k; ++i)
    if (!eq(t3[i], "|")) t2[m++] = t3[i];
  if (!m) goto done;
  int neg = eq(t2[0], "NEG");
  k = 0;
  for (int i = 0; i < m; ++i)
    if (!eq(t2[i], "NEG")) t3[k++] = t2[i];
  const char* const* tok = t3;
  n = k;
  int sep = -1, pt = -1;
  for (int i = 0; i < n; ++i) {
    if (sep < 0 && eq(tok[i], "SEP")) sep = i;
    if (pt < 0 && eq(tok[i], "point")) pt = i;
  }
  if (sep >= 0) {
    pn a, b, x2;
    int h;
    if (!compose(tok, sep, NULL, &a, &x2, &h) || !compose(tok + sep + 1, n - sep - 1, NULL, &b, &x2, &h)) goto done;
    if (IN(kind, "clock", "time_of_day")) *v = padd(pmul(a, PI(60)), b);
    else {
      *v = a;
      *v2 = b;
      *has2 = 1;
    }
    ok = 1;
    goto done;
  }
  if (pt >= 0) {
    pn a = PI(0);
    if (pt && !compose_int(tok, pt, &a)) goto done;
    if (pt + 1 >= n) goto done;
    char buf[1100];
    int o = snprintf(buf, sizeof buf, "%lld.", (long long)a.d);
    for (int i = pt + 1; i < n; ++i) {
      int dv = val_of(tok[i]);
      if (!((dv >= 0 && dv < 10) || eq(tok[i], "OH"))) goto done;
      if (o < (int)sizeof buf - 2) buf[o++] = (char)('0' + (dv >= 0 ? dv : 0));
    }
    buf[o] = 0;
    double f = arc_strtod(buf, NULL);
    *v = PF(neg ? -f : f);
    ok = 1;
    goto done;
  }
  pn vi;
  int vok = compose_int(tok, n, &vi);
  if (IN(kind, "id_pii", "ref_number", "jersey_number") && vok) {
    *v = vi;
    ok = 1;
    goto done;
  }
  if (!vok || IN(kind, "date", "clock", "time_of_day", "field_position", "score", "ratio_odds", "measure")) {
    long long g[1024];
    int ng = groups(tok, n, g);
    int two = ng == 2;
    if (two && IN(kind, "score", "ratio_odds") && (g[0] > g[1] ? g[0] : g[1]) < 20 && !vok) {
      *v = PI(g[0]);
      *v2 = PI(g[1]);
      *has2 = ok = 1;
      goto done;
    }
    if (two && eq(kind, "measure") && g[0] < 8 && g[1] < 12) {
      *v = PI(g[0]);
      *v2 = PI(g[1]);
      *has2 = ok = 1;
      goto done;
    }
    if (vok && IN(kind, "score", "ratio_odds", "measure", "field_position")) {
      *v = vi;
      ok = 1;
      goto done;
    }
    if (two && IN(kind, "clock", "time_of_day")) {
      *v = PI(g[0] * 60 + g[1]);
      ok = 1;
      goto done;
    }
    if (two && eq(kind, "date") && g[0] >= 10 && g[0] <= 20 && g[1] < 100) {
      *v = PI(g[0] * 100 + g[1]);
      ok = 1;
      goto done;
    }
    if (ng >= 2 && IN(kind, "id_pii", "ref_number", "jersey_number")) {
      char buf[2048];
      int o = 0;
      for (int i = 0; i < ng && o < (int)sizeof buf - 24; ++i) o += snprintf(buf + o, sizeof buf - (size_t)o, "%lld", g[i]);
      *v = o <= 18 ? PI(strtoll(buf, NULL, 10)) : (pn){arc_strtod(buf, NULL), 1};
      ok = 1;
      goto done;
    }
    if (!vok && two) {
      *v = PI(g[0] * 100 + g[1]);
      ok = 1;
      goto done;
    }
    if (!vok) goto done;
  }
  *v = neg ? (pn){-vi.d, vi.isint} : vi;
  ok = 1;
done:
  free(t2);
  free(t3);
  return ok;
}

arc_numval arc_num_compose(const char* const* tok, int n, const char* kind) {
  pn v = {0}, v2 = {0};
  int h = 0;
  int ok = compose(tok, n, kind, &v, &v2, &h);
  return mkval(ok, v, h, v2);
}

/* ------------------------------------------------------------------ finalize (numvalue.py) */

static const char* DUR_U[8] = {"second", "minute", "hour", "day", "week", "month", "year", "decade"};
static pn dur_sec(const char* u) {
  if (eq(u, "second")) return PI(1);
  if (eq(u, "minute")) return PI(60);
  if (eq(u, "hour")) return PI(3600);
  if (eq(u, "day")) return PI(86400);
  if (eq(u, "week")) return PI(7 * 86400);
  if (eq(u, "month")) return PF(30.44 * 86400);
  if (eq(u, "year")) return PF(365.25 * 86400);
  return PF(3652.5 * 86400);   /* decade */
}

/* numvalue.is_pair */
static int is_pair(const char* const* tok, int n) {
  const char** t = malloc(sizeof(char*) * (size_t)(n + 1));
  int m = 0, sep = 0;
  for (int i = 0; i < n; ++i)
    if (!eq(tok[i], "UNITW")) {
      if (eq(tok[i], "SEP")) sep = 1;
      if (!eq(tok[i], "NEG")) t[m++] = tok[i];
    }
  pn x;
  int r = sep || !compose_int(t, m, &x);
  free(t);
  return r;
}

static arc_numval finalize(const char* kind, const char* const* tok_in, int n, const char* unit) {
  arc_numval none = {0};
  const char** t = malloc(sizeof(char*) * (size_t)(n + 1));
  const char** tn = malloc(sizeof(char*) * (size_t)(n + 1));
  int m = 0;
  for (int i = 0; i < n; ++i)
    if (!eq(tok_in[i], "<none>")) t[m++] = tok_in[i];
  arc_numval r = none;
  pn v = {0}, v2 = {0};
  int h = 0;
  if (!m || !compose(t, m, kind, &v, &v2, &h)) goto done;
  if (eq(kind, "time_of_day")) {
    int k = 0, sep = 0, hund = 0;
    for (int i = 0; i < m; ++i)
      if (!IN(t[i], "UNITW", "NEG")) {
        tn[k++] = t[i];
        if (eq(t[i], "SEP")) sep = 1;
        if (IN(t[i], "hundred", "thousand")) hund = 1;
      }
    pn vi = {0};
    int vok = !sep && compose_int(tn, k, &vi);
    long long hh, mm;
    if (vok && hund) {
      if (!is_integer(vi)) goto done;
      long long iv = (long long)vi.d;
      if (iv % 100 >= 60 || iv / 100 > 24) goto done;
      hh = iv / 100;
      mm = iv % 100;
    } else if (vok && vi.d < 24 && is_integer(vi)) {
      hh = (long long)vi.d;
      mm = 0;
    } else if (v.d >= 0 && is_integer(v) && !vok) {
      long long iv = (long long)v.d;
      hh = iv / 60;
      mm = iv % 60;
    } else goto done;
    if (eq(unit, "pm") && hh < 12) hh += 12;
    if (eq(unit, "am") && hh == 12) hh = 0;
    if (!(hh >= 0 && hh <= 24 && mm >= 0 && mm < 60)) goto done;
    r = mkval(1, PI(hh % 24 * 3600 + mm * 60), 0, v2);
    goto done;
  }
  if (eq(kind, "duration")) {
    if (!in_list(unit, DUR_U, 8) || h) goto done;
    pn x = pmul(v, dur_sec(unit));
    if (!x.isint) x.d = round3(x.d);
    r = mkval(1, x, 0, v2);
    goto done;
  }
  if (eq(kind, "clock")) {
    if (h) goto done;
    if (is_pair(t, m)) r = mkval(1, v, 0, v2);
    else if (IN(unit, "minute", "hour")) r = mkval(1, pmul(v, dur_sec(unit)), 0, v2);
    else r = mkval(1, v, 0, v2);
    goto done;
  }
  if (eq(kind, "id_pii")) goto done;
  r = mkval(1, v, h, v2);
done:
  free(t);
  free(tn);
  return r;
}

arc_numval arc_num_finalize(const char* kind, const char* const* tok, int n, const char* unit) {
  return finalize(kind, tok, n, unit);
}

/* ------------------------------------------------------------------ unit_of (numvalue.py) */

/* Python's \w for a code point (ASCII exact; other scripts approximated: letters of Latin-1 / Latin Extended,
 * Greek, Cyrillic and beyond count, general punctuation and currency symbols do not) */
static int cp_word(unsigned cp) {
  if (cp < 0x80) return (cp >= '0' && cp <= '9') || (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') || cp == '_';
  if (cp == 0xAA || cp == 0xB5 || cp == 0xBA) return 1;
  if (cp >= 0xC0 && cp <= 0x24F) return cp != 0xD7 && cp != 0xF7;
  if (cp >= 0x2000 && cp <= 0x2BFF) return 0;
  if (cp >= 0x3000 && cp <= 0x303F) return 0;
  return cp >= 0x370;
}
static unsigned cp_at(const char* s, int i) {   /* the code point starting at byte i */
  const unsigned char* p = (const unsigned char*)s + i;
  if (p[0] < 0x80) return p[0];
  if ((p[0] & 0xE0) == 0xC0 && p[1]) return (unsigned)(p[0] & 0x1F) << 6 | (p[1] & 0x3F);
  if ((p[0] & 0xF0) == 0xE0 && p[1] && p[2]) return (unsigned)(p[0] & 0x0F) << 12 | (unsigned)(p[1] & 0x3F) << 6 | (p[2] & 0x3F);
  if ((p[0] & 0xF8) == 0xF0 && p[1] && p[2] && p[3])
    return (unsigned)(p[0] & 0x07) << 18 | (unsigned)(p[1] & 0x3F) << 12 | (unsigned)(p[2] & 0x3F) << 6 | (p[3] & 0x3F);
  return 0xFFFD;
}
static int word_before(const char* s, int i) {   /* \w just before byte i */
  if (i <= 0) return 0;
  int j = i - 1;
  while (j > 0 && ((unsigned char)s[j] & 0xC0) == 0x80) j--;
  return cp_word(cp_at(s, j));
}
static int word_at(const char* s, int i) { return s[i] && cp_word(cp_at(s, i)); }

/* re.search(r'\bX\.?\s?m\b|\d\sXm\b', raw) for X = 'a' or 'p' (the am / pm patterns of unit_of) */
static int ampm(const char* raw, char x) {
  int n = (int)strlen(raw);
  for (int i = 0; i < n; ++i) {
    if (raw[i] == x && !word_before(raw, i)) {
      for (int dot = 1; dot >= 0; --dot) {
        int j = i + 1;
        if (dot) {
          if (raw[j] != '.') continue;
          j++;
        }
        for (int sp = 1; sp >= 0; --sp) {
          int k = j;
          if (sp) {
            if (!is_ws((unsigned char)raw[k])) continue;
            k++;
          }
          if (raw[k] == 'm' && !word_at(raw, k + 1)) return 1;
        }
      }
    }
    if (isdig(raw[i])) {
      for (int sp = 1; sp >= 0; --sp) {
        int k = i + 1;
        if (sp) {
          if (!is_ws((unsigned char)raw[k])) continue;
          k++;
        }
        if (raw[k] == x && raw[k + 1] == 'm' && !word_at(raw, k + 2)) return 1;
      }
    }
  }
  return 0;
}

const char* arc_num_unit_of(const char* kind, const char* const* words, int n, const char* prev_word) {
  const char* res = NULL;
  if (eq(kind, "ref_number")) {
    char* l = lw(prev_word ? prev_word : "");
    res = "number";
    for (int i = 0; i < LEN(REFW); ++i)
      if (!strcmp(l, REFW[i].w)) res = REFW[i].u;
    free(l);
    return res;
  }
  char** lws = malloc(sizeof(char*) * (size_t)(n + 1));
  size_t tot = 1;
  for (int i = 0; i < n; ++i) {
    lws[i] = lw(words[i]);
    tot += strlen(words[i]) + 1;
  }
  char* raw = malloc(tot);
  size_t o = 0;
  for (int i = 0; i < n; ++i) {
    if (i) raw[o++] = ' ';
    for (const char* p = words[i]; *p; ++p)
      if (*p != '_') raw[o++] = (*p >= 'A' && *p <= 'Z') ? (char)(*p + 32) : *p;   /* lower, then '_' dropped */
  }
  raw[o] = 0;
  if (eq(kind, "money")) {
    if (strstr(raw, "\xe2\x82\xac") || strstr(raw, "euro")) res = "EUR";
    else if (strstr(raw, "\xc2\xa3") || strstr(raw, "pound") || strstr(raw, "quid")) res = "GBP";
    else {
      for (int i = 0; i < n && !res; ++i) {
        const char* u = uword(lws[i]);
        if (IN(u, "USD", "JPY", "INR", "CAD", "AUD")) res = u;
      }
      if (!res) res = (strchr(raw, '$') || strstr(raw, "dollar") || strstr(raw, "buck")) ? "USD" : "XXX";
    }
    goto done;
  }
  if (eq(kind, "time_of_day")) {
    if (ampm(raw, 'p')) res = "pm";
    else if (ampm(raw, 'a')) res = "am";
    else res = (strstr(raw, "o'clock") || strstr(raw, "oclock")) ? "oclock" : NULL;
    goto done;
  }
  if (eq(kind, "measure")) {
    int mi = 0, hr = 0;
    for (int i = 0; i < n; ++i) {
      mi |= IN(lws[i], "mile", "miles");
      hr |= IN(lws[i], "hour", "hr");
    }
    if (mi && hr) {
      res = "mph";
      goto done;
    }
  }
  for (int i = 0; i < n && !res; ++i) {
    const char* u = uword(lws[i]);
    if (!u) continue;
    int id = unit_id(u);
    if (IN(kind, "duration", "clock") && in_list(u, DUR_U, 8)) res = u;
    else if (eq(kind, "measure") && id >= unit_id("mile") && id <= unit_id("acre")) res = u;
    else if (IN(kind, "stat", "count") && id >= unit_id("points") && id <= unit_id("age")) res = u;
  }
done:
  for (int i = 0; i < n; ++i) free(lws[i]);
  free(lws);
  free(raw);
  return res;
}

/* ------------------------------------------------------------------ text_value (textlayer.py) */

arc_numval arc_text_value(const char* kind, const char* const* words, int n, const char* const* next, int n_next,
                          const char* prev_word, const char* unit_in, const char** unit_out) {
  arc_numval none = {0};
  if (unit_out) *unit_out = NULL;
  if (n <= 0) return none;
  const char** wt;
  int nt = arc_num_word_tokens(words, n, &wt, NULL, NULL);
  const char** tok = malloc(sizeof(char*) * (size_t)(nt + 1));
  int m = 0, any = 0;
  for (int i = 0; i < nt; ++i)
    if (!eq(wt[i], "<none>")) {
      tok[m++] = wt[i];
      if (numtok_vsc(wt[i]) || IN(wt[i], "OH", "half", "quarter", "dozen")) any = 1;
    }
  free(wt);
  if (!any) {
    free(tok);
    return none;
  }
  const char* unit = unit_in;
  int has_kind = kind && kind[0];
  if (!unit && has_kind) {
    const char** ws = malloc(sizeof(char*) * (size_t)(n + 1));
    memcpy(ws, words, sizeof(char*) * (size_t)n);
    int nw = n;
    if (n_next > 0) ws[nw++] = next[0];
    unit = arc_num_unit_of(kind, ws, nw, prev_word ? prev_word : "");
    free(ws);
  }
  arc_numval v;
  if (has_kind) v = finalize(kind, tok, m, unit);
  else {
    int k = 0;
    for (int i = 0; i < m; ++i)
      if (!eq(tok[i], "UNITW")) tok[k++] = tok[i];
    v = arc_num_compose(tok, k, NULL);
  }
  free(tok);
  if (unit_out) *unit_out = unit;
  return v.ok ? v : none;
}
