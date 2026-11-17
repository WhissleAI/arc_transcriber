/* Arc Transcriber Space read-outs on an utterance's words; line by line the code of arc_audio/quantity.py,
 * phonetic.py and nft2_post.py (see arc_post.h). Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include "arc_post.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arc_align.h"
#include "arc_base.h"
#include "arc_nft2.h"
#include "arc_text.h"

/* ------------------------------------------------------------------ small strings */

typedef struct {
  char* s;
  size_t n, cap;
} pbuf;
static void pb_putn(pbuf* b, const char* s, size_t n) {
  if (b->n + n + 1 > b->cap) {
    size_t c = (b->n + n + 1) * 2 + 16;
    char* t = arc_malloc(c);
    if (b->n) memcpy(t, b->s, b->n);
    arc_free(b->s);
    b->s = t;
    b->cap = c;
  }
  memcpy(b->s + b->n, s, n);
  b->n += n;
  b->s[b->n] = 0;
}
static void pb_put(pbuf* b, const char* s) { pb_putn(b, s, strlen(s)); }
static char* pb_take(pbuf* b) {
  if (!b->s) pb_putn(b, "", 0);
  return b->s;
}

char* arc_post_wk(const char* w) {
  pbuf b = {0};
  pb_put(&b, "");
  for (const unsigned char* p = (const unsigned char*)(w ? w : ""); *p; ++p) {
    int c = *p < 0x80 ? tolower(*p) : 0;
    if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '\'') {
      char ch = (char)c;
      pb_putn(&b, &ch, 1);
    }
  }
  return pb_take(&b);
}

static int in_set(const char* w, const char* const* set) {
  for (int i = 0; set[i]; ++i)
    if (!strcmp(w, set[i])) return 1;
  return 0;
}

/* ================================================================== quantities */

static const char* const ONES_W[] = {"zero", "oh", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", NULL};
static const int ONES_V[] = {0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
static const char* const TEENS_W[] = {"ten",     "eleven",  "twelve",    "thirteen", "fourteen",
                                      "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", NULL};
static const char* const TENS_W[] = {"twenty", "thirty", "forty", "fourty", "fifty", "sixty", "seventy", "eighty", "ninety", NULL};
static const int TENS_V[] = {20, 30, 40, 40, 50, 60, 70, 80, 90};
static const char* const SCALE_W[] = {"thousand", "million", "billion", NULL};
static const double SCALE_V[] = {1e3, 1e6, 1e9};
static const char* const MORE_W[] = {"more", "extra", "additional", "less", "fewer", NULL};
static const int MORE_V[] = {1, 1, 1, -1, -1};

static int idx_of(const char* w, const char* const* set) {
  for (int i = 0; set[i]; ++i)
    if (!strcmp(w, set[i])) return i;
  return -1;
}

typedef struct {
  const char* words;
  const char* type;
  const char* unit;
} unit_t;
/* quantity.UNITS, longest (in words) first, then in list order (Python's stable sort) */
static const unit_t UNITS[] = {
    {"kilometers per hour", "speed", "km/h"}, {"kilometres per hour", "speed", "km/h"}, {"km per hour", "speed", "km/h"},
    {"miles per hour", "speed", "mph"},       {"meters per second", "speed", "m/s"},    {"metres per second", "speed", "m/s"},
    {"km/h", "speed", "km/h"},                {"kph", "speed", "km/h"},                 {"mph", "speed", "mph"},
    {"percentage points", "percent", "%"},    {"per cent", "percent", "%"},             {"percent", "percent", "%"},
    {"%", "percent", "%"},                    {"pct", "percent", "%"},
    {"kilometers", "distance", "km"},         {"kilometer", "distance", "km"},          {"kilometres", "distance", "km"},
    {"kilometre", "distance", "km"},          {"km", "distance", "km"},                 {"klicks", "distance", "km"},
    {"klick", "distance", "km"},              {"meters", "distance", "m"},              {"meter", "distance", "m"},
    {"metres", "distance", "m"},              {"metre", "distance", "m"},               {"centimeters", "distance", "cm"},
    {"centimeter", "distance", "cm"},         {"centimetres", "distance", "cm"},        {"cm", "distance", "cm"},
    {"millimeters", "distance", "mm"},        {"millimetres", "distance", "mm"},        {"mm", "distance", "mm"},
    {"miles", "distance", "mi"},              {"mile", "distance", "mi"},               {"feet", "distance", "ft"},
    {"foot", "distance", "ft"},               {"ft", "distance", "ft"},                 {"inches", "distance", "in"},
    {"inch", "distance", "in"},               {"yards", "distance", "yd"},              {"yard", "distance", "yd"},
    {"light years", "distance", "ly"},        {"light year", "distance", "ly"},         {"astronomical units", "distance", "au"},
    {"seconds", "duration", "s"},             {"second", "duration", "s"},              {"secs", "duration", "s"},
    {"sec", "duration", "s"},                 {"minutes", "duration", "min"},           {"minute", "duration", "min"},
    {"mins", "duration", "min"},              {"hours", "duration", "h"},               {"hour", "duration", "h"},
    {"hrs", "duration", "h"},                 {"days", "duration", "d"},                {"day", "duration", "d"},
    {"weeks", "duration", "wk"},              {"week", "duration", "wk"},               {"months", "duration", "mo"},
    {"month", "duration", "mo"},              {"years", "duration", "yr"},              {"year", "duration", "yr"},
    {"degrees celsius", "temperature", "C"},  {"degrees centigrade", "temperature", "C"},
    {"degrees fahrenheit", "temperature", "F"}, {"degrees kelvin", "temperature", "K"},  {"celsius", "temperature", "C"},
    {"fahrenheit", "temperature", "F"},       {"kelvin", "temperature", "K"},           {"degrees", "degrees", "deg"},
    {"degree", "degrees", "deg"},             {"kilograms", "mass", "kg"},              {"kilogram", "mass", "kg"},
    {"kilos", "mass", "kg"},                  {"kg", "mass", "kg"},                     {"grams", "mass", "g"},
    {"gram", "mass", "g"},                    {"pounds", "mass", "lb"},                 {"pound", "mass", "lb"},
    {"lbs", "mass", "lb"},                    {"tons", "mass", "t"},                    {"tonnes", "mass", "t"},
    {"ton", "mass", "t"},                     {"kilowatt hours", "energy", "kWh"},      {"kilowatts", "power", "kW"},
    {"kilowatt", "power", "kW"},              {"watts", "power", "W"},                  {"watt", "power", "W"},
    {"volts", "voltage", "V"},                {"volt", "voltage", "V"},                 {"amps", "current", "A"},
    {"amp", "current", "A"},                  {"times", "multiplier", "x"}};
#define N_UNITS ((int)(sizeof UNITS / sizeof *UNITS))

typedef struct {
  char* t;
  int w;
} qtok;

static int is_digits_num(const char* t) {   /* ^\d+(?:,\d{3})*(?:\.\d+)?$ */
  const char* p = t;
  if (!isdigit((unsigned char)*p)) return 0;
  while (isdigit((unsigned char)*p)) p++;
  while (*p == ',') {
    for (int k = 1; k <= 3; ++k)
      if (!isdigit((unsigned char)p[k])) return 0;
    if (isdigit((unsigned char)p[4])) return 0;
    p += 4;
  }
  if (*p == '.') {
    p++;
    if (!isdigit((unsigned char)*p)) return 0;
    while (isdigit((unsigned char)*p)) p++;
  }
  return *p == 0;
}

static int keep_qchar(unsigned char c) {
  return c >= 0x80 || isalnum(c) || c == '_' || c == '%' || c == '.' || c == ',' || c == '\'' || c == '/' || c == '-';
}

/* quantity.tokens */
static int q_tokens(const char* const* words, int n, qtok** out) {
  int cap = 8, nt = 0;
  qtok* tk = arc_malloc(sizeof(qtok) * (size_t)cap);
#define QPUSH(str, len, wi)                                         \
  do {                                                              \
    if (nt == cap) {                                                \
      cap *= 2;                                                     \
      qtok* nn = arc_malloc(sizeof(qtok) * (size_t)cap);            \
      memcpy(nn, tk, sizeof(qtok) * (size_t)nt);                    \
      arc_free(tk);                                                 \
      tk = nn;                                                      \
    }                                                               \
    tk[nt].t = arc_malloc((size_t)(len) + 1);                       \
    memcpy(tk[nt].t, (str), (size_t)(len));                         \
    tk[nt].t[(len)] = 0;                                            \
    tk[nt++].w = (wi);                                              \
  } while (0)
  for (int i = 0; i < n; ++i) {
    pbuf b = {0};
    pb_put(&b, "");
    const unsigned char* p = (const unsigned char*)(words[i] ? words[i] : "");
    while (*p) {
      if (p[0] == 0xE2 && p[1] == 0x80 && p[2] == 0x99) {   /* ’ */
        pb_put(&b, "'");
        p += 3;
        continue;
      }
      unsigned char c = (unsigned char)(*p < 0x80 ? tolower(*p) : *p);
      if (keep_qchar(c)) pb_putn(&b, (const char*)&c, 1);
      p++;
    }
    char* t = b.s;
    size_t a = 0, e = strlen(t);
    while (a < e && strchr(".,'-", t[a])) a++;
    while (e > a && strchr(".,'-", t[e - 1])) e--;
    t[e] = 0;
    char* s = t + a;
    if (!*s) {
      arc_free(t);
      continue;
    }
    /* ^(\d+(?:,\d{3})*(?:\.\d+)?)(%|km|cm|mm|kg|m|s)?$ */
    static const char* const SUF[] = {"%", "km", "cm", "mm", "kg", "m", "s", NULL};
    int matched = 0;
    if (isdigit((unsigned char)*s)) {
      size_t L = strlen(s);
      for (int k = -1; SUF[k + 1] || k == -1; ++k) {
        const char* suf = k < 0 ? "" : SUF[k];
        size_t sl = strlen(suf);
        if (sl >= L || strcmp(s + L - sl, suf)) continue;
        char* num = arc_malloc(L - sl + 1);
        memcpy(num, s, L - sl);
        num[L - sl] = 0;
        int ok = is_digits_num(num);
        if (ok) {
          QPUSH(num, L - sl, i);
          if (sl) QPUSH(suf, sl, i);
          matched = 1;
        }
        arc_free(num);
        if (matched) break;
      }
    }
    if (!matched) {
      char* q = s;
      while (*q) {
        char* d = strchr(q, '-');
        size_t l = d ? (size_t)(d - q) : strlen(q);
        if (l) QPUSH(q, l, i);
        if (!d) break;
        q = d + 1;
      }
    }
    arc_free(t);
  }
#undef QPUSH
  *out = tk;
  return nt;
}

static int q_isones(const char* w) { return idx_of(w, ONES_W) >= 0; }
static int q_isteens(const char* w) { return idx_of(w, TEENS_W) >= 0; }
static int q_istens(const char* w) { return idx_of(w, TENS_W) >= 0; }
static int q_isscale(const char* w) { return idx_of(w, SCALE_W) >= 0; }

/* quantity._num: 0 none, else 1 with *v, *j, *is_int, *used */
static int q_num(qtok* tk, int n, int i, double* vo, int* jo, int* io, int* uo) {
  const char* t = tk[i].t;
  if (is_digits_num(t)) {
    char buf[128];
    size_t k = 0;
    for (const char* p = t; *p && k < sizeof buf - 1; ++p)
      if (*p != ',') buf[k++] = *p;
    buf[k] = 0;
    double v = strtod(buf, NULL);
    int j = i + 1;
    int s;
    if (j < n && (s = idx_of(tk[j].t, SCALE_W)) >= 0) {
      v *= SCALE_V[s];
      j++;
    }
    *vo = v;
    *jo = j;
    *io = floor(v) == v;
    *uo = 1;
    return 1;
  }
  enum { L_NONE, L_A, L_ONE, L_TEEN, L_TEN, L_HUNDRED, L_SCALE };
  long long total = 0, cur = 0;
  int j = i, last = L_NONE, used = 0;
  while (j < n) {
    const char* w = tk[j].t;
    int x;
    if (!strcmp(w, "a") && last == L_NONE && j + 1 < n &&
        (!strcmp(tk[j + 1].t, "hundred") || q_isscale(tk[j + 1].t))) {
      cur = 1;
      last = L_A;
    } else if ((x = idx_of(w, ONES_W)) >= 0 && strcmp(w, "oh")) {
      if (last == L_ONE || last == L_TEEN) break;
      if (last == L_TEN && cur % 10) break;
      cur += ONES_V[x];
      last = L_ONE;
    } else if ((x = idx_of(w, TEENS_W)) >= 0) {
      if (last == L_ONE || last == L_TEEN || last == L_TEN) break;
      cur += 10 + x;
      last = L_TEEN;
    } else if ((x = idx_of(w, TENS_W)) >= 0) {
      if (last == L_ONE || last == L_TEEN || last == L_TEN) break;
      cur += TENS_V[x];
      last = L_TEN;
    } else if (!strcmp(w, "hundred")) {
      if (last == L_HUNDRED || (cur % 100 == 0 && cur > 0 && last != L_A)) break;
      cur = (cur ? cur : 1) * 100;
      last = L_HUNDRED;
    } else if ((x = idx_of(w, SCALE_W)) >= 0) {
      if (last == L_NONE) break;
      total += (cur ? cur : 1) * (long long)SCALE_V[x];
      cur = 0;
      last = L_SCALE;
    } else if (!strcmp(w, "and") && (last == L_HUNDRED || last == L_SCALE) && j + 1 < n &&
               (q_isones(tk[j + 1].t) || q_isteens(tk[j + 1].t) || q_istens(tk[j + 1].t))) {
      j++;
      continue;
    } else
      break;
    used++;
    j++;
  }
  if (last == L_NONE || last == L_A) return 0;
  double v = (double)(total + cur);
  int is_int = 1;
  if (j + 1 < n && !strcmp(tk[j].t, "point") && q_isones(tk[j + 1].t)) {
    char buf[96];
    int k = j + 1, bl = snprintf(buf, sizeof buf, "%lld.", (long long)v);
    while (k < n && q_isones(tk[k].t)) {
      if (bl < (int)sizeof buf - 2) buf[bl++] = (char)('0' + ONES_V[idx_of(tk[k].t, ONES_W)]);
      k++;
    }
    buf[bl] = 0;
    v = strtod(buf, NULL);
    is_int = 0;
    j = k;
  } else if (j + 2 < n && !strcmp(tk[j].t, "and") && !strcmp(tk[j + 1].t, "a") && !strcmp(tk[j + 2].t, "half")) {
    v += 0.5;
    is_int = 0;
    j += 3;
  }
  *vo = v;
  *jo = j;
  *io = is_int;
  *uo = used;
  return 1;
}

/* quantity._unit: index into UNITS or -1; *je the token after it */
static int q_unit(qtok* tk, int n, int j, int* je) {
  int best = -1, bw = 0;
  for (int u = 0; u < N_UNITS; ++u) {
    char** w;
    int nw = arc_split_words(UNITS[u].words, &w);
    int ok = j + nw <= n;
    for (int k = 0; ok && k < nw; ++k) ok = !strcmp(tk[j + k].t, w[k]);
    arc_free_words(w, nw);
    if (ok && nw > bw) {   /* longest first; the first of equal length */
      best = u;
      bw = nw;
    }
  }
  if (best >= 0) *je = j + bw;
  return best;
}

static void fmt_num(double v, int is_int, char* out, size_t cap) {
  if (is_int) snprintf(out, cap, "%lld", (long long)v);
  else
    snprintf(out, cap, "%g", v);
}

int arc_post_quantities(const char* const* words, const char* const* tags, int nw, arc_qty** out) {
  qtok* tk;
  int n = q_tokens(words, nw, &tk);
  int cap = 4, nq = 0;
  arc_qty* q = arc_calloc(sizeof(arc_qty) * (size_t)cap);
  int i = 0;
  while (i < n) {
    int neg = 0, ok;
    double v;
    int j, is_int, used;
    if ((!strcmp(tk[i].t, "minus") || !strcmp(tk[i].t, "negative")) && i + 1 < n) {
      ok = q_num(tk, n, i + 1, &v, &j, &is_int, &used);
      neg = ok;
    } else
      ok = q_num(tk, n, i, &v, &j, &is_int, &used);
    int wi = tk[i].w;
    int skip = tags && wi < nw && tags[wi] && !strncmp(tags[wi], "ENTITY_", 7);
    if (!ok || skip) {
      i++;
      continue;
    }
    if (neg) v = -v;
    int rel = 0, je, x;
    if (j < n && (x = idx_of(tk[j].t, MORE_W)) >= 0 && j + 1 < n && q_unit(tk, n, j + 1, &je) >= 0) {
      rel = MORE_V[x];
      j++;
    }
    int u = j < n ? q_unit(tk, n, j, &je) : -1;
    const char *typ = "number", *unit = NULL;
    if (u >= 0) {
      typ = UNITS[u].type;
      unit = UNITS[u].unit;
      j = je;
    } else if (rel == 0 && j < n && (x = idx_of(tk[j].t, MORE_W)) >= 0) {
      rel = MORE_V[x];
      j++;
    }
    if (u < 0 && used == 1 && (!strcmp(tk[i].t, "one") || !strcmp(tk[i].t, "a")) && !neg && !rel) {
      i++;
      continue;
    }
    int wb = tk[i].w, we = tk[j - 1].w + 1;
    if (nq == cap) {
      cap *= 2;
      arc_qty* nn = arc_calloc(sizeof(arc_qty) * (size_t)cap);
      memcpy(nn, q, sizeof(arc_qty) * (size_t)nq);
      arc_free(q);
      q = nn;
    }
    arc_qty* r = &q[nq++];
    r->type = typ;
    r->unit = unit;
    r->value = is_int ? v : arc_pyround(v, 6);
    r->is_int = is_int;
    r->relative = rel;
    r->word_begin = wb;
    r->word_end = we;
    pbuf b = {0};
    pb_put(&b, "");
    for (int k = wb; k < we; ++k) {
      if (k > wb) pb_put(&b, " ");
      pb_put(&b, words[k] ? words[k] : "");
    }
    r->text = pb_take(&b);
    char nb[96], wr[160];
    fmt_num(v, is_int, nb, sizeof nb);
    if (!unit) snprintf(wr, sizeof wr, "%s", nb);
    else if (!strcmp(unit, "%") || !strcmp(unit, "x"))
      snprintf(wr, sizeof wr, "%s%s", nb, unit);
    else
      snprintf(wr, sizeof wr, "%s %s", nb, unit);
    r->written = arc_strdup(wr);
    i = j;
  }
  for (int k = 0; k < n; ++k) arc_free(tk[k].t);
  arc_free(tk);
  *out = q;
  return nq;
}

void arc_post_qty_free(arc_qty* q, int n) {
  for (int i = 0; i < n; ++i) {
    arc_free(q[i].text);
    arc_free(q[i].written);
  }
  arc_free(q);
}

/* ================================================================== sound-alike keys */

static const char* const PH_RULES[][2] = {{"tch", "C"}, {"sch", "sk"}, {"ph", "f"}, {"gh", "g"}, {"ck", "k"},
                                          {"qu", "kw"}, {"wh", "w"},   {"sh", "S"}, {"ch", "C"}, {"th", "T"},
                                          {"dg", "j"},  {"kn", "n"},   {"wr", "r"}, {"x", "ks"}, {"q", "k"}};

static char* str_replace_all(const char* s, const char* a, const char* b) {
  pbuf o = {0};
  pb_put(&o, "");
  size_t la = strlen(a);
  while (*s) {
    if (!strncmp(s, a, la)) {
      pb_put(&o, b);
      s += la;
    } else
      pb_putn(&o, s++, 1);
  }
  return pb_take(&o);
}

char* arc_phon_key(const char* text) {
  char* nm = arc_nft2_norm(text ? text : "");
  pbuf b = {0};
  pb_put(&b, "");
  for (const char* p = nm; *p; ++p)
    if (*p >= 'a' && *p <= 'z') pb_putn(&b, p, 1);
  arc_free(nm);
  char* s = pb_take(&b);
  for (size_t r = 0; r < sizeof PH_RULES / sizeof *PH_RULES; ++r) {
    char* t = str_replace_all(s, PH_RULES[r][0], PH_RULES[r][1]);
    arc_free(s);
    s = t;
  }
  size_t n = strlen(s);
  char* out = arc_malloc(n + 1);
  size_t k = 0;
  for (size_t i = 0; i < n; ++i) {
    char c = s[i], nx = i + 1 < n ? s[i + 1] : 0;
    if (c == 'c') c = (nx == 'e' || nx == 'i' || nx == 'y') ? 's' : 'k';
    else if (c == 'g' && (nx == 'e' || nx == 'i' || nx == 'y') && i > 0)
      c = 'j';
    else if (c == 'h')
      continue;
    else if (c == 'e' && i == n - 1 && n > 3)
      continue;
    else if (c == 'y' && i == 0)
      c = 'y';
    else if (strchr("aeiouy", c))
      c = 'A';
    if (k && out[k - 1] == c) continue;
    out[k++] = c;
  }
  out[k] = 0;
  arc_free(s);
  return out;
}

static const char* const PH_NEAR[] = {"bp", "dtT", "gkCj", "fvw", "szSCj", "mn", "lr", "A"};
static double ph_sub(char a, char b) {
  if (a == b) return 0.0;
  for (size_t g = 0; g < sizeof PH_NEAR / sizeof *PH_NEAR; ++g)
    if (strchr(PH_NEAR[g], a) && strchr(PH_NEAR[g], b)) return 0.4;
  return 1.0;
}
static double ph_indel(char c) { return (c == 'A' || c == 'y' || c == 'w') ? 0.5 : 1.0; }

static double ph_distance(const char* a, const char* b) {
  int n = (int)strlen(a), m = (int)strlen(b);
  double* prev = arc_malloc(sizeof(double) * (size_t)(m + 1));
  double* cur = arc_malloc(sizeof(double) * (size_t)(m + 1));
  prev[0] = 0;
  for (int j = 1; j <= m; ++j) prev[j] = prev[j - 1] + ph_indel(b[j - 1]);
  for (int i = 1; i <= n; ++i) {
    cur[0] = prev[0] + ph_indel(a[i - 1]);
    for (int j = 1; j <= m; ++j) {
      double x = prev[j] + ph_indel(a[i - 1]), y = cur[j - 1] + ph_indel(b[j - 1]),
             z = prev[j - 1] + ph_sub(a[i - 1], b[j - 1]);
      cur[j] = x < y ? (x < z ? x : z) : (y < z ? y : z);
    }
    double* t = prev;
    prev = cur;
    cur = t;
  }
  double d = prev[m];
  arc_free(prev);
  arc_free(cur);
  return d;
}

double arc_phon_similarity(const char* x, const char* y) {
  char *a = arc_phon_key(x), *b = arc_phon_key(y);
  double r = 0.0;
  if (a[0] && b[0]) {
    double wa = 0, wb = 0;
    for (const char* p = a; *p; ++p) wa += ph_indel(*p);
    for (const char* p = b; *p; ++p) wb += ph_indel(*p);
    double w = wa > wb ? wa : wb;
    r = 1.0 - ph_distance(a, b) / w;
    if (r < 0) r = 0;
  }
  arc_free(a);
  arc_free(b);
  return r;
}

/* ================================================================== guard */

void arc_guard_defaults(arc_guard_cfg* c) { *c = (arc_guard_cfg){-60.0, 6.0, 0.6, 0.35, 0.8, 0.4}; }

int arc_post_frame_dbs(const float* y, long n, double** out) {
  int nf = (int)((n + 1599) / 1600);
  double* d = arc_malloc(sizeof(double) * (size_t)(nf + 1));
  for (int f = 0; f < nf; ++f) {
    long a = (long)f * 1600, b = a + 1600 < n ? a + 1600 : n;
    double s = 0;
    for (long i = a; i < b; ++i) s += (double)y[i] * y[i];
    d[f] = b > a ? 20 * log10(sqrt(s / (double)(b - a)) + 1e-06) : -120.0;
  }
  *out = d;
  return nf;
}

static int cmp_dbl(const void* x, const void* y) {
  double a = *(const double*)x, b = *(const double*)y;
  return (a > b) - (a < b);
}

int arc_post_guard(const char* const* words, const double* start, const float* conf, int n, const float* y, long ny,
                   double t0, const arc_guard_cfg* c, int* keep, const char** why) {
  for (int i = 0; i < n; ++i) {
    keep[i] = 1;
    if (why) why[i] = NULL;
  }
  if (!n) return 0;
  double* dbs;
  int nf = arc_post_frame_dbs(y, ny, &dbs);
  double cs = 0;
  int nc = 0;
  for (int i = 0; i < n; ++i)
    if (conf[i] >= 0) cs += conf[i], nc++;
  double ve = -1;
  if (nf) {
    double* s = arc_malloc(sizeof(double) * (size_t)nf);
    memcpy(s, dbs, sizeof(double) * (size_t)nf);
    qsort(s, (size_t)nf, sizeof(double), cmp_dbl);
    double top = s[nf - 1], floor_ = s[(int)(0.1 * (nf - 1))];
    arc_free(s);
    if (top < c->silence_db || (nc && top - floor_ < c->flat_db && cs / nc < c->flat_conf)) {
      for (int i = 0; i < n; ++i) {
        keep[i] = 0;
        if (why) why[i] = "no_speech";
      }
      arc_free(dbs);
      return 0;
    }
    int last = -1;
    for (int f = 0; f < nf; ++f)
      if (dbs[f] > floor_ + 6.0) last = f;
    if (last >= 0) ve = t0 + (last + 1) * 0.1;
  }
  arc_free(dbs);
  int kept = 0;
  char* prev = NULL;
  for (int i = 0; i < n; ++i) {
    char* k = arc_post_wk(words[i]);
    if (conf[i] >= 0 && prev && k[0] && !strcmp(k, prev) && conf[i] < c->rep_conf) {
      keep[i] = 0;
      if (why) why[i] = "repeat";
      arc_free(k);
      continue;
    }
    if (conf[i] >= 0 && ve >= 0 && start[i] > ve + c->tail_s && conf[i] < c->tail_conf) {
      keep[i] = 0;
      if (why) why[i] = "tail";
      arc_free(k);
      continue;
    }
    kept++;
    arc_free(prev);
    prev = k;
  }
  arc_free(prev);
  return kept;
}

/* ================================================================== disfluency */

static const char* const FILLERS[] = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "ahh", "eh", "hmm", "hmmm", "hm", "mm", "mmm", NULL};
static const char* const YK_PREV[] = {"do", "did", "don't", "didn't", "does", "doesn't", "would", "if", "as", "what", "how", "let", NULL};
static const char* const YK_NEXT[] = {"what", "how", "where", "who", "why", "when", "that", "about", "it", "him", "her", "them", "me", "the", "a", "an", "this", NULL};
static const char* const IM_PREV[] = {"what", "do", "did", "don't", "didn't", "does", NULL};
static const char* const IM_NEXT[] = {"it", "that", "what", "this", "him", "her", "them", "you", NULL};

static int ends_dash(const char* w) {
  size_t l = strlen(w ? w : "");
  return l && w[l - 1] == '-';
}

void arc_post_disfluency(const char* const* words, int n, const char** tag) {
  char** k = arc_malloc(sizeof(char*) * (size_t)(n + 1));
  for (int i = 0; i < n; ++i) {
    k[i] = arc_post_wk(words[i]);
    tag[i] = in_set(k[i], FILLERS) ? "filler" : NULL;
  }
#define K(i, s) ((i) < n && !strcmp(k[i], s))
  int i = 0;
  while (i < n) {
    if (K(i, "you") && K(i + 1, "know") && K(i + 2, "what") && K(i + 3, "i") && K(i + 4, "mean")) {
      for (int j = i; j < i + 5; ++j) tag[j] = "filler";
      i += 5;
      continue;
    }
    if (K(i, "you") && K(i + 1, "know") && (i == 0 || !in_set(k[i - 1], YK_PREV)) && (i + 2 >= n || !in_set(k[i + 2], YK_NEXT)) && n > 2) {
      tag[i] = tag[i + 1] = "filler";
      i += 2;
      continue;
    }
    if (K(i, "i") && K(i + 1, "mean") && i + 2 < n && (i == 0 || !in_set(k[i - 1], IM_PREV)) && !in_set(k[i + 2], IM_NEXT)) {
      tag[i] = tag[i + 1] = "filler";
      i += 2;
      continue;
    }
    i++;
  }
#undef K
  for (i = 0; i < n - 1; ++i) {
    size_t li = strlen(k[i]), lj = strlen(k[i + 1]);
    if (!tag[i] && li && !strcmp(k[i], k[i + 1])) tag[i] = "repeat";
    else if (i + 3 < n && !tag[i] && li && !strcmp(k[i], k[i + 2]) && !strcmp(k[i + 1], k[i + 3]))
      tag[i] = tag[i + 1] = "repeat";
    else if (!tag[i] && li >= 2 && lj > li && !strncmp(k[i + 1], k[i], li) && (ends_dash(words[i]) || li >= 3))
      tag[i] = "fragment";
    else if (!tag[i] && ends_dash(words[i]))
      tag[i] = "fragment";
  }
  for (i = 0; i < n; ++i) arc_free(k[i]);
  arc_free(k);
}

/* ================================================================== sound-alike names */

static const char* const STOPW[] = {
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "am", "i", "you", "he", "she", "it", "we", "they", "me", "my", "your", "our", "us", "them", "this", "that", "these",
    "those", "what", "where", "when", "who", "why", "how", "which", "there", "here", "go", "goes", "going", "show",
    "take", "fly", "see", "find", "look", "zoom", "tell", "about", "please", "can", "could", "would", "will", "do",
    "does", "did", "not", "no", "yes", "yeah", "okay", "ok", "so", "then", "than", "just", "like", "um", "uh", "oh",
    "with", "from", "by", "up", "down", "out", "over", "into", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "more", "less", "some", "any", "all", "very", "too", "also", "its", "it's", "let's", "lets",
    "get", "give", "set", "send", "move", "turn", "drive", "put", "make", "have", "has", "had", "know", "think", "want",
    "need", NULL};

int arc_post_is_stop(const char* key) { return in_set(key, STOPW); }

void arc_almost_defaults(arc_almost_cfg* c) { *c = (arc_almost_cfg){0.6, 0.6, 12.0, 0.72, 3.0, 0.06, 4, 4, 0.7}; }

static char* letters_of(const char* s) {
  pbuf b = {0};
  pb_put(&b, "");
  for (const unsigned char* p = (const unsigned char*)(s ? s : ""); *p; ++p) {
    int c = *p < 0x80 ? tolower(*p) : 0;
    if (c >= 'a' && c <= 'z') {
      char ch = (char)c;
      pb_putn(&b, &ch, 1);
    }
  }
  return pb_take(&b);
}

double arc_almost_p(const char* heard, const char* form, const arc_almost_cfg* c) {
  double sp = arc_phon_similarity(heard, form);
  if (sp < c->min_sim) return 0.0;
  char *a = letters_of(heard), *b = letters_of(form);
  double sc = (a[0] && b[0]) ? arc_sm_ratio_bytes(a, b) : 0.0;
  arc_free(a);
  arc_free(b);
  char* k = arc_phon_key(form);
  int nl = 0;
  for (const char* p = k; *p; ++p) nl += *p != 'A';
  arc_free(k);
  double z = c->a * (sp - c->b) + c->c * (sc - 0.5) + c->d * ((nl < 10 ? nl : 10) - 5);
  return 1.0 / (1.0 + exp(-z));
}

/* nft2_post._suffix_ok: a spelling ending in a one-letter word needs as many heard words */
static int n_words_of(const char* s) {
  int n = 0, in = 0;
  for (; *s; ++s) {
    int sp = isspace((unsigned char)*s);
    if (!sp && !in) n++;
    in = !sp;
  }
  return n;
}

static int suffix_ok(const char* heard, const char* form) {
  int nf = n_words_of(form);
  if (nf < 2) return 1;
  const char* e = form + strlen(form);
  while (e > form && isspace((unsigned char)e[-1])) e--;
  const char* b = e;
  while (b > form && !isspace((unsigned char)b[-1])) b--;
  char* last = arc_malloc((size_t)(e - b) + 1);
  memcpy(last, b, (size_t)(e - b));
  last[e - b] = 0;
  char* lt = letters_of(last);
  int one = strlen(lt) == 1;
  arc_free(lt);
  arc_free(last);
  return !(one && n_words_of(heard) < nf);
}

int arc_almost_best(const char* heard, const arc_form* forms, int nf, const arc_almost_cfg* c, double* p) {
  int best = -1;
  double bp = 0;
  for (int i = 0; i < nf; ++i) {
    if (!suffix_ok(heard, forms[i].form)) continue;
    double q = arc_almost_p(heard, forms[i].form, c);
    if (q > 0 && (best < 0 || q > bp)) {
      best = i;
      bp = q;
    }
  }
  *p = bp;
  return best;
}

typedef struct {
  double p;
  int a, b, form;
} almost_cand;
static int cmp_cand(const void* x, const void* y) {
  const almost_cand *a = x, *b = y;
  if (a->p != b->p) return a->p > b->p ? -1 : 1;
  if (a->a != b->a) return a->a < b->a ? -1 : 1;
  return (a->b > b->b) - (a->b < b->b);
}
static int cmp_hit(const void* x, const void* y) {
  const arc_almost_hit *a = x, *b = y;
  return (a->word_begin > b->word_begin) - (a->word_begin < b->word_begin);
}

int arc_post_almost(const char* const* words, const float* conf, const int* common, int n, const int* taken,
                    const arc_form* forms, int nf, const arc_almost_cfg* c, arc_almost_hit** out) {
  *out = NULL;
  if (!nf || !n) return 0;
  char** k = arc_malloc(sizeof(char*) * (size_t)n);
  for (int i = 0; i < n; ++i) k[i] = arc_post_wk(words[i]);
  char** ex = arc_malloc(sizeof(char*) * (size_t)nf);
  for (int f = 0; f < nf; ++f) ex[f] = letters_of(forms[f].form);
  int cap = 16, nc = 0;
  almost_cand* cs = arc_malloc(sizeof(almost_cand) * (size_t)cap);
  for (int a = 0; a < n; ++a)
    for (int b = a + 1; b <= n && b <= a + c->max_words; ++b) {
      int skip = 0, allstop = 1, nconf = 0;
      double minc = 1e9;
      for (int j = a; j < b; ++j) {
        if (taken && taken[j]) skip = 1;
        if (k[j][0] && !in_set(k[j], STOPW) && !(common && common[j])) allstop = 0;
        if (conf && conf[j] >= 0 && !in_set(k[j], STOPW)) {   /* _unsure */
          nconf++;
          if (conf[j] < minc) minc = conf[j];
        }
      }
      if (skip || allstop || in_set(k[a], STOPW) || in_set(k[b - 1], STOPW)) continue;
      if (nconf && !(minc < c->max_conf)) continue;
      pbuf h = {0};
      pb_put(&h, "");
      for (int j = a; j < b; ++j) {
        if (j > a) pb_put(&h, " ");
        pb_put(&h, words[j] ? words[j] : "");
      }
      char* heard = pb_take(&h);
      char* lt = letters_of(heard);
      int bad = (int)strlen(lt) < c->min_chars;
      for (int f = 0; f < nf && !bad; ++f) bad = !strcmp(lt, ex[f]);
      arc_free(lt);
      double p;
      int bi = bad ? -1 : arc_almost_best(heard, forms, nf, c, &p);
      arc_free(heard);
      if (bi < 0 || p < c->min_p) continue;
      if (nc == cap) {
        cap *= 2;
        almost_cand* nn = arc_malloc(sizeof(almost_cand) * (size_t)cap);
        memcpy(nn, cs, sizeof(almost_cand) * (size_t)nc);
        arc_free(cs);
        cs = nn;
      }
      cs[nc++] = (almost_cand){p, a, b, bi};
    }
  qsort(cs, (size_t)nc, sizeof(almost_cand), cmp_cand);
  int* used = arc_calloc(sizeof(int) * (size_t)n);
  arc_almost_hit* h = arc_malloc(sizeof(arc_almost_hit) * (size_t)(nc + 1));
  int nh = 0;
  for (int i = 0; i < nc; ++i) {
    int clash = 0;
    for (int j = cs[i].a; j < cs[i].b; ++j) clash |= used[j];
    if (clash) continue;
    for (int j = cs[i].a; j < cs[i].b; ++j) used[j] = 1;
    h[nh++] = (arc_almost_hit){cs[i].a, cs[i].b, forms[cs[i].form].kb, arc_pyround(cs[i].p, 4)};
  }
  qsort(h, (size_t)nh, sizeof(arc_almost_hit), cmp_hit);
  arc_free(used);
  arc_free(cs);
  for (int i = 0; i < n; ++i) arc_free(k[i]);
  arc_free(k);
  for (int f = 0; f < nf; ++f) arc_free(ex[f]);
  arc_free(ex);
  *out = h;
  return nh;
}

/* ================================================================== end of turn */

static const char* const CONT_W[] = {"the", "a", "an", "to", "and", "or", "but", "of", "for", "with", "at", "in", "on",
                                     "from", "into", "onto", "is", "are", "was", "were", "my", "your", "our", "his",
                                     "her", "their", "this", "that", "um", "uh", "er", "erm", "like", "so", "because",
                                     "then", "if", "than", "very", "go", "set", "give", "send", "turn", "move", "drive",
                                     "put", "make", "what", "where", "how", "who", "why", "when", "which", "more",
                                     "less", "by", "about", "can", "could", "would", "will", "should", "let", "let's", NULL};
static const char* const DONE_W[] = {"yes", "no", "yeah", "yep", "nope", "okay", "ok", "roger", "copy", "affirmative",
                                     "negative", "done", "stop", "thanks", "thank", "you", "please", "correct",
                                     "right", "wrong", NULL};

double arc_post_eot(const char* const* words, int n, double pause_s) {
  if (!n) return 0.0;
  char** k = arc_malloc(sizeof(char*) * (size_t)n);
  for (int i = 0; i < n; ++i) k[i] = arc_post_wk(words[i]);
  int short_ = 0;
  if (n <= 3) {
    int all = 1, head = 1;
    for (int i = 0; i < n; ++i) all &= in_set(k[i], DONE_W);
    for (int i = 0; i < n - 2; ++i) head &= in_set(k[i], DONE_W);
    short_ = all || (n >= 2 && !strcmp(k[n - 2], "copy") && !strcmp(k[n - 1], "that") && head);
  }
  double p;
  if (short_) p = 1.0 / (1.0 + exp(-(pause_s - 0.3) / 0.08));
  else {
    p = 1.0 / (1.0 + exp(-(pause_s - 0.45) / 0.08));
    if (in_set(k[n - 1], CONT_W)) p *= 0.35;
  }
  for (int i = 0; i < n; ++i) arc_free(k[i]);
  arc_free(k);
  return arc_pyround(p, 4);
}
