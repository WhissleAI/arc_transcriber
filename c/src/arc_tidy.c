/* Output tidying, as arc_audio/tidy.py. See arc_tidy.h.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include "arc_tidy.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "arc_base.h"

/* one UTF-8 character at s: its code point and length */
static int cp_at(const char* s, int* len) {
  const unsigned char* u = (const unsigned char*)s;
  if (u[0] < 0x80) {
    *len = 1;
    return u[0];
  }
  if ((u[0] >> 5) == 6 && u[1]) {
    *len = 2;
    return (u[0] & 31) << 6 | (u[1] & 63);
  }
  if ((u[0] >> 4) == 14 && u[1] && u[2]) {
    *len = 3;
    return (u[0] & 15) << 12 | (u[1] & 63) << 6 | (u[2] & 63);
  }
  if ((u[0] >> 3) == 30 && u[1] && u[2] && u[3]) {
    *len = 4;
    return (u[0] & 7) << 18 | (u[1] & 63) << 12 | (u[2] & 63) << 6 | (u[3] & 63);
  }
  *len = 1;
  return u[0];
}

/* Python's \w, for the characters transcripts hold: ASCII letters, digits, '_', and letters beyond ASCII (not
 * Latin-1 punctuation or symbols, general punctuation, or currency signs) */
static int is_word(int c) {
  if (c < 0x80) return isalnum(c) || c == '_';
  if (c < 0xC0) return c == 0xAA || c == 0xB5 || c == 0xBA || c == 0xB2 || c == 0xB3 || c == 0xB9 || c == 0xBC || c == 0xBD || c == 0xBE;
  if (c == 0xD7 || c == 0xF7) return 0;
  if (c >= 0x2000 && c <= 0x206F) return 0;   /* general punctuation */
  if (c >= 0x20A0 && c <= 0x20CF) return 0;   /* currency */
  if (c >= 0x3000 && c <= 0x303F) return 0;   /* CJK punctuation */
  return 1;
}

static int lead_keep(int c) { return is_word(c) || c == '$' || c == 0xA3 || c == 0x20AC || c == '%' || c == '#' || c == '@' || c == '+'; }
static int tail_keep(int c) { return is_word(c) || c == '%' || c == '\''; }

void arc_clean_word(const char* w, char* out) {
  /* str(t).strip() first */
  while (*w && isspace((unsigned char)*w)) w++;
  size_t n = strlen(w);
  while (n && isspace((unsigned char)w[n - 1])) n--;
  /* leading */
  size_t a = 0;
  while (a < n) {
    int l, c = cp_at(w + a, &l);
    if (lead_keep(c)) break;
    a += (size_t)l;
  }
  /* trailing: the last run of characters to drop */
  size_t b = n, cut = n, i = a;
  while (i < n) {
    int l, c = cp_at(w + i, &l);
    if (tail_keep(c)) cut = n;
    else if (cut == n) cut = i;
    i += (size_t)l;
  }
  b = cut;
  if (b < a) b = a;
  memcpy(out, w + a, b - a);
  out[b - a] = 0;
}

char* arc_clean_text(const char* t) {
  char** w;
  int n = arc_split_words_tidy(t, &w);
  size_t cap = strlen(t ? t : "") + 2, k = 0;
  char* out = arc_malloc(cap);
  char* tmp = arc_malloc(cap);
  out[0] = 0;
  for (int i = 0; i < n; ++i) {
    arc_clean_word(w[i], tmp);
    if (!tmp[0]) continue;
    size_t l = strlen(tmp);
    if (k) out[k++] = ' ';
    memcpy(out + k, tmp, l);
    k += l;
    out[k] = 0;
  }
  for (int i = 0; i < n; ++i) arc_free(w[i]);
  arc_free(w);
  arc_free(tmp);
  return out;
}

static void lower(char* s) {
  for (; *s; ++s) *s = (char)tolower((unsigned char)*s);
}

static const char* NUMWORDS[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                                 "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                                 "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
                                 "eighty", "ninety", "hundred", "thousand", "million", "billion", "half", "quarter",
                                 "dozen", "first", "second", "third", "fourth", "fifth", "point", "oh", NULL};
static const char* ORDINALS[] = {"sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
                                 "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth",
                                 "twentieth", "thirtieth", "fortieth", "fiftieth", "sixtieth", "seventieth",
                                 "eightieth", "ninetieth", "hundredth", "thousandth", "millionth", "nil", "none",
                                 "nothing", "love", "twice", "once", NULL};
static const char* TENS[] = {"twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", NULL};
static const char* UNIT_WORDS[] = {
    "percent", "per", "cent", "points", "point", "pts", "dollars", "dollar", "cents", "euros", "euro", "pounds", "pound",
    "yen", "bucks", "k", "million", "billion", "thousand", "hundred", "seconds", "second", "minutes", "minute", "hours",
    "hour", "days", "day", "weeks", "week", "months", "month", "years", "year", "yards", "yard", "feet", "foot",
    "inches", "inch", "miles", "mile", "meters", "metres", "meter", "kilometers", "km", "kg", "mph", "am", "pm", "a.m",
    "p.m", "o'clock", "oclock", "st", "nd", "rd", "th", "to", "and", "of", "out", "by", "a", "an", "x", "times",
    "percentage", "degrees", "degree", "goals", "goal", "runs", "run", "wickets", "overs", "rebounds", "assists",
    "shots", "kills", "games", "sets", "laps", NULL};
static const char* LEAD_WORDS[] = {"$", "\xC2\xA3", "\xE2\x82\xAC", "minus", "negative", "number", "no", NULL};

static int in_list(const char* w, const char* const* l) {
  for (; *l; ++l)
    if (!strcmp(w, *l)) return 1;
  return 0;
}

int arc_numeric_word(const char* w) {
  size_t n = strlen(w);
  char* c = arc_malloc(n + 1);
  arc_clean_word(w, c);
  lower(c);
  int r = 0;
  if (!c[0]) r = 0;
  else if (in_list(c, NUMWORDS) || in_list(c, ORDINALS)) r = 1;
  else {
    for (const char* p = c; *p && !r; ++p)
      if (isdigit((unsigned char)*p)) r = 1;   /* any digit (covers \d[\d,.:-]* too) */
    for (int k = 0; !r && TENS[k]; ++k) {   /* (twenty|...|ninety)-\w+ */
      size_t l = strlen(TENS[k]);
      if (!strncmp(c, TENS[k], l) && c[l] == '-' && c[l + 1]) {
        int ok = 1;
        for (const char* p = c + l + 1; *p;) {
          int len, cp = cp_at(p, &len);
          if (!is_word(cp)) ok = 0;
          p += len;
        }
        r = ok;
      }
    }
  }
  arc_free(c);
  return r;
}

/* str.split() */
int arc_split_words_tidy(const char* t, char*** out) {
  int cap = 8, n = 0;
  char** w = arc_malloc(sizeof(char*) * (size_t)cap);
  const char* p = t ? t : "";
  while (*p) {
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
    w[n++][q - p] = 0;
    p = q;
  }
  *out = w;
  return n;
}

static int clean_lower_in(const char* w, const char* const* l) {
  char* c = arc_malloc(strlen(w) + 1);
  arc_clean_word(w, c);
  lower(c);
  int r = in_list(c, l);
  arc_free(c);
  return r;
}

char* arc_tidy_number(const char* heard) {
  char** w;
  int n = arc_split_words_tidy(heard, &w), i = 0, j = n - 1, any = 0;
  for (int k = 0; k < n; ++k)
    if (arc_numeric_word(w[k])) {
      if (!any) i = k;
      j = k;
      any = 1;
    }
  if (!any) {
    i = 0;
    j = n - 1;
  } else {
    while (j + 1 < n && clean_lower_in(w[j + 1], UNIT_WORDS)) j++;
    while (i > 0 && clean_lower_in(w[i - 1], LEAD_WORDS)) i--;
  }
  size_t cap = strlen(heard ? heard : "") + 2, k = 0;
  char* joined = arc_malloc(cap);
  joined[0] = 0;
  for (int x = i; x <= j && x < n; ++x) {
    size_t l = strlen(w[x]);
    if (k) joined[k++] = ' ';
    memcpy(joined + k, w[x], l);
    k += l;
    joined[k] = 0;
  }
  char* out = arc_clean_text(joined);
  arc_free(joined);
  for (int x = 0; x < n; ++x) arc_free(w[x]);
  arc_free(w);
  return out;
}

int arc_keep_number(int has_value, const char* heard) {
  if (has_value) return 1;
  char** w;
  int n = arc_split_words_tidy(heard, &w), r = 0;
  for (int k = 0; k < n; ++k) {
    if (arc_numeric_word(w[k])) r = 1;
    arc_free(w[k]);
  }
  arc_free(w);
  return r;
}

/* ------------------------------------------------------------------ names */

static int capital(const char* w) {
  char* c = arc_malloc(strlen(w) + 1);
  arc_clean_word(w, c);
  int r = c[0] && isupper((unsigned char)c[0]);
  arc_free(c);
  return r;
}

/* is clean(w).lower() among the entity's cleaned lower-case words */
static int in_entity(const char* w, char** ent, int ne) {
  char* c = arc_malloc(strlen(w) + 1);
  arc_clean_word(w, c);
  lower(c);
  int r = 0;
  for (int i = 0; i < ne && !r; ++i) r = !strcmp(c, ent[i]);
  arc_free(c);
  return r;
}

void arc_tidy_name(arc_tname* r) {
  if (r->words && r->n_words > 1) {
    char** ent = NULL;
    int ne = 0;
    if (r->entity && *r->entity) {
      ne = arc_split_words_tidy(r->entity, &ent);
      for (int i = 0; i < ne; ++i) {
        char* c = arc_malloc(strlen(ent[i]) + 1);
        arc_clean_word(ent[i], c);
        lower(c);
        arc_free(ent[i]);
        ent[i] = c;
      }
    }
    int a = 0, b = r->n_words;   /* words [a, b) */
    for (;;) {
      if (b - a > 1) {
        int k = b - 1;
        char* c = arc_malloc(strlen(r->words[k].w) + 1);
        arc_clean_word(r->words[k].w, c);
        int d;
        if (!c[0]) d = 1;
        else if (ne) {
          int others = 0;
          for (int j = a; j < b; ++j)
            if (j != k && in_entity(r->words[j].w, ent, ne)) others = 1;
          d = !in_entity(r->words[k].w, ent, ne) && others;
        } else {
          int others = 0;
          for (int j = a; j < b; ++j)
            if (j != k && capital(r->words[j].w)) others = 1;
          d = !capital(r->words[k].w) && others;
        }
        arc_free(c);
        if (d) {
          b--;
          continue;
        }
      }
      break;
    }
    for (;;) {
      if (b - a > 1) {
        int k = a;
        char* c = arc_malloc(strlen(r->words[k].w) + 1);
        arc_clean_word(r->words[k].w, c);
        int d;
        if (!c[0]) d = 1;
        else if (ne) {
          int others = 0;
          for (int j = a; j < b; ++j)
            if (j != k && in_entity(r->words[j].w, ent, ne)) others = 1;
          d = !in_entity(r->words[k].w, ent, ne) && others;
        } else {
          int others = 0;
          for (int j = a; j < b; ++j)
            if (j != k && capital(r->words[j].w)) others = 1;
          d = !capital(r->words[k].w) && others;
        }
        arc_free(c);
        if (d) {
          a++;
          continue;
        }
      }
      break;
    }
    if (b - a != r->n_words) {
      r->words += a;
      r->n_words = b - a;
      r->start = r->words[0].start;
      r->end = r->words[r->n_words - 1].end;
      size_t cap = 2;
      for (int i = 0; i < r->n_words; ++i) cap += strlen(r->words[i].w) + 1;
      char* h = arc_malloc(cap);
      h[0] = 0;
      for (int i = 0; i < r->n_words; ++i) {
        if (i) strcat(h, " ");
        strcat(h, r->words[i].w);
      }
      arc_free(r->heard);
      r->heard = h;
    }
    for (int i = 0; i < ne; ++i) arc_free(ent[i]);
    arc_free(ent);
  }
  char* c = arc_clean_text(r->heard ? r->heard : "");
  arc_free(r->heard);
  r->heard = c;
}

int arc_same_mention(const arc_tname* a, const arc_tname* b) {
  double ov = (a->end < b->end ? a->end : b->end) - (a->start > b->start ? a->start : b->start);
  if (ov <= 0) return 0;
  double da = a->end - a->start, db = b->end - b->start;
  double sh = da < db ? da : db;
  if (sh < 1e-6) sh = 1e-6;
  if (ov / sh < 0.5) return 0;
  if ((a->entity && *a->entity) || (b->entity && *b->entity))
    return a->entity && b->entity && !strcmp(a->entity, b->entity);
  char* x = arc_clean_text(a->heard ? a->heard : "");
  char* y = arc_clean_text(b->heard ? b->heard : "");
  lower(x);
  lower(y);
  int r = !strcmp(x, y);
  arc_free(x);
  arc_free(y);
  return r;
}

static int via_rank(const char* v) {
  if (!v) return 1;
  if (!strcmp(v, "exact")) return 0;
  if (!strcmp(v, "snap")) return 2;
  if (!strcmp(v, "fallback")) return 3;
  return 1;
}

int arc_dedupe_names(arc_tname* r, int n) {
  int* ord = arc_malloc(sizeof(int) * (size_t)(n + 1));
  for (int i = 0; i < n; ++i) ord[i] = i;
  /* stable insertion sort by rank */
  for (int i = 1; i < n; ++i) {
    int x = ord[i], j = i - 1;
    for (; j >= 0; --j) {
      const arc_tname *a = &r[ord[j]], *b = &r[x];
      int la = !(a->state && !strcmp(a->state, "linked")), lb = !(b->state && !strcmp(b->state, "linked"));
      int greater = la != lb ? la > lb : via_rank(a->via) != via_rank(b->via) ? via_rank(a->via) > via_rank(b->via) : -a->p > -b->p;
      if (!greater) break;
      ord[j + 1] = ord[j];
    }
    ord[j + 1] = x;
  }
  arc_tname* keep = arc_malloc(sizeof(arc_tname) * (size_t)(n + 1));
  unsigned char* used = arc_calloc((size_t)n + 1);
  int nk = 0;
  for (int i = 0; i < n; ++i) {
    const arc_tname* x = &r[ord[i]];
    int dup = 0;
    for (int k = 0; k < nk && !dup; ++k) dup = arc_same_mention(x, &keep[k]);
    if (dup) continue;
    keep[nk++] = *x;
    used[ord[i]] = 1;
  }
  for (int i = 0; i < n; ++i)
    if (!used[i]) arc_free(r[i].heard);
  /* sorted by (start, end), stable */
  for (int i = 1; i < nk; ++i) {
    arc_tname x = keep[i];
    int j = i - 1;
    while (j >= 0 && (keep[j].start > x.start || (keep[j].start == x.start && keep[j].end > x.end))) {
      keep[j + 1] = keep[j];
      --j;
    }
    keep[j + 1] = x;
  }
  memcpy(r, keep, sizeof(arc_tname) * (size_t)nk);
  arc_free(keep);
  arc_free(used);
  arc_free(ord);
  return nk;
}
