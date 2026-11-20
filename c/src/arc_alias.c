/* A port of arc_audio kbsnap.py AliasIndex / akey / trim. Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see
 * LICENSE. */
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "arc_alias.h"
#include "arc_base.h"

/* kbsnap.py IGNORE = COMMAND_WORDS | STOPWORDS (150 words, sorted for bsearch) */
static const char* const IGNORE_W[] = {
    "a", "about", "add", "after", "again", "ah", "all", "also", "an", "and", "any", "as", "ask", "at", "be", "before",
    "book", "books", "buy", "by", "call", "called", "can", "close", "continue", "could", "delete", "did", "do",
    "does", "down", "er", "every", "find", "for", "forward", "from", "get", "give", "go", "had", "has", "have", "her",
    "here", "hey", "his", "how", "i", "in", "into", "is", "it", "its", "just", "launch", "let", "let's", "lets",
    "like", "listen", "look", "louder", "me", "message", "more", "most", "mute", "my", "named", "need", "next", "no",
    "not", "now", "of", "oh", "ok", "okay", "on", "one", "open", "or", "order", "our", "over", "pause", "play",
    "playing", "please", "previous", "put", "queue", "quieter", "read", "reading", "remind", "remove", "repeat",
    "resume", "rewind", "save", "search", "send", "set", "shall", "share", "show", "shuffle", "skip", "so", "some",
    "something", "start", "stop", "take", "tell", "text", "that", "the", "their", "then", "there", "these", "this",
    "those", "to", "too", "turn", "uh", "um", "unmute", "up", "very", "volume", "want", "watch", "what", "when",
    "where", "which", "who", "why", "will", "with", "would", "yeah", "yes", "you", "your",
};
/* nametext.py COMMON (428 words, sorted for bsearch) */
static const char* const COMMON_W[] = {
    "a", "about", "above", "after", "again", "against", "all", "almost", "along", "already", "also", "although",
    "always", "am", "among", "an", "and", "another", "any", "anyone", "anything", "are", "around", "as", "ask",
    "asked", "assist", "at", "away", "back", "bad", "ball", "basket", "be", "became", "because", "become", "been",
    "before", "began", "behind", "being", "below", "best", "better", "between", "big", "block", "both", "bring",
    "brought", "but", "by", "call", "called", "came", "can", "can't", "cannot", "case", "change", "coach", "coaches",
    "come", "comes", "coming", "corner", "could", "couldn't", "course", "court", "crowd", "day", "days", "did",
    "didn't", "different", "do", "does", "doesn't", "doing", "don't", "done", "down", "downs", "dunk", "during",
    "each", "early", "either", "else", "end", "enough", "even", "ever", "every", "everyone", "everything", "fact",
    "fans", "far", "feel", "felt", "few", "field", "find", "first", "five", "floor", "for", "foul", "found", "four",
    "free", "from", "full", "game", "games", "gave", "get", "gets", "getting", "give", "given", "go", "goal", "goals",
    "goes", "going", "gone", "good", "got", "great", "had", "half", "hand", "happen", "happened", "hard", "has",
    "hasn't", "have", "haven't", "having", "he", "head", "hear", "heard", "help", "her", "here", "hers", "herself",
    "high", "him", "himself", "his", "hit", "hold", "home", "how", "however", "i", "i'm", "ice", "if", "in", "inning",
    "inside", "instead", "into", "is", "isn't", "it", "it's", "its", "itself", "just", "keep", "kick", "kind", "knew",
    "know", "known", "last", "late", "later", "layup", "lead", "least", "left", "less", "let", "let's", "life",
    "like", "line", "little", "long", "look", "looked", "looking", "lot", "lots", "made", "make", "makes", "making",
    "man", "many", "may", "maybe", "me", "mean", "might", "mind", "minute", "minutes", "more", "most", "move", "much",
    "must", "my", "myself", "name", "need", "needs", "net", "never", "new", "next", "nice", "night", "no", "none",
    "nor", "not", "nothing", "now", "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "only",
    "open", "or", "other", "others", "our", "out", "over", "overs", "own", "part", "pass", "people", "perhaps",
    "period", "pitch", "place", "play", "played", "player", "players", "playing", "point", "points", "put", "quarter",
    "quite", "rather", "really", "rebound", "right", "run", "runs", "said", "same", "save", "saw", "say", "says",
    "score", "scored", "season", "second", "see", "seem", "seemed", "seen", "set", "several", "shall", "she", "she's",
    "shot", "shots", "should", "show", "side", "since", "six", "small", "so", "some", "someone", "something",
    "sometimes", "soon", "sort", "start", "started", "steal", "still", "stop", "such", "sure", "take", "taken",
    "talk", "talking", "team", "ten", "than", "thank", "thanks", "that", "that's", "the", "their", "them",
    "themselves", "then", "there", "there's", "these", "they", "they're", "thing", "things", "think", "thinking",
    "third", "this", "those", "though", "thought", "three", "three-pointer", "through", "time", "timeout", "timeouts",
    "times", "to", "today", "together", "told", "tonight", "too", "took", "top", "toward", "try", "trying", "turn",
    "two", "under", "until", "up", "upon", "us", "use", "used", "very", "want", "wanted", "was", "wasn't", "watch",
    "way", "we", "week", "well", "went", "were", "weren't", "what", "what's", "when", "where", "whether", "which",
    "while", "who", "whole", "whose", "why", "wicket", "will", "win", "with", "within", "without", "won't", "word",
    "work", "world", "would", "wouldn't", "yard", "yards", "yeah", "year", "years", "yes", "yet", "you", "you're",
    "your", "yours",
};

static int cmp_str(const void* a, const void* b) { return strcmp(*(const char* const*)a, *(const char* const*)b); }
static int in_set(const char* const* set, size_t n, const char* w) { return bsearch(&w, set, n, sizeof *set, cmp_str) != NULL; }
#define NIGN (sizeof IGNORE_W / sizeof *IGNORE_W)
#define NCOM (sizeof COMMON_W / sizeof *COMMON_W)
/* kbsnap.ignorable on an akey word (already lower-case alphanumerics, so nrm leaves it as it is) */
static int ignorable(const char* w) { return in_set(IGNORE_W, NIGN, w); }

static int is_space(unsigned char c) { return c == ' ' || (c >= '\t' && c <= '\r') || c == 0x1c || c == 0x1d || c == 0x1e || c == 0x1f; }

char* arc_alias_akey(const char* t) {
  size_t L = t ? strlen(t) : 0;
  char* out = arc_malloc(L + 1);
  size_t o = 0;
  char* w = arc_malloc(L + 1);
  for (size_t i = 0; i < L;) {
    while (i < L && is_space((unsigned char)t[i])) i++;
    size_t a = i;
    while (i < L && !is_space((unsigned char)t[i])) i++;
    if (i == a) break;
    /* w.lower().strip('.,?!;:"()[]') */
    size_t b = i;
    while (a < b && strchr(".,?!;:\"()[]", t[a])) a++;
    while (b > a && strchr(".,?!;:\"()[]", t[b - 1])) b--;
    size_t n = 0;
    for (size_t k = a; k < b; ++k) w[n++] = (char)tolower((unsigned char)t[k]);
    w[n] = 0;
    /* re.sub("['’]s$", '') */
    if (n >= 2 && w[n - 1] == 's' && w[n - 2] == '\'') n -= 2;
    else if (n >= 4 && w[n - 1] == 's' && !memcmp(w + n - 4, "\xe2\x80\x99", 3)) n -= 4;
    /* re.sub('[^a-z0-9]', '') */
    size_t start = o;
    if (o) out[o++] = ' ';
    size_t m = 0;
    for (size_t k = 0; k < n; ++k)
      if ((w[k] >= 'a' && w[k] <= 'z') || (w[k] >= '0' && w[k] <= '9')) out[o++] = w[k], m++;
    if (!m) o = start;   /* an empty word adds nothing */
  }
  out[o] = 0;
  arc_free(w);
  return out;
}

typedef struct {
  char* key;
  int* rows;   /* distinct, in insertion order */
  int nr, cap;
} akey_ent;

typedef struct {
  akey_ent* e;
  int n, cap;
} amap;

static akey_ent* amap_get(const amap* m, const char* k) {
  for (int i = 0; i < m->n; ++i)
    if (!strcmp(m->e[i].key, k)) return &m->e[i];
  return NULL;
}

static void amap_add(amap* m, const char* k, int row) {   /* m.setdefault(k, set()).add(row) */
  akey_ent* e = amap_get(m, k);
  if (!e) {
    if (m->n == m->cap) {
      int nc = m->cap ? m->cap * 2 : 32;
      akey_ent* ne = arc_malloc(sizeof(akey_ent) * (size_t)nc);
      if (m->n) memcpy(ne, m->e, sizeof(akey_ent) * (size_t)m->n);
      arc_free(m->e);
      m->e = ne;
      m->cap = nc;
    }
    e = &m->e[m->n++];
    e->key = arc_strdup(k);
    e->rows = NULL;
    e->nr = e->cap = 0;
  }
  for (int i = 0; i < e->nr; ++i)
    if (e->rows[i] == row) return;
  if (e->nr == e->cap) {
    int nc = e->cap ? e->cap * 2 : 2;
    int* nr = arc_malloc(sizeof(int) * (size_t)nc);
    if (e->nr) memcpy(nr, e->rows, sizeof(int) * (size_t)e->nr);
    arc_free(e->rows);
    e->rows = nr;
    e->cap = nc;
  }
  e->rows[e->nr++] = row;
}

static void amap_free(amap* m) {
  for (int i = 0; i < m->n; ++i) {
    arc_free(m->e[i].key);
    arc_free(m->e[i].rows);
  }
  arc_free(m->e);
}

struct arc_alias {
  amap full, part;
};

static char* nospace(const char* k) {
  char* o = arc_malloc(strlen(k) + 1);
  size_t n = 0;
  for (; *k; ++k)
    if (*k != ' ') o[n++] = *k;
  o[n] = 0;
  return o;
}

static void add_both(amap* m, const char* k, int row) {   /* k and k.replace(' ', '') */
  amap_add(m, k, row);
  char* j = nospace(k);
  amap_add(m, j, row);
  arc_free(j);
}

static int split_sp(char* s, char*** out) {   /* an akey string: words separated by single spaces (modified in place) */
  int n = 0, cap = 8;
  char** w = arc_malloc(sizeof(char*) * (size_t)cap);
  for (char* p = s; *p;) {
    if (n == cap) {
      char** nw = arc_malloc(sizeof(char*) * (size_t)cap * 2);
      memcpy(nw, w, sizeof(char*) * (size_t)n);
      arc_free(w);
      w = nw;
      cap *= 2;
    }
    w[n++] = p;
    char* q = strchr(p, ' ');
    if (!q) break;
    *q = 0;
    p = q + 1;
  }
  *out = w;
  return n;
}

arc_alias* arc_alias_new(const arc_kbent* kb, int n) {
  arc_alias* a = arc_calloc(sizeof *a);
  for (int r = 0; r < n; ++r) {
    int nf = 1 + (kb[r].n_aliases > 0 ? kb[r].n_aliases : 0);
    for (int f = 0; f < nf; ++f) {
      char* k = arc_alias_akey(f == 0 ? (kb[r].name ? kb[r].name : "") : kb[r].aliases[f - 1]);
      if (k[0]) add_both(&a->full, k, r);
      arc_free(k);
    }
    /* every run of 1 .. len - 1 words of the name, unless all its words are stop / common words or shorter than 3 */
    char* t = arc_alias_akey(kb[r].name ? kb[r].name : "");
    char** w;
    int nw = t[0] ? split_sp(t, &w) : 0;
    if (!t[0]) w = NULL;
    for (int len = 1; len < nw; ++len)
      for (int i = 0; i + len <= nw; ++i) {
        int all_stop = 1;
        size_t sz = 1;
        for (int k = i; k < i + len; ++k) {
          if (!(ignorable(w[k]) || in_set(COMMON_W, NCOM, w[k]) || strlen(w[k]) < 3)) all_stop = 0;
          sz += strlen(w[k]) + 1;
        }
        if (all_stop) continue;
        char* key = arc_malloc(sz);
        key[0] = 0;
        for (int k = i; k < i + len; ++k) {
          if (k > i) strcat(key, " ");
          strcat(key, w[k]);
        }
        add_both(&a->part, key, r);
        arc_free(key);
      }
    arc_free(w);
    arc_free(t);
  }
  return a;
}

void arc_alias_free(arc_alias* a) {
  if (!a) return;
  amap_free(&a->full);
  amap_free(&a->part);
  arc_free(a);
}

int arc_alias_any(const arc_alias* a) { return a && a->full.n > 0; }

static int allowed_has(const int* allowed, int na, int r) {
  if (!allowed) return 1;
  for (int i = 0; i < na; ++i)
    if (allowed[i] == r) return 1;
  return 0;
}

int arc_alias_lookup(const arc_alias* a, const char* heard, const int* allowed, int na, const char** via) {
  *via = NULL;
  if (!a) return -1;
  char* h = arc_alias_akey(heard);
  char** ws;
  int nw = h[0] ? split_sp(h, &ws) : 0;
  if (!h[0]) ws = NULL;
  /* trim: command words and stopwords off both ends */
  int i = 0, j = nw;
  while (i < j && ignorable(ws[i])) i++;
  while (j > i && ignorable(ws[j - 1])) j--;
  int row = -1;
  if (i < j) {
    size_t sz = 1;
    for (int k = i; k < j; ++k) sz += strlen(ws[k]) + 1;
    char* sp = arc_malloc(sz);
    char* jn = arc_malloc(sz);
    sp[0] = jn[0] = 0;
    for (int k = i; k < j; ++k) {
      if (k > i) strcat(sp, " ");
      strcat(sp, ws[k]);
      strcat(jn, ws[k]);
    }
    const char* keys[2] = {sp, jn};
    int done = 0;
    for (int q = 0; q < 2 && !done; ++q) {
      const akey_ent* e = amap_get(&a->full, keys[q]);
      if (!e || !e->nr) continue;
      done = 1;
      if (e->nr == 1) {
        row = e->rows[0];
        *via = "alias";
        break;
      }
      int ok = -1, nok = 0;
      for (int k = 0; k < e->nr; ++k)
        if (allowed_has(allowed, na, e->rows[k])) {
          ok = e->rows[k];
          nok++;
        }
      if (nok == 1) {
        row = ok;
        *via = "alias";
      }
    }
    for (int q = 0; q < 2 && !done; ++q) {
      const akey_ent* e = amap_get(&a->part, keys[q]);
      if (e && e->nr == 1 && !amap_get(&a->full, keys[q])) {
        done = 1;
        if (allowed_has(allowed, na, e->rows[0])) {
          row = e->rows[0];
          *via = "partial";
        }
      }
    }
    arc_free(sp);
    arc_free(jn);
  }
  arc_free(ws);
  arc_free(h);
  return row;
}
