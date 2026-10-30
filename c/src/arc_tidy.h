/* Output tidying (arc_audio/tidy.py): word edges, numeric words, trimming a number's or a name's words to the mention,
 * and de-duplicating names. Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#ifndef ARC_TIDY_H
#define ARC_TIDY_H

/* clean_word: punctuation off a word's edges (leading: keeps $ £ € % # @ +; trailing: keeps % and '). Writes to out
 * (at least strlen(w) + 1). */
void arc_clean_word(const char* w, char* out);
/* str.split(): returns the count; words malloc'd (arc_free each and the array). */
int arc_split_words_tidy(const char* t, char*** out);
/* clean_text: clean_word on every word, empty ones dropped, joined by single spaces. Returns malloc'd text. */
char* arc_clean_text(const char* t);
/* _numeric: a number word (digits, a number name, an ordinal word, "twenty-six"). */
int arc_numeric_word(const char* w);
/* tidy_number on a record's heard text: from the first to the last numeric word, plus unit words after and a
 * currency / sign word before; cleaned. Returns malloc'd text. */
char* arc_tidy_number(const char* heard);
/* keep_number: a value, or a numeric word in what was heard. */
int arc_keep_number(int has_value, const char* heard);

typedef struct {
  const char* w;
  double start, end;
} arc_tword;

typedef struct {
  char* heard;           /* owned */
  const char* entity;    /* the list's name, or NULL */
  const char* state;     /* "linked", ... */
  const char* via;       /* NULL, "exact", "snap", "fallback" */
  float p;               /* for the ranking: p, else span_p, else 0 (r.get('p') or r.get('span_p') or 0.0) */
  double start, end;
  arc_tword* words;      /* borrowed; may be NULL (tidy_name moves it to the kept words) */
  int n_words;
  int id;                /* the caller's */
} arc_tname;

/* tidy_name: drops edge words that are not part of the mention (lower-case next to capitalised ones, or not in the
 * linked entity), then cleans heard. */
void arc_tidy_name(arc_tname* r);
int arc_same_mention(const arc_tname* a, const arc_tname* b);
/* dedupe_names: the best of each set of the same mention (linked first, then via, then p), sorted by time. Returns the
 * count kept (records moved to the front of r; the rest are freed). */
int arc_dedupe_names(arc_tname* r, int n);

#endif
