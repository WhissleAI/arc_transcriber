/* Number values from words and value-head tokens: a C port of arc_audio numlib.py / numvalue.py / textlayer.py
 * (text_value) and engine.py's numeric_word / connector.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
 *
 * Tokens are strings: the numerals vocabulary (arc_nwt_token: "<none>", "zero" .. "ninety", "hundred" .. "billion",
 * "point", "and", "a", "half", "quarter", "dozen", "SEP", "OH", "NEG", "UNITW"). Kinds and units are the strings
 * numvalue.py uses ("score", "duration", ...; "minute", "USD", ...); NULL stands for Python's None. */
#ifndef ARC_NUM_H
#define ARC_NUM_H

/* A value as Python returns it: ok = 0 for None; v_int / v2_int: the Python value is an int (print without ".0"). */
typedef struct {
  int ok;
  double v, v2;
  int has_v2;
  int v_int, v2_int;
} arc_numval;

/* numlib.compose(tok, kind). */
arc_numval arc_num_compose(const char* const* tok, int n, const char* kind);
/* numvalue.finalize(kind, tok, unit, words) ('<none>' tokens are dropped, as Python does). */
arc_numval arc_num_finalize(const char* kind, const char* const* tok, int n, const char* unit);
/* textlayer.text_value(kind, words, next_words, prev_word, unit): the value (ok = 0 for None) and *unit_out (a static
 * string, unit_in itself, or NULL), exactly Python's (v, unit) pair. unit_in NULL: Python's unit=None. */
arc_numval arc_text_value(const char* kind, const char* const* words, int n, const char* const* next, int n_next,
                          const char* prev_word, const char* unit_in, const char** unit_out);
/* engine.numeric_word: numlib.tokens_of([w]) finds a number. */
int arc_num_word(const char* w);
/* engine.connector: every token of the word is 'and', 'SEP', 'a', 'point' or 'NEG' (1 for a non-number word, where
 * Python would raise). */
int arc_num_connector(const char* w);
/* numlib.NWT[k] (== config.json numerals.nwt_vocab), k in [0, 43); NULL outside. */
const char* arc_nwt_token(int k);
#define ARC_NWT_N 43

/* numvalue.unit_of(kind, words, prev_word): a static unit string or NULL. */
const char* arc_num_unit_of(const char* kind, const char* const* words, int n, const char* prev_word);
/* numlib.tokens_of(words): the token count (*toks malloc'd, free() it; static strings), or -1 for None. */
int arc_num_tokens_of(const char* const* words, int n, const char*** toks, int* ordf);
/* numvalue.word_tokens(words): count of (word index, token) pairs; *toks and *wi malloc'd (free() them). */
int arc_num_word_tokens(const char* const* words, int n, const char*** toks, int** wi, int* ordf);
/* textlayer.has_num(w): a numeral token in the word. */
int arc_num_has_num(const char* w);

#endif
