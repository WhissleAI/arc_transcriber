/* Name-linking text side of Arc: a C port of arc_audio rescorer.py (NameRescorer, sim, norm, ratio, metaphone),
 * namesim.py (CandTable, sim_feats, phon), textlayer.py (ExactNames, is_cap) and kbsnap.py (KB, link, add_spans),
 * with the library functions they use (rapidfuzz fuzz.ratio, jellyfish.metaphone, difflib.SequenceMatcher.ratio,
 * unicodedata NFKD). Checked against them by tests/test_ntext.c (tools/dump_ntext.py).
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
 *
 * Strings are UTF-8. Functions returning char* return malloc'd text (arc_free). */
#ifndef ARC_NTEXT_H
#define ARC_NTEXT_H

#include "arc_base.h"

/* One entry of the names list (Arc's kb): kind NULL = "other"; group may be NULL. */
typedef struct {
  const char* name;
  const char* kind;
  const char* group;
  const char* const* aliases;
  int n_aliases;
} arc_kbent;

/* ------------------------------------------------------------------ text */
char* arc_nt_norm(const char* s);    /* rescorer.norm: NFKD, ascii only, lower, 's off, [a-z0-9 ] words */
char* arc_nt_snorm(const char* s);   /* namesim.norm (= textlayer.norm): lower, [a-z0-9 ] words */
char* arc_nt_nrm(const char* s);     /* kbsnap.nrm: lower, [a-z0-9' ] words */
/* jellyfish.metaphone (input folded NFKD -> ASCII and lowered first; exact for ASCII input). */
char* arc_nt_metaphone(const char* s);
char* arc_nt_phon(const char* t);    /* namesim.phon of one token */
char* arc_nt_kbmeta(const char* t);  /* kbsnap.metaphone: phon of each word, joined by spaces */
double arc_nt_ratio(const char* a, const char* b);      /* rapidfuzz fuzz.ratio(a, b) / 100 */
double arc_nt_seqratio(const char* a, const char* b);   /* difflib.SequenceMatcher(None, a, b).ratio() */
int arc_nt_isupper0(const char* t);  /* t[:1].isupper() */
int arc_nt_is_cap(const char* t);    /* textlayer.is_cap: first char after stripping "'( is upper case */
/* rescorer.name_forms(c): rescorer-normalised name and aliases (deduplicated). Returns the count; *out malloc'd
 * array of malloc'd strings (arc_nt_free_strs). */
int arc_nt_name_forms(const arc_kbent* c, char*** out);
void arc_nt_free_strs(char** s, int n);
/* rescorer.sim(span, forms): span already rescorer-normalised. */
double arc_nt_sim(const char* span, const char* const* forms, int n_forms);

/* ------------------------------------------------------------------ namesim */
typedef struct arc_nt_candtab arc_nt_candtab;
arc_nt_candtab* arc_nt_candtab_new(const char* const* names, int R);   /* CandTable(names) */
void arc_nt_candtab_free(arc_nt_candtab* ct);
int arc_nt_candtab_rows(const arc_nt_candtab* ct);
/* sim_feats(spans, ct): out [S, R, 6] (s_full, s_tok, s_ptok, s_phon, best, shaped), float16 values as float32. */
void arc_nt_sim_feats(const arc_nt_candtab* ct, const char* const* spans, int S, float* out);

/* ------------------------------------------------------------------ rescorer */
/* A rescorer parameter set (config.json "rescorer"[domain]); a missing key is 0, thr defaults to 0.5. */
typedef struct {
  double A, A2, B, T, R, bias, thr;
} arc_nt_resc_params;
/* The engine's name policy (config.json "name_policy"[source]: "*" merged with the domain's entry). */
typedef struct {
  int has_delta, has_thr, require_heard;
  double delta, thr, fuzzy_tau;   /* fuzzy_tau 0: none */
} arc_nt_policy;
/* Engine.npol: word_source "builtin" | "external" (anything else reads "external"). */
void arc_nt_policy_cfg(const arc_json* cfg, const char* domain, const char* word_source, arc_nt_policy* out);
/* Engine._resc_params: the domain's set (else "unknown"), the policy's delta added to bias and its thr. */
void arc_nt_resc_params_cfg(const arc_json* cfg, const char* domain, const arc_nt_policy* pol, arc_nt_resc_params* out);

typedef struct arc_nt_rescorer arc_nt_rescorer;
typedef struct {
  int pred;     /* kb row, -1: None (new / not in list) */
  double p;     /* round(conf, 4) */
  int link;     /* pred >= 0 and conf >= thr */
  double sim;   /* round(sim of pred, 3); NAN when pred < 0 */
} arc_nt_resc_out;
arc_nt_rescorer* arc_nt_rescorer_new(const arc_kbent* kb, int n, const arc_nt_resc_params* p);
void arc_nt_rescorer_free(arc_nt_rescorer* r);
/* NameRescorer.score(t_s, z, allowed, span_text, past_text, feed_text): z the pointer logits [2 + n] (0 new,
 * 1 masked, 2.. kb rows), allowed kb rows; past / feed may be NULL. A link is remembered (recent). 0 or -1. */
int arc_nt_rescorer_score(arc_nt_rescorer* r, double t_s, const float* z, int nz, const int* allowed, int na,
                          const char* span_text, const char* past_text, const char* feed_text, arc_nt_resc_out* out);
/* resc.recent.append((t, row)) */
void arc_nt_rescorer_push(arc_nt_rescorer* r, double t, int row);

/* ------------------------------------------------------------------ textlayer.ExactNames */
typedef struct arc_nt_exact arc_nt_exact;
typedef struct {
  int i, j, row;   /* words [i, j) are the list's row */
} arc_nt_xspan;
arc_nt_exact* arc_nt_exact_new(const arc_kbent* kb, int n);
void arc_nt_exact_free(arc_nt_exact* e);
int arc_nt_exact_empty(const arc_nt_exact* e);   /* not bool(ExactNames) */
/* rows_in(text): sorted rows whose name / alias occurs as whole words; rows has room for n. Returns the count. */
int arc_nt_exact_rows_in(const arc_nt_exact* e, const char* text, int* rows);
/* spans(texts, used): used may be NULL; out has room for nw. Returns the count (sorted). */
int arc_nt_exact_spans(const arc_nt_exact* e, const char* const* texts, int nw, const unsigned char* used,
                       arc_nt_xspan* out);
const char* arc_nt_exact_tag_of(const arc_nt_exact* e, int row);   /* entity.mention.<kind> (person default) */

/* ------------------------------------------------------------------ kbsnap */
typedef struct arc_nt_kbsnap arc_nt_kbsnap;
arc_nt_kbsnap* arc_nt_kbsnap_new(const arc_kbent* kb, int n);
void arc_nt_kbsnap_free(arc_nt_kbsnap* k);
int arc_nt_kbsnap_empty(const arc_nt_kbsnap* k);
/* KB.score(text): the best score; *row the kb row (-1: none), *ph 1 when the phonetic keys matched. */
double arc_nt_kbsnap_score(const arc_nt_kbsnap* k, const char* text, int* row, int* ph);
/* link(rec, kb) for a record with no entity and not linked: 1 and the row and round(score, 3) when it snaps. */
int arc_nt_kbsnap_link(const arc_nt_kbsnap* k, const char* heard, int* row, double* score);

typedef struct {
  const char* text;   /* word text ("" / NULL words are skipped) */
  double start, end;
  const char* tag;    /* inline tag ("ENTITY_PERSON" ...) or NULL */
} arc_nt_word;
typedef struct {
  double start, end;   /* rounded to 3 */
  int w[3], n_words;   /* indices into the words passed */
  char* heard;         /* the words joined (owned) */
  int row;
  const char* tag;     /* entity.mention.person | team | organization | place */
  double snap_score;   /* rounded to 3 */
} arc_nt_snap_span;
/* add_spans(words, names, kb): taken [n_taken][2] start / end of the names so far. Returns the count and *out
 * (arc_nt_snap_spans_free), -1 on error. */
int arc_nt_add_spans(const arc_nt_kbsnap* k, const arc_nt_word* words, int nw, const double (*taken)[2], int n_taken,
                     arc_nt_snap_span** out);
void arc_nt_snap_spans_free(arc_nt_snap_span* s, int n);

#endif
