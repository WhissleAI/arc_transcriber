/* arc_num against arc_audio numlib.py / numvalue.py / textlayer.py (tools/dump_num.py -> tests/data/num.jsonl).
 *   test_num tests/data/num.jsonl
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arc_base.h"
#include "arc_num.h"

static int n_all = 0, n_ok = 0;

static const char** strs(const arc_json* a, int* n) {
  *n = a && a->type == ARC_JARR ? a->n : 0;
  const char** s = malloc(sizeof(char*) * (size_t)(*n + 1));
  for (int i = 0; i < *n; ++i) s[i] = a->items[i].str;
  return s;
}
static const char* ostr(const arc_json* j) { return j && j->type == ARC_JSTR ? j->str : NULL; }

/* a Python number (int or float) equal to v */
static int same_num(const arc_json* j, double v, int is_int) {
  if (!j || j->type != ARC_JNUM) return 0;
  return j->num == v && j->is_int == is_int;
}
/* r: null or [v, v2] */
static int same_val(const arc_json* r, arc_numval v) {
  if (!r || r->type == ARC_JNULL) return !v.ok;
  if (!v.ok || r->type != ARC_JARR || r->n != 2) return 0;
  if (!same_num(&r->items[0], v.v, v.v_int)) return 0;
  if (r->items[1].type == ARC_JNULL) return !v.has_v2;
  return v.has_v2 && same_num(&r->items[1], v.v2, v.v2_int);
}
static int same_str(const char* a, const char* b) { return (!a && !b) || (a && b && !strcmp(a, b)); }
static int same_toks(const arc_json* want, const char** got, int n) {
  if (!want || want->type == ARC_JNULL) return n < 0;
  if (n < 0 || want->n != n) return 0;
  for (int i = 0; i < n; ++i)
    if (strcmp(want->items[i].str, got[i])) return 0;
  return 1;
}

static void check(int good, const char* line) {
  n_all++;
  n_ok += good;
  if (!good && n_all - n_ok <= 40) printf("MISMATCH %s", line);
}

int main(int argc, char** argv) {
  FILE* f = fopen(argc > 1 ? argv[1] : "tests/data/num.jsonl", "r");
  if (!f) {
    perror("num.jsonl");
    return 2;
  }
  static char line[1 << 16];
  while (fgets(line, sizeof line, f)) {
    arc_json* j = arc_json_parse(line, strlen(line));
    if (!j) continue;
    const char* k = ostr(arc_json_get(j, "k"));
    int n, nn;
    const char** w = strs(arc_json_get(j, "w"), &n);
    if (!strcmp(k, "compose") || !strcmp(k, "finalize")) {
      int nt;
      const char** tok = strs(arc_json_get(j, "tok"), &nt);
      const char* kind = ostr(arc_json_get(j, "kind"));
      arc_numval v = !strcmp(k, "compose") ? arc_num_compose(tok, nt, kind)
                                           : arc_num_finalize(kind, tok, nt, ostr(arc_json_get(j, "unit")));
      check(same_val(arc_json_get(j, "r"), v), line);
      free(tok);
    } else if (!strcmp(k, "word")) {
      const char* s = ostr(arc_json_get(j, "w"));
      int num = arc_num_word(s);
      int good = num == (int)arc_json_num(arc_json_get(j, "num"), 0);
      if (num) good &= arc_num_connector(s) == (int)arc_json_num(arc_json_get(j, "conn"), 0);
      good &= arc_num_has_num(s) == (int)arc_json_num(arc_json_get(j, "has"), 0);
      check(good, line);
    } else if (!strcmp(k, "to")) {
      const char** t = NULL;
      int o;
      int nt = arc_num_tokens_of(w, n, &t, &o);
      check(same_toks(arc_json_get(j, "toks"), t, nt) && o == (int)arc_json_num(arc_json_get(j, "ordf"), 0), line);
      free(t);
    } else if (!strcmp(k, "wt")) {
      const char** t;
      int *wi, o;
      int nt = arc_num_word_tokens(w, n, &t, &wi, &o);
      int good = same_toks(arc_json_get(j, "toks"), t, nt) && o == (int)arc_json_num(arc_json_get(j, "ordf"), 0);
      for (int i = 0; good && i < nt; ++i) good = wi[i] == (int)arc_json_num(arc_json_at(arc_json_get(j, "wi"), i), -1);
      check(good, line);
      free(t);
      free(wi);
    } else if (!strcmp(k, "unit")) {
      const char* u = arc_num_unit_of(ostr(arc_json_get(j, "kind")), w, n, ostr(arc_json_get(j, "prev")));
      check(same_str(u, ostr(arc_json_get(j, "u"))), line);
    } else if (!strcmp(k, "tv")) {
      const char** nx = strs(arc_json_get(j, "next"), &nn);
      const char* u = NULL;
      arc_numval v = arc_text_value(ostr(arc_json_get(j, "kind")), w, n, nx, nn, ostr(arc_json_get(j, "prev")),
                                    ostr(arc_json_get(j, "unit_in")), &u);
      check(same_val(arc_json_get(j, "r"), v) && same_str(u, ostr(arc_json_get(j, "u"))), line);
      free(nx);
    }
    free(w);
    arc_json_free(j);
  }
  fclose(f);
  printf("num: %d/%d ok\n", n_ok, n_all);
  return n_ok == n_all ? 0 : 1;
}
