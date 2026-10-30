/* arc_tidy against tidy.py (tools/dump_tidy.py).
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "arc_base.h"
#include "arc_tidy.h"

static int n = 0, ok = 0;
static void check(int good, const char* what, const char* in, const char* got, const char* want) {
  n++;
  ok += good;
  if (!good) printf("%s [%s]: got [%s] want [%s]\n", what, in, got, want);
}

static void load_name(const arc_json* j, arc_tname* r) {
  memset(r, 0, sizeof *r);
  r->heard = arc_strdup(arc_json_str(arc_json_get(j, "heard"), ""));
  r->entity = arc_json_str(arc_json_get(j, "entity"), NULL);
  r->state = arc_json_str(arc_json_get(j, "state"), NULL);
  r->via = arc_json_str(arc_json_get(j, "via"), NULL);
  r->p = (float)arc_json_num(arc_json_get(j, "p"), 0);
  r->start = (float)arc_json_num(arc_json_get(j, "start"), 0);
  r->end = (float)arc_json_num(arc_json_get(j, "end"), 0);
  const arc_json* w = arc_json_get(j, "words");
  r->n_words = w ? w->n : 0;
  r->words = arc_calloc(sizeof(arc_tword) * (size_t)(r->n_words + 1));
  for (int i = 0; i < r->n_words; ++i) {
    r->words[i].w = arc_json_str(arc_json_get(&w->items[i], "w"), "");
    r->words[i].start = (float)arc_json_num(arc_json_get(&w->items[i], "start"), 0);
    r->words[i].end = (float)arc_json_num(arc_json_get(&w->items[i], "end"), 0);
  }
}

int main(int argc, char** argv) {
  FILE* f = fopen(argv[1], "r");
  static char line[1 << 16];
  arc_json* keep[64];
  int nkeep = 0;
  while (fgets(line, sizeof line, f)) {
    arc_json* j = arc_json_parse(line, strlen(line));
    keep[nkeep++] = j;
    const char* k = arc_json_str(arc_json_get(j, "k"), "");
    const char* in = arc_json_str(arc_json_get(j, "i"), "");
    if (!strcmp(k, "word")) {
      char buf[256];
      arc_clean_word(in, buf);
      check(!strcmp(buf, arc_json_str(arc_json_get(j, "clean"), "")), "clean_word", in, buf, arc_json_str(arc_json_get(j, "clean"), ""));
      int num = arc_numeric_word(in), want = (int)arc_json_num(arc_json_get(j, "num"), 0);
      check(num == want, "numeric", in, num ? "1" : "0", want ? "1" : "0");
    } else if (!strcmp(k, "phrase")) {
      char* c = arc_clean_text(in);
      check(!strcmp(c, arc_json_str(arc_json_get(j, "clean"), "")), "clean_text", in, c, arc_json_str(arc_json_get(j, "clean"), ""));
      char* t = arc_tidy_number(in);
      check(!strcmp(t, arc_json_str(arc_json_get(j, "num"), "")), "tidy_number", in, t, arc_json_str(arc_json_get(j, "num"), ""));
      int kp = arc_keep_number(0, in), want = (int)arc_json_num(arc_json_get(j, "keep"), 0);
      check(kp == want, "keep_number", in, kp ? "1" : "0", want ? "1" : "0");
      arc_free(c);
      arc_free(t);
    } else if (!strcmp(k, "name")) {
      arc_tname r;
      load_name(arc_json_get(j, "i"), &r);
      arc_tword* base = r.words;
      arc_tidy_name(&r);
      int good = !strcmp(r.heard, arc_json_str(arc_json_get(j, "heard"), "")) && fabs(r.start - arc_json_num(arc_json_get(j, "start"), 0)) < 1e-6 &&
                 fabs(r.end - arc_json_num(arc_json_get(j, "end"), 0)) < 1e-6;
      check(good, "tidy_name", arc_json_str(arc_json_get(arc_json_get(j, "i"), "heard"), ""), r.heard, arc_json_str(arc_json_get(j, "heard"), ""));
      arc_free(r.heard);
      arc_free(base);
    } else if (!strcmp(k, "dedupe")) {
      const arc_json* I = arc_json_get(j, "i");
      arc_tname r[16];
      for (int i = 0; i < I->n; ++i) load_name(&I->items[i], &r[i]);
      int nk = arc_dedupe_names(r, I->n);
      const arc_json* K = arc_json_get(j, "kept");
      int good = nk == K->n;
      for (int i = 0; good && i < nk; ++i) good = !strcmp(r[i].heard, arc_json_str(arc_json_at(arc_json_at(K, i), 0), ""));
      char got[256] = "";
      for (int i = 0; i < nk; ++i) { strcat(got, r[i].heard); strcat(got, "|"); }
      check(good, "dedupe_names", "", got, "(see python)");
    }
  }
  printf("%d/%d tidy cases identical to tidy.py\n", ok, n);
  return ok == n ? 0 : 1;
}
