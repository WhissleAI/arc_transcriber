/* arc_post (quantities, sound-alike keys, guard, disfluency, sound-alike names, end of turn) against arc_audio
 * (tools/dump_post.py -> tests/data/post.jsonl). Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arc_base.h"
#include "arc_post.h"

static int n = 0, ok = 0;
static void check(int good, const char* what, const char* in) {
  n++;
  ok += good;
  if (!good) printf("FAIL %s: %s\n", what, in);
}

static int words_of(const arc_json* a, const char** w) {
  int k = a ? a->n : 0;
  for (int i = 0; i < k; ++i) w[i] = arc_json_str(&a->items[i], "");
  return k;
}

int main(int argc, char** argv) {
  FILE* f = fopen(argc > 1 ? argv[1] : "tests/data/post.jsonl", "r");
  if (!f) {
    perror("post.jsonl");
    return 1;
  }
  static char line[1 << 20];
  while (fgets(line, sizeof line, f)) {
    arc_json* j = arc_json_parse(line, strlen(line));
    const char* k = arc_json_str(arc_json_get(j, "k"), "");
    const char* w[256];
    int nw = words_of(arc_json_get(j, "w"), w);
    char in[2048] = "";
    for (int i = 0; i < nw; ++i) snprintf(in + strlen(in), sizeof in - strlen(in), "%s%s", i ? " " : "", w[i]);
    if (!strcmp(k, "qty")) {
      const char* tags[256] = {0};
      const arc_json* tj = arc_json_get(j, "tags");
      for (int i = 0; tj && i < tj->n; ++i) tags[i] = arc_json_str(&tj->items[i], NULL);
      arc_qty* q;
      int nq = arc_post_quantities(w, tj ? tags : NULL, nw, &q);
      const arc_json* want = arc_json_get(j, "q");
      int good = nq == want->n;
      for (int i = 0; good && i < nq; ++i) {
        const arc_json* x = &want->items[i];
        const char* unit = arc_json_str(arc_json_get(x, "unit"), NULL);
        good = !strcmp(q[i].type, arc_json_str(arc_json_get(x, "type"), "")) &&
               fabs(q[i].value - arc_json_num(arc_json_get(x, "value"), 0)) < 1e-9 &&
               ((!unit && !q[i].unit) || (unit && q[i].unit && !strcmp(unit, q[i].unit))) &&
               q[i].relative == (int)arc_json_num(arc_json_get(x, "relative"), 0) &&
               q[i].word_begin == (int)arc_json_num(arc_json_get(x, "word_begin"), 0) &&
               q[i].word_end == (int)arc_json_num(arc_json_get(x, "word_end"), 0) &&
               !strcmp(q[i].written, arc_json_str(arc_json_get(x, "written"), "")) &&
               !strcmp(q[i].text, arc_json_str(arc_json_get(x, "text"), ""));
        if (!good) printf("  got %s %g %s %d [%s] [%s]\n", q[i].type, q[i].value, q[i].unit ? q[i].unit : "-", q[i].relative, q[i].written, q[i].text);
      }
      if (nq != want->n) printf("  got %d quantities, want %d\n", nq, want->n);
      check(good, "quantities", in);
      arc_post_qty_free(q, nq);
    } else if (!strcmp(k, "phon")) {
      const char *a = arc_json_str(arc_json_get(j, "a"), ""), *b = arc_json_str(arc_json_get(j, "b"), "");
      char *ka = arc_phon_key(a), *kb = arc_phon_key(b);
      double s = arc_phon_similarity(a, b);
      int good = !strcmp(ka, arc_json_str(arc_json_get(j, "ka"), "")) && !strcmp(kb, arc_json_str(arc_json_get(j, "kb"), "")) &&
                 fabs(s - arc_json_num(arc_json_get(j, "s"), -1)) < 1e-9;
      if (!good) printf("  got %s %s %.6f\n", ka, kb, s);
      check(good, "phonetic", a);
      arc_free(ka);
      arc_free(kb);
    } else if (!strcmp(k, "disfl")) {
      const char* t[256];
      arc_post_disfluency(w, nw, t);
      const arc_json* want = arc_json_get(j, "t");
      int good = 1;
      for (int i = 0; i < nw; ++i) {
        const char* x = arc_json_str(&want->items[i], NULL);
        good &= (!x && !t[i]) || (x && t[i] && !strcmp(x, t[i]));
      }
      check(good, "disfluency", in);
    } else if (!strcmp(k, "eot")) {
      double p = arc_post_eot(w, nw, arc_json_num(arc_json_get(j, "pause"), 0));
      check(fabs(p - arc_json_num(arc_json_get(j, "p"), -1)) < 1e-9, "eot", in);
    } else if (!strcmp(k, "almost")) {
      const arc_json* nm = arc_json_get(j, "names");
      arc_form forms[64];
      int nf = 0;
      for (int i = 0; i < nm->n; ++i) {
        const arc_json* e = &nm->items[i];
        forms[nf++] = (arc_form){arc_json_str(arc_json_get(e, "name"), ""), i};
        const arc_json* al = arc_json_get(e, "aliases");
        for (int a = 0; al && a < al->n; ++a) forms[nf++] = (arc_form){arc_json_str(&al->items[a], ""), i};
      }
      arc_almost_cfg c;
      arc_almost_defaults(&c);
      arc_almost_hit* h;
      int nh = arc_post_almost(w, NULL, NULL, nw, NULL, forms, nf, &c, &h);
      const arc_json* want = arc_json_get(j, "hits");
      int good = nh == want->n;
      for (int i = 0; good && i < nh; ++i) {
        const arc_json* x = &want->items[i];
        good = h[i].word_begin == (int)arc_json_num(arc_json_at(x, 0), -1) && h[i].word_end == (int)arc_json_num(arc_json_at(x, 1), -1) &&
               h[i].kb == (int)arc_json_num(arc_json_at(x, 2), -1) && fabs(h[i].p - arc_json_num(arc_json_at(x, 3), -1)) < 1e-9;
      }
      if (!good)
        for (int i = 0; i < nh; ++i) printf("  got %d %d %d %.4f\n", h[i].word_begin, h[i].word_end, h[i].kb, h[i].p);
      check(good, "almost", in);
      arc_free(h);
    } else if (!strcmp(k, "guard")) {
      const arc_json* sg = arc_json_get(j, "sig");
      long ny = (long)arc_json_num(arc_json_get(sg, "n"), 0);
      double base = arc_json_num(arc_json_get(sg, "base"), 0), loud = arc_json_num(arc_json_get(sg, "loud"), 1);
      long a = (long)arc_json_num(arc_json_get(sg, "a"), 0), b = (long)arc_json_num(arc_json_get(sg, "b"), 0);
      float* y = arc_malloc(sizeof(float) * (size_t)ny);
      for (long i = 0; i < ny; ++i) {
        double v = base * sin(i * 0.05) * (1 + 0.3 * sin(i * 0.0007));
        if (i >= a && i < b) v *= loud;
        y[i] = (float)v;
      }
      const arc_json* ws = arc_json_get(j, "words");
      const char* tx[16];
      double st[16];
      float cf[16];
      int nws = ws->n;
      for (int i = 0; i < nws; ++i) {
        tx[i] = arc_json_str(arc_json_get(&ws->items[i], "text"), "");
        st[i] = arc_json_num(arc_json_get(&ws->items[i], "start"), 0);
        cf[i] = (float)arc_json_num(arc_json_get(&ws->items[i], "conf"), -1);
      }
      arc_guard_cfg c;
      arc_guard_defaults(&c);
      int keep[16];
      const char* why[16];
      arc_post_guard(tx, st, cf, nws, y, ny, 0.0, &c, keep, why);
      const arc_json* wy = arc_json_get(j, "why");
      int good = 1, d = 0;
      for (int i = 0; i < nws; ++i)
        if (!keep[i]) good &= d < wy->n && !strcmp(why[i], arc_json_str(&wy->items[d++], ""));
      good &= d == wy->n;
      check(good, "guard", "signal");
      arc_free(y);
    }
    arc_json_free(j);
  }
  fclose(f);
  printf("test_post: %d / %d\n", ok, n);
  return ok == n ? 0 : 1;
}
