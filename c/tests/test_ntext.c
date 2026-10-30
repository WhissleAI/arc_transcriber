/* arc_ntext against arc_audio rescorer.py / namesim.py / textlayer.py / kbsnap.py (tools/dump_ntext.py).
 *   build/test_ntext tests/data/ntext.jsonl
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arc_base.h"
#include "arc_ntext.h"

#define NK 12
static const char* KINDS[NK] = {"norm", "meta", "phon", "kbmeta", "ratio", "seqratio", "sim", "feats", "policy",
                                "resc", "exact", "snap"};
static int n_k[NK], ok_k[NK], shown = 0;

static void check(int k, int good, const char* fmt, ...) {
  n_k[k]++;
  ok_k[k] += good;
  if (!good && shown++ < 60) {
    va_list ap;
    va_start(ap, fmt);
    printf("%s: ", KINDS[k]);
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
  }
}

static const char* S(const arc_json* j) { return arc_json_str(j, ""); }
static int B(const arc_json* j) { return j && j->type == ARC_JBOOL && j->num != 0; }
static int streq(const char* a, const char* b) { return !strcmp(a ? a : "", b ? b : ""); }

typedef struct {
  arc_kbent* e;
  int n;
} kbv;

static kbv load_kb(const arc_json* a) {
  kbv k = {arc_calloc(sizeof(arc_kbent) * (size_t)(a->n + 1)), a->n};
  for (int i = 0; i < a->n; ++i) {
    const arc_json* c = &a->items[i];
    k.e[i].name = arc_json_str(arc_json_get(c, "name"), "");
    k.e[i].kind = arc_json_str(arc_json_get(c, "kind"), NULL);
    k.e[i].group = arc_json_str(arc_json_get(c, "group"), NULL);
    const arc_json* al = arc_json_get(c, "aliases");
    if (al && al->n) {
      const char** v = arc_calloc(sizeof(char*) * (size_t)al->n);
      for (int q = 0; q < al->n; ++q) v[q] = al->items[q].str;
      k.e[i].aliases = v;
      k.e[i].n_aliases = al->n;
    }
  }
  return k;
}
static void free_kb(kbv k) {
  for (int i = 0; i < k.n; ++i) arc_free((void*)k.e[i].aliases);
  arc_free(k.e);
}

static void params_of(const arc_json* o, arc_nt_resc_params* p) {
  p->A = arc_json_num(arc_json_get(o, "A"), 0);
  p->A2 = arc_json_num(arc_json_get(o, "A2"), 0);
  p->B = arc_json_num(arc_json_get(o, "B"), 0);
  p->T = arc_json_num(arc_json_get(o, "T"), 0);
  p->R = arc_json_num(arc_json_get(o, "R"), 0);
  p->bias = arc_json_num(arc_json_get(o, "bias"), 0);
  p->thr = arc_json_num(arc_json_get(o, "thr"), 0.5);
}

static void t_norm(const arc_json* j) {
  const char* in = S(arc_json_get(j, "i"));
  char *a = arc_nt_norm(in), *b = arc_nt_snorm(in), *c = arc_nt_nrm(in);
  check(0, streq(a, S(arc_json_get(j, "norm"))), "norm [%s] -> [%s] want [%s]", in, a, S(arc_json_get(j, "norm")));
  check(0, streq(b, S(arc_json_get(j, "snorm"))), "snorm [%s] -> [%s] want [%s]", in, b, S(arc_json_get(j, "snorm")));
  check(0, streq(c, S(arc_json_get(j, "nrm"))), "nrm [%s] -> [%s] want [%s]", in, c, S(arc_json_get(j, "nrm")));
  check(0, arc_nt_is_cap(in) == B(arc_json_get(j, "cap")), "is_cap [%s]", in);
  check(0, arc_nt_isupper0(in) == B(arc_json_get(j, "up")), "isupper [%s]", in);
  arc_free(a);
  arc_free(b);
  arc_free(c);
}

static void t_meta(const arc_json* j) {
  const char* in = S(arc_json_get(j, "i"));
  char* m = arc_nt_metaphone(in);
  check(1, streq(m, S(arc_json_get(j, "o"))), "[%s] -> [%s] want [%s]", in, m, S(arc_json_get(j, "o")));
  char* lw = arc_strdup(in);
  for (char* p = lw; *p; ++p) *p = (char)tolower((unsigned char)*p);
  char* ph = arc_nt_phon(lw);
  check(2, streq(ph, S(arc_json_get(j, "ph"))), "[%s] -> [%s] want [%s]", lw, ph, S(arc_json_get(j, "ph")));
  char* nn = arc_nt_nrm(in);
  char* km = arc_nt_kbmeta(nn);
  check(3, streq(km, S(arc_json_get(j, "km"))), "[%s] -> [%s] want [%s]", in, km, S(arc_json_get(j, "km")));
  arc_free(m);
  arc_free(lw);
  arc_free(ph);
  arc_free(nn);
  arc_free(km);
}

static void t_ratio(const arc_json* j) {
  const char *a = S(arc_json_get(j, "a")), *b = S(arc_json_get(j, "b"));
  double r = arc_nt_ratio(a, b), q = arc_nt_seqratio(a, b);
  double wr = arc_json_num(arc_json_get(j, "o"), -1), wq = arc_json_num(arc_json_get(j, "sq"), -1);
  check(4, r == wr, "[%.40s] [%.40s] %.17g want %.17g", a, b, r, wr);
  check(5, q == wq, "[%.40s] [%.40s] %.17g want %.17g", a, b, q, wq);
}

static void t_sim(const arc_json* j) {
  const arc_json* f = arc_json_get(j, "forms");
  const char** v = arc_calloc(sizeof(char*) * (size_t)(f->n + 1));
  for (int i = 0; i < f->n; ++i) v[i] = f->items[i].str;
  double s = arc_nt_sim(S(arc_json_get(j, "span")), v, f->n), w = arc_json_num(arc_json_get(j, "o"), -1);
  check(6, s == w, "[%s] vs [%s..] %.17g want %.17g", S(arc_json_get(j, "span")), f->n ? v[0] : "", s, w);
  arc_free(v);
}

static void t_feats(const arc_json* j) {
  const arc_json *nm = arc_json_get(j, "names"), *sp = arc_json_get(j, "spans"), *o = arc_json_get(j, "o");
  const char** a = arc_calloc(sizeof(char*) * (size_t)(nm->n + 1));
  const char** b = arc_calloc(sizeof(char*) * (size_t)(sp->n + 1));
  for (int i = 0; i < nm->n; ++i) a[i] = nm->items[i].str;
  for (int i = 0; i < sp->n; ++i) b[i] = sp->items[i].str;
  arc_nt_candtab* ct = arc_nt_candtab_new(a, nm->n);
  float* out = arc_malloc(sizeof(float) * (size_t)(sp->n * nm->n * 6 + 1));
  arc_nt_sim_feats(ct, b, sp->n, out);
  int bad = 0;
  for (int i = 0; i < o->n; ++i)
    if (out[i] != (float)o->items[i].num) {
      if (!bad) printf("feats: [%d] span %s row %d f%d: %g want %g\n", i, b[i / 6 / nm->n], i / 6 % nm->n, i % 6,
                       out[i], o->items[i].num);
      bad++;
    }
  check(7, !bad && o->n == sp->n * nm->n * 6, "%d values differ", bad);
  arc_nt_candtab_free(ct);
  arc_free(out);
  arc_free(a);
  arc_free(b);
}

static void t_policy(const arc_json* j) {
  const arc_json *cfg = arc_json_get(j, "cfg"), *cs = arc_json_get(j, "cases");
  for (int i = 0; i < cs->n; ++i) {
    const arc_json* c = &cs->items[i];
    const char *d = S(arc_json_at(c, 0)), *ws = S(arc_json_at(c, 1));
    arc_nt_policy pol;
    arc_nt_resc_params p, w;
    arc_nt_policy_cfg(cfg, d, ws, &pol);
    arc_nt_resc_params_cfg(cfg, d, &pol, &p);
    params_of(arc_json_at(c, 2), &w);
    const arc_json* np = arc_json_at(c, 3);
    int good = p.A == w.A && p.A2 == w.A2 && p.B == w.B && p.T == w.T && p.R == w.R && p.bias == w.bias && p.thr == w.thr;
    good &= pol.require_heard == B(arc_json_get(np, "require_heard"));
    good &= pol.fuzzy_tau == arc_json_num(arc_json_get(np, "fuzzy_tau"), 0);
    const arc_json* dl = arc_json_get(np, "delta");
    good &= pol.has_delta == (dl && dl->type == ARC_JNUM) && (!pol.has_delta || pol.delta == dl->num);
    check(8, good, "%s / %s: bias %g thr %g want %g %g", d, ws, p.bias, p.thr, w.bias, w.thr);
  }
}

static void t_resc(const arc_json* j) {
  kbv kb = load_kb(arc_json_get(j, "kb"));
  arc_nt_resc_params p;
  params_of(arc_json_get(j, "params"), &p);
  arc_nt_rescorer* r = arc_nt_rescorer_new(kb.e, kb.n, &p);
  const arc_json* st = arc_json_get(j, "steps");
  for (int i = 0; i < st->n; ++i) {
    const arc_json* s = &st->items[i];
    const arc_json* pu = arc_json_get(s, "push");
    if (pu) {
      arc_nt_rescorer_push(r, arc_json_at(pu, 0)->num, (int)arc_json_at(pu, 1)->num);
      continue;
    }
    const arc_json *z = arc_json_get(s, "z"), *al = arc_json_get(s, "allowed"), *o = arc_json_get(s, "o");
    float* zz = arc_malloc(sizeof(float) * (size_t)(z->n + 1));
    int* aa = arc_malloc(sizeof(int) * (size_t)(al->n + 1));
    for (int q = 0; q < z->n; ++q) zz[q] = (float)z->items[q].num;
    for (int q = 0; q < al->n; ++q) aa[q] = (int)al->items[q].num;
    const arc_json* fd = arc_json_get(s, "feed");
    arc_nt_resc_out out;
    arc_nt_rescorer_score(r, arc_json_num(arc_json_get(s, "t"), 0), zz, z->n, aa, al->n, S(arc_json_get(s, "span")),
                          S(arc_json_get(s, "past")), fd && fd->type == ARC_JSTR ? fd->str : NULL, &out);
    const arc_json* wp = arc_json_get(o, "pred");
    int wpred = wp && wp->type == ARC_JNUM ? (int)wp->num : -1;
    const arc_json* wsim = arc_json_get(o, "sim");
    int good = out.pred == wpred && out.p == arc_json_num(arc_json_get(o, "p"), -1) && out.link == B(arc_json_get(o, "link")) &&
               (wsim && wsim->type == ARC_JNUM ? out.sim == wsim->num : isnan(out.sim));
    check(9, good, "step %d [%s]: pred %d p %.4f link %d sim %.3f want %d %.4f %d %.3f", i, S(arc_json_get(s, "span")),
          out.pred, out.p, out.link, out.sim, wpred, arc_json_num(arc_json_get(o, "p"), -1), B(arc_json_get(o, "link")),
          arc_json_num(wsim, NAN));
    arc_free(zz);
    arc_free(aa);
  }
  arc_nt_rescorer_free(r);
  free_kb(kb);
}

static void t_exact(const arc_json* j) {
  kbv kb = load_kb(arc_json_get(j, "kb"));
  arc_nt_exact* e = arc_nt_exact_new(kb.e, kb.n);
  check(10, arc_nt_exact_empty(e) == B(arc_json_get(j, "empty")), "empty");
  const arc_json* ri = arc_json_get(j, "rows_in");
  int* rows = arc_malloc(sizeof(int) * (size_t)(kb.n + 1));
  for (int i = 0; i < ri->n; ++i) {
    const arc_json *c = &ri->items[i], *w = arc_json_at(c, 1);
    int n = arc_nt_exact_rows_in(e, S(arc_json_at(c, 0)), rows), good = n == w->n;
    for (int q = 0; q < n && good; ++q) good = rows[q] == (int)w->items[q].num;
    check(10, good, "rows_in [%s]: %d rows want %d", S(arc_json_at(c, 0)), n, w->n);
  }
  const arc_json* sp = arc_json_get(j, "spans");
  for (int i = 0; i < sp->n; ++i) {
    const arc_json *c = &sp->items[i], *ws = arc_json_at(c, 0), *us = arc_json_at(c, 1), *w = arc_json_at(c, 2);
    const char** t = arc_calloc(sizeof(char*) * (size_t)(ws->n + 1));
    unsigned char* u = arc_calloc((size_t)ws->n + 1);
    for (int q = 0; q < ws->n; ++q) {
      t[q] = ws->items[q].str;
      if (us->type == ARC_JARR) u[q] = (unsigned char)B(&us->items[q]);
    }
    arc_nt_xspan* o = arc_malloc(sizeof(arc_nt_xspan) * (size_t)(ws->n + 1));
    int n = arc_nt_exact_spans(e, t, ws->n, us->type == ARC_JARR ? u : NULL, o), good = n == w->n;
    for (int q = 0; q < n && good; ++q)
      good = o[q].i == (int)arc_json_at(&w->items[q], 0)->num && o[q].j == (int)arc_json_at(&w->items[q], 1)->num &&
             o[q].row == (int)arc_json_at(&w->items[q], 2)->num;
    check(10, good, "spans [%s ...]: %d want %d", ws->n ? t[0] : "", n, w->n);
    arc_free(t);
    arc_free(u);
    arc_free(o);
  }
  const arc_json* tg = arc_json_get(j, "tag_of");
  for (int i = 0; i < tg->n; ++i)
    check(10, streq(arc_nt_exact_tag_of(e, i), tg->items[i].str), "tag_of %d: %s", i, arc_nt_exact_tag_of(e, i));
  arc_free(rows);
  arc_nt_exact_free(e);
  free_kb(kb);
}

static char* name_of(const kbv* kb, int row) {   /* ' '.join(name.split()) */
  char* s = arc_strdup(row >= 0 ? kb->e[row].name : "");
  char* o = s;
  int sp = 1;
  char* w = s;
  for (char* p = s; *p; ++p) {
    if (isspace((unsigned char)*p)) {
      if (!sp) *w++ = ' ';
      sp = 1;
    } else {
      *w++ = *p;
      sp = 0;
    }
  }
  if (w > s && w[-1] == ' ') w--;
  *w = 0;
  return o;
}

static void t_snap(const arc_json* j) {
  kbv kb = load_kb(arc_json_get(j, "kb"));
  arc_nt_kbsnap* k = arc_nt_kbsnap_new(kb.e, kb.n);
  const arc_json* sc = arc_json_get(j, "score");
  for (int i = 0; i < sc->n; ++i) {
    const arc_json* c = &sc->items[i];
    int row, ph;
    double s = arc_nt_kbsnap_score(k, S(arc_json_at(c, 0)), &row, &ph);
    char* nm = name_of(&kb, row);
    const arc_json* wn = arc_json_at(c, 2);
    int good = s == arc_json_at(c, 1)->num && ph == B(arc_json_at(c, 4)) &&
               (wn->type == ARC_JSTR ? row >= 0 && streq(nm, wn->str) : row < 0);
    check(11, good, "score [%s]: %.17g %s %d want %.17g %s", S(arc_json_at(c, 0)), s, nm, ph, arc_json_at(c, 1)->num,
          arc_json_str(wn, "None"));
    arc_free(nm);
  }
  const arc_json* ln = arc_json_get(j, "link");
  for (int i = 0; i < ln->n; ++i) {
    const arc_json *c = &ln->items[i], *rec = arc_json_at(c, 0);
    int skip = arc_json_str(arc_json_get(rec, "entity"), "")[0] || streq(arc_json_str(arc_json_get(rec, "state"), ""), "linked");
    int row = -1;
    double s = 0;
    int got = skip ? 0 : arc_nt_kbsnap_link(k, S(arc_json_get(rec, "heard")), &row, &s);
    char* nm = name_of(&kb, row);
    int good = got == B(arc_json_at(c, 1)) && (!got || (streq(nm, S(arc_json_at(c, 2))) && s == arc_json_at(c, 3)->num));
    check(11, good, "link [%s]: %d %s %g", S(arc_json_get(rec, "heard")), got, nm, s);
    arc_free(nm);
  }
  const arc_json* ad = arc_json_get(j, "add");
  for (int i = 0; i < ad->n; ++i) {
    const arc_json *c = &ad->items[i], *ws = arc_json_get(c, "words"), *nm = arc_json_get(c, "names"),
                   *w = arc_json_get(c, "out");
    arc_nt_word* words = arc_calloc(sizeof(arc_nt_word) * (size_t)(ws->n + 1));
    for (int q = 0; q < ws->n; ++q) {
      words[q].text = S(arc_json_get(&ws->items[q], "text"));
      words[q].start = arc_json_num(arc_json_get(&ws->items[q], "start"), 0);
      words[q].end = arc_json_num(arc_json_get(&ws->items[q], "end"), 0);
      words[q].tag = arc_json_str(arc_json_get(&ws->items[q], "tag"), NULL);
    }
    double(*tk)[2] = arc_malloc(sizeof(double[2]) * (size_t)(nm->n + 1));
    for (int q = 0; q < nm->n; ++q) {
      tk[q][0] = arc_json_at(&nm->items[q], 0)->num;
      tk[q][1] = arc_json_at(&nm->items[q], 1)->num;
    }
    arc_nt_snap_span* o;
    int n = arc_nt_add_spans(k, words, ws->n, (const double(*)[2])tk, nm->n, &o), good = n == w->n;
    for (int q = 0; q < n && good; ++q) {
      const arc_json* r = &w->items[q];
      char* en = name_of(&kb, o[q].row);
      good = o[q].start == arc_json_num(arc_json_get(r, "start"), -1) && o[q].end == arc_json_num(arc_json_get(r, "end"), -1) &&
             streq(o[q].heard, S(arc_json_get(r, "heard"))) && streq(en, S(arc_json_get(r, "entity"))) &&
             streq(o[q].tag, S(arc_json_get(r, "tag"))) && o[q].snap_score == arc_json_num(arc_json_get(r, "snap_score"), -1) &&
             o[q].n_words == arc_json_get(r, "words")->n;
      arc_free(en);
    }
    check(11, good, "add_spans case %d: %d spans want %d", i, n, w->n);
    arc_nt_snap_spans_free(o, n);
    arc_free(words);
    arc_free(tk);
  }
  arc_nt_kbsnap_free(k);
  free_kb(kb);
}

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "tests/data/ntext.jsonl";
  char* text;
  size_t len;
  if (arc_read_file(path, &text, &len)) {
    fprintf(stderr, "%s\n", arc_geterr());
    return 2;
  }
  for (char *line = text, *nl; line && *line; line = nl ? nl + 1 : NULL) {
    nl = strchr(line, '\n');
    size_t L = nl ? (size_t)(nl - line) : strlen(line);
    if (!L) continue;
    arc_json* j = arc_json_parse(line, L);
    if (!j) {
      fprintf(stderr, "bad line: %s\n", arc_geterr());
      return 2;
    }
    const char* k = S(arc_json_get(j, "k"));
    if (!strcmp(k, "norm")) t_norm(j);
    else if (!strcmp(k, "meta")) t_meta(j);
    else if (!strcmp(k, "ratio")) t_ratio(j);
    else if (!strcmp(k, "sim")) t_sim(j);
    else if (!strcmp(k, "feats")) t_feats(j);
    else if (!strcmp(k, "policy")) t_policy(j);
    else if (!strcmp(k, "resc")) t_resc(j);
    else if (!strcmp(k, "exact")) t_exact(j);
    else if (!strcmp(k, "snap")) t_snap(j);
    arc_json_free(j);
  }
  arc_free(text);
  int n = 0, ok = 0;
  for (int k = 0; k < NK; ++k) {
    printf("  %-8s %d/%d\n", KINDS[k], ok_k[k], n_k[k]);
    n += n_k[k];
    ok += ok_k[k];
  }
  printf("ntext: %d/%d ok\n", ok, n);
  return ok == n && n > 0 ? 0 : 1;
}
