/* arc_alias against arc_audio kbsnap.py (tools/dump_alias.py): akey, and AliasIndex.lookup (row, via) on a table of
 * lists, heard texts and allowed rows: whole names / aliases with punctuation, case, spacing and possessive variants,
 * unique and shared partial names, edge command / stop words, stop / common / short words, ambiguous matches.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "arc_alias.h"
#include "arc_base.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: test_alias tests/data/alias.jsonl\n");
    return 2;
  }
  char* text = NULL;
  size_t len = 0;
  if (arc_read_file(argv[1], &text, &len)) {
    perror(argv[1]);
    return 2;
  }
  int n = 0, ok = 0, nk = 0, nl = 0, np = 0, na = 0, namb = 0;
  for (char *p = text, *e; *p; p = e + (*e != 0)) {
    e = strchr(p, '\n');
    if (!e) e = p + strlen(p);
    if (e == p) continue;
    arc_json* j = arc_json_parse(p, (size_t)(e - p));
    if (!j) continue;
    const char* k = arc_json_str(arc_json_get(j, "k"), "");
    if (!strcmp(k, "akey")) {
      const char* t = arc_json_str(arc_json_get(j, "text"), "");
      const char* want = arc_json_str(arc_json_get(j, "key"), "");
      char* got = arc_alias_akey(t);
      int good = !strcmp(got, want);
      n++, ok += good, nk++;
      if (!good) printf("akey [%s]: got [%s] want [%s]\n", t, got, want);
      arc_free(got);
    } else {
      const arc_json* kb = arc_json_get(j, "kb");
      arc_kbent* ents = calloc((size_t)kb->n + 1, sizeof *ents);
      const char** als[64];
      for (int i = 0; i < kb->n && i < 64; ++i) {
        const arc_json* c = arc_json_at(kb, i);
        ents[i].name = arc_json_str(arc_json_get(c, "name"), "");
        ents[i].kind = arc_json_str(arc_json_get(c, "kind"), "person");
        const arc_json* a = arc_json_get(c, "aliases");
        als[i] = NULL;
        if (a && a->n) {
          als[i] = calloc((size_t)a->n, sizeof(char*));
          for (int q = 0; q < a->n; ++q) als[i][q] = arc_json_str(arc_json_at(a, q), "");
          ents[i].aliases = als[i];
          ents[i].n_aliases = a->n;
        }
      }
      arc_alias* ix = arc_alias_new(ents, kb->n);
      const arc_json* al = arc_json_get(j, "allowed");
      int rows[64], nr = 0, has_al = al && al->type == ARC_JARR;
      for (int i = 0; has_al && i < al->n && i < 64; ++i) rows[nr++] = (int)arc_json_num(arc_json_at(al, i), -1);
      const char* heard = arc_json_str(arc_json_get(j, "heard"), "");
      const char* via;
      int row = arc_alias_lookup(ix, heard, has_al ? rows : NULL, nr, &via);
      const arc_json* wr = arc_json_get(j, "row");
      int want_row = wr && wr->type == ARC_JNUM ? (int)wr->num : -1;
      const char* want_via = arc_json_str(arc_json_get(j, "via"), NULL);
      int good = row == want_row && (via == NULL) == (want_via == NULL) && (!via || !strcmp(via, want_via));
      n++, ok += good, nl++;
      if (want_via && !strcmp(want_via, "partial")) np++;
      else if (want_via) na++;
      else namb++;
      if (!good)
        printf("lookup %s [%s] allowed %s: got %d %s want %d %s\n", arc_json_str(arc_json_get(j, "list"), "?"), heard,
               has_al ? "set" : "None", row, via ? via : "None", want_row, want_via ? want_via : "None");
      arc_alias_free(ix);
      for (int i = 0; i < kb->n && i < 64; ++i) free(als[i]);
      free(ents);
    }
    arc_json_free(j);
  }
  arc_free(text);
  printf("test_alias: %d / %d identical to kbsnap.py (%d akey; %d lookups: %d partial, %d alias, %d unlinked)\n", ok, n, nk, nl,
         np, na, namb);
  return ok == n && n > 0 ? 0 : 1;
}
