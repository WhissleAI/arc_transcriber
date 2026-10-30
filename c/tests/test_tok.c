/* Tokenizer against HF tokenizers (tools/dump_tok.py). Strings are JSON-quoted in the cases file.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "arc_base.h"
#include "arc_tok.h"

static char* unjson(const char* q) {
  arc_json* j = arc_json_parse(q, strlen(q));
  char* s = j && j->type == ARC_JSTR ? arc_strdup(j->str) : arc_strdup("");
  arc_json_free(j);
  return s;
}

int main(int argc, char** argv) {
  arc_tok* t = arc_tok_load(argv[1]);
  if (!t) { fprintf(stderr, "%s\n", arc_geterr()); return 1; }
  FILE* f = fopen(argv[2], "r");
  static char line[1 << 16];
  int n = 0, ok = 0;
  while (fgets(line, sizeof line, f)) {
    line[strcspn(line, "\n")] = 0;
    char* kind = strtok(line, "\t");
    char* a = strtok(NULL, "\t");
    char* b = strtok(NULL, "\t");
    if (!b) b = "";
    n++;
    if (kind[0] == 'E') {
      char* s = unjson(a);
      int ids[4096], m = arc_tok_encode(t, s, ids, 4096);
      char got[1 << 15];
      size_t k = 0;
      got[0] = 0;
      for (int i = 0; i < m; ++i) k += (size_t)snprintf(got + k, sizeof got - k, i ? " %d" : "%d", ids[i]);
      if (!strcmp(got, b)) ok++;
      else printf("ENCODE %s\n  hf: %s\n  c : %s\n", a, b, got);
      arc_free(s);
    } else {
      int ids[4096], m = 0;
      for (char* p = a; *p;) { ids[m++] = (int)strtol(p, &p, 10); while (*p == ' ') p++; }
      char* want = unjson(b);
      char* got = arc_tok_decode(t, ids, m);
      if (!strcmp(got, want)) ok++;
      else printf("DECODE %s\n  hf: %s\n  c : %s\n", a, want, got);
      arc_free(want);
      arc_free(got);
    }
  }
  printf("%d/%d tokenizer cases identical\n", ok, n);
  arc_tok_free(t);
  return ok == n ? 0 : 1;
}
