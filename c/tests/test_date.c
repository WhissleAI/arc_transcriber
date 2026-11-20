/* arc_date against arc_audio's spoken dates (tools/dump_date.py): written.parse_date / date_iso, and a date record
 * through api.extend_date + tidy.tidy_number (heard, start, end, date, iso).
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "arc_base.h"
#include "arc_date.h"
#include "arc_tidy.h"

static int n = 0, ok = 0;
static void check(int good, const char* what, const char* in, const char* got, const char* want) {
  n++;
  ok += good;
  if (!good) printf("%s [%s]: got [%s] want [%s]\n", what, in, got, want);
}

/* the fixture's date (null or {year, month, day}) as arc_ymd; 0: null */
static int want_date(const arc_json* j, arc_ymd* d) {
  const arc_json* x = arc_json_get(j, "date");
  memset(d, 0, sizeof *d);
  if (!x || !arc_json_get(x, "month")) return 0;
  d->year = (int)arc_json_num(arc_json_get(x, "year"), 0);
  d->month = (int)arc_json_num(arc_json_get(x, "month"), 0);
  d->day = (int)arc_json_num(arc_json_get(x, "day"), 0);
  return 1;
}
static void fmt(char* b, int has, const arc_ymd* d, const char* iso) {
  if (!has) snprintf(b, 96, "None");
  else snprintf(b, 96, "%d-%d-%d iso %s", d->year, d->month, d->day, iso && iso[0] ? iso : "None");
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: test_date tests/data/date.jsonl\n");
    return 2;
  }
  FILE* f = fopen(argv[1], "r");
  if (!f) {
    perror(argv[1]);
    return 2;
  }
  static char line[1 << 16];
  int np = 0, ne = 0;
  while (fgets(line, sizeof line, f)) {
    arc_json* j = arc_json_parse(line, strlen(line));
    const char* k = arc_json_str(arc_json_get(j, "k"), "");
    const arc_json* W = arc_json_get(j, "words");
    const char* wiso = arc_json_str(arc_json_get(j, "iso"), "");
    arc_ymd want, got;
    int has_want = want_date(j, &want);
    char in[512] = "", gs[96], ws[96];
    if (!strcmp(k, "parse")) {
      const char* w[64];
      for (int i = 0; i < W->n && i < 64; ++i) {
        w[i] = arc_json_str(arc_json_at(W, i), "");
        if (i) strcat(in, " ");
        strncat(in, w[i], sizeof in - strlen(in) - 2);
      }
      int has = arc_parse_date(w, W->n, &got);
      char iso[16] = "";
      if (has) arc_date_iso(&got, iso);
      fmt(gs, has, &got, iso);
      fmt(ws, has_want, &want, wiso);
      check(has == has_want && (!has || (!memcmp(&got, &want, sizeof got) && !strcmp(iso, wiso))), "parse_date", in, gs, ws);
      np++;
    } else if (!strcmp(k, "extend")) {
      arc_tword tw[64];
      for (int i = 0; i < W->n && i < 64; ++i) {
        const arc_json* x = arc_json_at(W, i);
        tw[i] = (arc_tword){arc_json_str(arc_json_at(x, 0), ""), arc_json_num(arc_json_at(x, 1), 0),
                            arc_json_num(arc_json_at(x, 2), 0)};
        if (i) strcat(in, " ");
        strncat(in, tw[i].w, sizeof in - strlen(in) - 2);
      }
      char* heard = arc_strdup(arc_json_str(arc_json_get(j, "heard_in"), ""));
      double s = arc_json_num(arc_json_get(j, "start_in"), 0), e = arc_json_num(arc_json_get(j, "end_in"), 0);
      int has = arc_extend_date(tw, W->n, (int)arc_json_num(arc_json_get(j, "i"), 0),
                                (int)arc_json_num(arc_json_get(j, "j"), 0), &heard, &s, &e, &got);
      char* h = arc_tidy_number(heard, "date");
      char iso[16] = "";
      if (has) arc_date_iso(&got, iso);
      const char* wh = arc_json_str(arc_json_get(j, "heard"), "");
      check(!strcmp(h, wh), "extend heard", in, h, wh);
      char a[64], b[64];
      snprintf(a, sizeof a, "%.3f-%.3f", s, e);
      snprintf(b, sizeof b, "%.3f-%.3f", arc_json_num(arc_json_get(j, "start"), 0), arc_json_num(arc_json_get(j, "end"), 0));
      check(!strcmp(a, b), "extend times", in, a, b);
      fmt(gs, has, &got, iso);
      fmt(ws, has_want, &want, wiso);
      check(has == has_want && (!has || (!memcmp(&got, &want, sizeof got) && !strcmp(iso, wiso))), "extend date", in, gs, ws);
      arc_free(heard);
      arc_free(h);
      ne++;
    }
    arc_json_free(j);
  }
  fclose(f);
  printf("%d/%d date checks identical to arc_audio (%d parse_date phrasings, %d extend_date records)\n", ok, n, np, ne);
  return ok == n ? 0 : 1;
}
