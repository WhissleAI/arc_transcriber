/* Moonshine port against the reference (tools/dump_moon.py): encoder states, then the final beam token ids.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "arc_base.h"
#include "arc_moon.h"
#include "npy.h"

static int parse_ids(const char* s, int* out, int cap) {
  int n = 0;
  while (*s && n < cap) {
    while (*s && (*s < '0' || *s > '9')) s++;
    if (!*s) break;
    out[n++] = (int)strtol(s, (char**)&s, 10);
  }
  return n;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: test_moon MOON_DIR REF_DIR [beams]\n"); return 2; }
  int beams = argc > 3 ? atoi(argv[3]) : 4;
  arc_moon* m = arc_moon_load(argv[1], 4);
  if (!m) { fprintf(stderr, "%s\n", arc_geterr()); return 1; }
  /* Arc's processors (config.json transcription.tag_offsets; NEVER_EMIT; the repeat guard) */
  static const int off_id[] = {32603, 32609, 32612, 32613, 32621, 32624, 32620, 32602};
  static const float off_val[] = {-1.25f, -0.5f, -0.75f, -0.25f, -0.25f, 0.5f, -1.5f, -0.75f};
  int sup[200], ns = 0;
  sup[ns++] = 32630; sup[ns++] = 32631;
  for (int i = 32636; i < 32768; ++i) sup[ns++] = i;
  arc_moon_procs pr = {8, off_id, off_val, ns, sup, 1, 8, 4, 6, NULL};
  char path[2048];
  snprintf(path, sizeof path, "%s/cases.tsv", argv[2]);
  FILE* f = fopen(path, "r");
  if (!f) { fprintf(stderr, "no %s\n", path); return 1; }
  char line[8192];
  int n = 0, same = 0;
  double worst = 0, t_enc = 0, t_dec = 0, audio = 0;
  while (fgets(line, sizeof line, f)) {
    char* stem = strtok(line, "\t");
    int max_len = atoi(strtok(NULL, "\t"));
    char* ids_s = strtok(NULL, "\t");
    int ref[512], nref = parse_ids(ids_s, ref, 512);
    int nd; long sh[4];
    snprintf(path, sizeof path, "%s/%s.wav.npy", argv[2], stem);
    float* wav = npy_load(path, &nd, sh);
    long nw = sh[0];
    snprintf(path, sizeof path, "%s/%s.H.npy", argv[2], stem);
    float* Href = npy_load(path, &nd, sh);
    long Tr = sh[0];
    int nn = (int)(nw / 80 * 80);
    if (nn < 80) nn = 80;
    double t0 = arc_time_ms();
    int T;
    float* Hc = arc_moon_encode(m, wav, nn, &T);
    double t1 = arc_time_ms();
    double md = 0, ma = 0;
    if (T == Tr)
      for (long i = 0; i < (long)T * 320; ++i) {
        double d = fabs((double)Hc[i] - Href[i]);
        if (d > md) md = d;
        if (fabs(Href[i]) > ma) ma = fabs(Href[i]);
      }
    if (md > worst) worst = md;
    int out[512];
    int no = arc_moon_decode(m, Hc, T, NULL, 0, max_len, beams, &pr, out, 512);
    double t2 = arc_time_ms();
    t_enc += t1 - t0; t_dec += t2 - t1; audio += nw / 16.0;
    int eq = no == nref && !memcmp(out, ref, sizeof(int) * (size_t)no);
    same += eq; n++;
    if (!eq || argc > 4) {
      printf("%s T=%d/%ld maxdiff=%.2e (|H|max %.1f)  %s\n  ref:", stem, T, Tr, md, ma, eq ? "same" : "DIFF");
      for (int i = 0; i < nref; ++i) printf(" %d", ref[i]);
      printf("\n  c  :");
      for (int i = 0; i < no; ++i) printf(" %d", out[i]);
      printf("\n");
    }
    free(wav); free(Href); arc_free(Hc);
  }
  fclose(f);
  printf("%d/%d utterances with identical tokens; encoder max |diff| %.2e; encode %.0f ms + decode %.0f ms for %.1f s of audio\n",
         same, n, worst, t_enc, t_dec, audio / 1000);
  arc_moon_free(m);
  return same == n ? 0 : 1;
}
