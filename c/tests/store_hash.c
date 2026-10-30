/* Prints "name ndim shape... sum of float32 bit patterns weighted by position" for every tensor of an Arc safetensors file (verify_store.py
 * prints the same from arc_audio's loader).
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <stdio.h>
#include <string.h>
#include "arc_base.h"
#include "arc_store.h"

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  arc_store* s = arc_store_open(argv[1]);
  if (!s) { fprintf(stderr, "%s\n", arc_geterr()); return 1; }
  for (int i = 0; i < arc_store_count(s); ++i) {
    const char* n = arc_store_name(s, i);
    int64_t sh[8];
    int nd = arc_store_shape(s, n, sh);
    int64_t cnt = 1;
    for (int k = 0; k < nd; ++k) cnt *= sh[k];
    const float* f = arc_store_f32(s, n, 0);
    if (!f) { printf("%s ERROR %s\n", n, arc_geterr()); continue; }
    uint64_t h = 0;   /* sum of (bits of element i) * (i + 1), mod 2^64 */
    for (int64_t i = 0; i < cnt; ++i) { uint32_t b; memcpy(&b, &f[i], 4); h += (uint64_t)b * (uint64_t)(i + 1); }
    printf("%s %lld %016llx\n", n, (long long)cnt, (unsigned long long)h);
  }
  arc_store_close(s);
  return 0;
}
