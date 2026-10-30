/* arc-trim: a copy of an Arc safetensors file with only the tensors whose (logical) names start with the given
 * prefixes; packed 4-bit tensors keep their .q / .s / .z parts and their int4_layout entries, so libarc (and
 * arc_audio) read the result exactly as the original. Tensor bytes are copied unchanged, in the original order.
 *   arc-trim IN.safetensors OUT.safetensors trunk. head. diar.slot ent.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arc_base.h"

static int keep_name(const char* n, char** pre, int np) {
  for (int i = 0; i < np; ++i)
    if (!strncmp(n, pre[i], strlen(pre[i]))) return 1;
  return 0;
}

/* JSON text of a DOM value */
static void emit(const arc_json* j, FILE* f) {
  switch (j->type) {
  case ARC_JNULL: fputs("null", f); break;
  case ARC_JBOOL: fputs(j->num ? "true" : "false", f); break;
  case ARC_JNUM:
    if (j->is_int) fprintf(f, "%lld", (long long)j->num);
    else fprintf(f, "%.17g", j->num);
    break;
  case ARC_JSTR:
    fputc('"', f);
    for (const char* p = j->str; *p; ++p) {
      if (*p == '"' || *p == '\\') fputc('\\', f);
      fputc(*p, f);
    }
    fputc('"', f);
    break;
  case ARC_JARR:
    fputc('[', f);
    for (int i = 0; i < j->n; ++i) {
      if (i) fputc(',', f);
      emit(&j->items[i], f);
    }
    fputc(']', f);
    break;
  case ARC_JOBJ:
    fputc('{', f);
    for (int i = 0; i < j->n; ++i) {
      if (i) fputc(',', f);
      arc_json k = {.type = ARC_JSTR, .str = j->keys[i]};
      emit(&k, f);
      fputc(':', f);
      emit(&j->items[i], f);
    }
    fputc('}', f);
    break;
  }
}

/* a JSON string literal holding s */
static void emit_str(const char* s, FILE* f) {
  arc_json k = {.type = ARC_JSTR, .str = (char*)s};
  emit(&k, f);
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: arc-trim IN.safetensors OUT.safetensors PREFIX...\n");
    return 2;
  }
  char* data;
  size_t len;
  if (arc_read_file(argv[1], &data, &len) || len < 8) {
    fprintf(stderr, "cannot read %s\n", argv[1]);
    return 1;
  }
  uint64_t hl = 0;
  for (int i = 0; i < 8; ++i) hl |= (uint64_t)(unsigned char)data[i] << (8 * i);
  arc_json* h = arc_json_parse(data + 8, (size_t)hl);
  if (!h) {
    fprintf(stderr, "bad header\n");
    return 1;
  }
  const char* body = data + 8 + hl;
  char** pre = argv + 3;
  int np = argc - 3;
  const arc_json* meta = arc_json_get(h, "__metadata__");
  const char* lay_s = arc_json_str(arc_json_get(meta, "int4_layout"), NULL);
  arc_json* lay = lay_s ? arc_json_parse(lay_s, strlen(lay_s)) : NULL;
  /* kept tensors, in file order (sorted by their offsets) */
  int n = h->n, nk = 0;
  int* idx = malloc(sizeof(int) * (size_t)n);
  for (int i = 0; i < n; ++i) {
    const char* name = h->keys[i];
    if (!strcmp(name, "__metadata__")) continue;
    /* a packed part (W.q / W.s / W.z with W in the layout) is judged by W's name */
    char base[512];
    snprintf(base, sizeof base, "%s", name);
    size_t l = strlen(base);
    if (lay && l > 2 && base[l - 2] == '.' && strchr("qsz", base[l - 1])) {
      base[l - 2] = 0;
      if (!arc_json_get(lay, base)) snprintf(base, sizeof base, "%s", name);
    }
    if (keep_name(base, pre, np)) idx[nk++] = i;
  }
  for (int a = 1; a < nk; ++a)   /* by start offset */
    for (int b = a; b > 0; --b) {
      double oa = arc_json_num(arc_json_at(arc_json_get(&h->items[idx[b]], "data_offsets"), 0), 0);
      double ob = arc_json_num(arc_json_at(arc_json_get(&h->items[idx[b - 1]], "data_offsets"), 0), 0);
      if (oa >= ob) break;
      int t = idx[b];
      idx[b] = idx[b - 1];
      idx[b - 1] = t;
    }
  /* header */
  char* hb = NULL;
  size_t hs = 0;
  FILE* hf = open_memstream(&hb, &hs);
  fputc('{', hf);
  fputs("\"__metadata__\":{", hf);
  int first = 1;
  for (int i = 0; meta && i < meta->n; ++i) {
    if (!first) fputc(',', hf);
    first = 0;
    emit_str(meta->keys[i], hf);
    fputc(':', hf);
    if (!strcmp(meta->keys[i], "int4_layout") && lay) {
      char* lb = NULL;
      size_t ls = 0;
      FILE* lf = open_memstream(&lb, &ls);
      fputc('{', lf);
      int f2 = 1;
      for (int k = 0; k < lay->n; ++k)
        if (keep_name(lay->keys[k], pre, np)) {
          if (!f2) fputc(',', lf);
          f2 = 0;
          emit_str(lay->keys[k], lf);
          fputc(':', lf);
          emit(&lay->items[k], lf);
        }
      fputc('}', lf);
      fclose(lf);
      emit_str(lb, hf);
      free(lb);
    } else emit(&meta->items[i], hf);
  }
  fputc('}', hf);
  uint64_t off = 0;
  for (int k = 0; k < nk; ++k) {
    const arc_json* t = &h->items[idx[k]];
    const arc_json* o = arc_json_get(t, "data_offsets");
    uint64_t a = (uint64_t)arc_json_num(arc_json_at(o, 0), 0), b = (uint64_t)arc_json_num(arc_json_at(o, 1), 0);
    fputc(',', hf);
    emit_str(h->keys[idx[k]], hf);
    fprintf(hf, ":{\"dtype\":");
    emit(arc_json_get(t, "dtype"), hf);
    fprintf(hf, ",\"shape\":");
    emit(arc_json_get(t, "shape"), hf);
    fprintf(hf, ",\"data_offsets\":[%llu,%llu]}", (unsigned long long)off, (unsigned long long)(off + (b - a)));
    off += b - a;
  }
  fputc('}', hf);
  fclose(hf);
  while (hs % 8) {   /* pad the header to 8 bytes with spaces, as safetensors writers do */
    hb = realloc(hb, hs + 2);
    hb[hs++] = ' ';
    hb[hs] = 0;
  }
  FILE* out = fopen(argv[2], "wb");
  if (!out) {
    fprintf(stderr, "cannot write %s\n", argv[2]);
    return 1;
  }
  for (int i = 0; i < 8; ++i) fputc((int)((uint64_t)hs >> (8 * i)) & 255, out);
  fwrite(hb, 1, hs, out);
  for (int k = 0; k < nk; ++k) {
    const arc_json* o = arc_json_get(&h->items[idx[k]], "data_offsets");
    uint64_t a = (uint64_t)arc_json_num(arc_json_at(o, 0), 0), b = (uint64_t)arc_json_num(arc_json_at(o, 1), 0);
    fwrite(body + a, 1, (size_t)(b - a), out);
  }
  fclose(out);
  fprintf(stderr, "%s: %d of %d tensors, %.1f MB\n", argv[2], nk, n - (meta ? 1 : 0), (double)(8 + hs + off) / 1e6);
  free(hb);
  free(idx);
  arc_json_free(lay);
  arc_json_free(h);
  arc_free(data);
  return 0;
}
