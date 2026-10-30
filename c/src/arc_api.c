/* libarc public API: the release folder, the transcript (Arc.transcribe), and (arc_trunk.c / arc_heads.c) the trunk
 * readout and events.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include "arc.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arc_base.h"
#include "arc_moon.h"
#include "arc_text.h"
#include "arc_tok.h"
#include "arc_engine.h"

/* ------------------------------------------------------------------ config */

static int load_cfg(arc_engine* e, const char* dir) {
  char path[1200];
  snprintf(path, sizeof path, "%s/config.json", dir);
  char* text = NULL;
  size_t len = 0;
  if (arc_read_file(path, &text, &len)) return -1;
  e->cfg = arc_json_parse(text, len);
  arc_free(text);
  if (!e->cfg) return -1;
  const arc_json* tr = arc_json_get(e->cfg, "transcription");
  arc_seg_cfg_default(&e->seg);
  e->seg.pause_s = (float)arc_json_num(arc_json_get(tr, "pause_s"), 0.4);
  e->seg.min_s = (float)arc_json_num(arc_json_get(tr, "min_s"), 1.0);
  e->seg.max_s = (float)arc_json_num(arc_json_get(tr, "max_s"), 15.0);
  e->seg.cut_lookback_s = (float)arc_json_num(arc_json_get(tr, "cut_lookback_s"), 3.0);
  e->seg.min_voiced_s = (float)arc_json_num(arc_json_get(tr, "min_voiced_s"), 0.3);
  e->pad_s = arc_json_num(arc_json_get(tr, "pad_s"), 0.15);
  e->tokens_per_s = arc_json_num(arc_json_get(tr, "tokens_per_s"), 7.0);
  e->len_mult = arc_json_num(arc_json_get(arc_json_get(arc_json_get(tr, "decoders"), "tagged"), "len_mult"), 1.3);
  e->loop_guard = (int)arc_json_num(arc_json_get(tr, "loop_guard"), 1);
  e->beams = 4;   /* Arc.TRANSCRIPT_CFG['builtin'].final_beams */
  /* tag offsets, suppressed ids (reserved tag slots and everything above the tag range), repeat guard */
  const arc_json* off = arc_json_get(tr, "tag_offsets");
  e->n_off = 0;
  if (off && off->type == ARC_JOBJ)
    for (int i = 0; i < off->n && e->n_off < 64; ++i) {
      int id = arc_tag_id(off->keys[i]);
      float v = (float)off->items[i].num;
      if (id >= 0 && v != 0.0f) {
        e->off_id[e->n_off] = id;
        e->off_val[e->n_off++] = v;
      }
    }
  e->n_sup = 0;
  e->sup[e->n_sup++] = 32630;
  e->sup[e->n_sup++] = 32631;
  for (int i = 32636; i < ARC_MOON_VOCAB; ++i) e->sup[e->n_sup++] = i;
  arc_align_cfg_load(&e->align, arc_json_get(e->cfg, "alignment"));
  e->procs = (arc_moon_procs){e->n_off, e->off_id, e->off_val, e->loop_guard ? e->n_sup : 0, e->sup, e->loop_guard,
                              8, 4, 6, NULL};
  return 0;
}

/* ------------------------------------------------------------------ lifecycle */

static __thread char g_err[512];
const char* arc_last_error(void) { return g_err[0] ? g_err : arc_geterr(); }

arc_engine* arc_create(const char* dir, int threads, char* err, size_t err_len) {
  arc_engine* e = arc_calloc(sizeof *e);
  if (!e) goto fail;
  if (threads <= 0) threads = 4;
  e->dir = arc_strdup(dir);
  if (load_cfg(e, dir)) goto fail;
  char path[1200];
  snprintf(path, sizeof path, "%s/%s", dir,
           arc_json_str(arc_json_get(arc_json_get(e->cfg, "encoder"), "subfolder"), "moonshine-streaming-tiny"));
  if (!(e->moon = arc_moon_load(path, threads))) goto fail;
  char tp[1300];
  snprintf(tp, sizeof tp, "%s/tokenizer.json", path);
  if (!(e->tok = arc_tok_load(tp))) goto fail;
  if (arc_engine_load_trunk(e, dir)) goto fail;   /* trunk, heads, ontology (arc_engine.c) */
  return e;
fail:
  if (err && err_len) snprintf(err, err_len, "%s", arc_geterr());
  arc_destroy(e);
  return NULL;
}

int arc_set_threads(arc_engine* e, int threads) {
  if (threads <= 0) threads = 4;
  if (arc_pool_size(e->moon->pool) == threads) return 0;
  arc_pool* p = arc_pool_create(threads);
  if (!p) return -1;
  arc_pool_destroy(e->moon->pool);
  e->moon->pool = p;
  if (e->trunk) arc_engine_set_pool(e, p);
  return 0;
}

void arc_destroy(arc_engine* e) {
  if (!e) return;
  arc_engine_free_trunk(e);
  arc_moon_free(e->moon);
  arc_tok_free(e->tok);
  arc_json_free(e->cfg);
  for (int i = 0; i < e->ctx_n; ++i) {
    arc_free(e->ctx_kb[i]);
    arc_free(e->ctx_ty[i]);
  }
  arc_free(e->ctx_kb);
  arc_free(e->ctx_ty);
  arc_free(e->ctx_dom);
  arc_free(e->ctx_prev);
  arc_free(e->nb_off);
  arc_free(e->nb_ids);
  arc_free(e->nb_w);
  for (int i = 0; i < e->n_hist; ++i) arc_free(e->hist[i].text);
  arc_free(e->dir);
  arc_free(e);
}

/* ------------------------------------------------------------------ context */

static const char* kb_kind(const char* k) {
  if (!k) return "term";
  if (!strcmp(k, "person") || !strcmp(k, "team") || !strcmp(k, "place") || !strcmp(k, "law") ||
      !strcmp(k, "product"))
    return k;
  if (!strcmp(k, "organization") || !strcmp(k, "org")) return "org";
  return "term";
}

/* The decoder bias (bias.py Trie) over bias_forms(names): each name (spaces normalised), its last word when it has
 * several (not a suffix like Jr / II), sorted and unique; tokenised as tok(form) without BOS. */
/* extra: phrases taken as they are (aliases), spaces normalised */
static void build_bias(arc_engine* e, const char* const* names, int n_names, const char* const* extra, int n_extra) {
  arc_free(e->nb_off);
  arc_free(e->nb_ids);
  arc_free(e->nb_w);
  e->nb_off = e->nb_ids = NULL;
  e->nb_w = NULL;
  e->nb_seq = e->nb_maxlen = 0;
  if (n_names < 0) n_names = 0;
  if (n_extra < 0) n_extra = 0;
  if (n_names + n_extra > 0) {
    int nf = 0;
    char** forms = arc_calloc(sizeof(char*) * (size_t)(2 * n_names + n_extra + 1));
    for (int i = 0; i < n_extra; ++i) {
      char** w;
      int nw = arc_split_words(extra[i], &w);
      if (nw) {
        size_t l = 0;
        for (int k = 0; k < nw; ++k) l += strlen(w[k]) + 1;
        char* full = arc_malloc(l + 1);
        full[0] = 0;
        for (int k = 0; k < nw; ++k) {
          if (k) strcat(full, " ");
          strcat(full, w[k]);
        }
        forms[nf++] = full;
      }
      arc_free_words(w, nw);
    }
    for (int i = 0; i < n_names; ++i) {
      char** w;
      int nw = arc_split_words(names[i], &w);
      if (!nw) {
        arc_free_words(w, nw);
        continue;
      }
      size_t l = 0;
      for (int k = 0; k < nw; ++k) l += strlen(w[k]) + 1;
      char* full = arc_malloc(l + 1);
      full[0] = 0;
      for (int k = 0; k < nw; ++k) {
        if (k) strcat(full, " ");
        strcat(full, w[k]);
      }
      forms[nf++] = full;
      int last = -1;   /* parts without suffixes; its last one */
      int np = 0;
      for (int k = 0; k < nw; ++k) {
        char lw[64];
        size_t j = 0;
        for (const char* c = w[k]; *c && j < sizeof lw - 1; ++c)
          if (*c != '.') lw[j++] = (char)tolower((unsigned char)*c);
        lw[j] = 0;
        if (strcmp(lw, "jr") && strcmp(lw, "sr") && strcmp(lw, "ii") && strcmp(lw, "iii") && strcmp(lw, "iv")) {
          last = k;
          np++;
        }
      }
      if (np > 1 && last >= 0) forms[nf++] = arc_strdup(w[last]);
      arc_free_words(w, nw);
    }
    /* sorted(set(...)) */
    for (int a = 0; a < nf; ++a)
      for (int b = a + 1; b < nf; ++b)
        if (strcmp(forms[b], forms[a]) < 0) {
          char* t = forms[a];
          forms[a] = forms[b];
          forms[b] = t;
        }
    int total = 0, ids[512];
    e->nb_off = arc_malloc(sizeof(int) * (size_t)(nf + 1));
    e->nb_w = arc_malloc(sizeof(float) * (size_t)(nf + 1));
    e->nb_ids = arc_malloc(sizeof(int) * (size_t)(nf * 64 + 1));
    for (int a = 0; a < nf; ++a) {
      if (a && !strcmp(forms[a], forms[a - 1])) continue;
      int m = arc_tok_encode(e->tok, forms[a], ids, 512);
      if (m <= 0 || m > 64) continue;
      e->nb_off[e->nb_seq] = total;
      memcpy(e->nb_ids + total, ids, sizeof(int) * (size_t)m);
      e->nb_w[e->nb_seq++] = 1.0f;   /* type 'player', always on: weight 'inside' = 1.0 */
      total += m;
      if (m > e->nb_maxlen) e->nb_maxlen = m;
    }
    e->nb_off[e->nb_seq] = total;
    for (int a = 0; a < nf; ++a) arc_free(forms[a]);
    arc_free(forms);
  }
  e->bias = (arc_moon_bias){e->nb_seq, e->nb_off, e->nb_ids, e->nb_w, e->nb_start > 0 ? e->nb_start : 2.0f,
                            e->nb_cont > 0 ? e->nb_cont : 1.5f, e->nb_maxlen, 1};
}

int arc_set_bias(arc_engine* e, const char* const* names, int n) {
  build_bias(e, names, n, NULL, 0);
  return 0;
}

int arc_set_names(arc_engine* e, const arc_name_entry* names, int n) {
  if (!e || n < 0) return -1;
  if (arc_engine_set_names(e, names, n)) return -1;
  /* the decoder bias: bias_forms(kb), names with their last words, and aliases */
  int na = 0;
  for (int i = 0; i < n; ++i) na += names[i].n_aliases > 0 ? names[i].n_aliases : 0;
  const char** nm = arc_malloc(sizeof(char*) * (size_t)(n + 1));
  const char** al = arc_malloc(sizeof(char*) * (size_t)(na + 1));
  int k = 0;
  for (int i = 0; i < n; ++i) {
    nm[i] = names[i].name ? names[i].name : "";
    for (int j = 0; j < names[i].n_aliases; ++j) al[k++] = names[i].aliases[j] ? names[i].aliases[j] : "";
  }
  build_bias(e, nm, n, al, k);
  arc_free(nm);
  arc_free(al);
  return 0;
}

int arc_set_bias_strength(arc_engine* e, float start_bonus, float cont_bonus) {
  if (!e || start_bonus < 0 || cont_bonus < 0) return -1;
  e->nb_start = start_bonus;
  e->nb_cont = cont_bonus;
  e->bias.start_bonus = start_bonus > 0 ? start_bonus : 2.0f;
  e->bias.cont_bonus = cont_bonus > 0 ? cont_bonus : 1.5f;
  return 0;
}

int arc_set_context(arc_engine* e, const char* const* names, const char* const* kinds, int n_names,
                    const char* const* terms, int n_terms, const char* topic, const char* previous_text,
                    int auto_prev) {
  for (int i = 0; i < e->ctx_n; ++i) {
    arc_free(e->ctx_kb[i]);
    arc_free(e->ctx_ty[i]);
  }
  arc_free(e->ctx_kb);
  arc_free(e->ctx_ty);
  arc_free(e->ctx_dom);
  arc_free(e->ctx_prev);
  e->ctx_kb = NULL;
  e->ctx_ty = NULL;
  e->ctx_dom = e->ctx_prev = NULL;
  e->ctx_n = 0;
  int n = (n_names > 0 ? n_names : 0) + (n_terms > 0 ? n_terms : 0);
  if (n) {
    e->ctx_kb = arc_calloc(sizeof(char*) * (size_t)n);
    e->ctx_ty = arc_calloc(sizeof(char*) * (size_t)n);
    for (int i = 0; i < n_names; ++i) {
      e->ctx_kb[e->ctx_n] = arc_strdup(names[i]);
      e->ctx_ty[e->ctx_n++] = arc_strdup(kb_kind(kinds ? kinds[i] : NULL));
    }
    for (int i = 0; i < n_terms; ++i) {
      e->ctx_kb[e->ctx_n] = arc_strdup(terms[i]);
      e->ctx_ty[e->ctx_n++] = arc_strdup("term");
    }
  }
  if (topic && *topic) e->ctx_dom = arc_strdup(topic);
  if (previous_text && *previous_text) e->ctx_prev = arc_strdup(previous_text);
  e->auto_prev = auto_prev;
  e->has_ctx = e->ctx_n || e->ctx_dom || e->ctx_prev;
  build_bias(e, names, n_names, NULL, 0);
  return 0;
}

/* prev_text(hist, t_now): normalised words of utterances in the last 60 s that ended 0.3 s before now, last 40 */
static char* auto_prev_text(arc_engine* e, double t_now) {
  char buf[8192];
  size_t k = 0;
  buf[0] = 0;
  for (int i = 0; i < e->n_hist; ++i) {
    if (e->hist[i].s >= t_now - 60.0 && e->hist[i].e <= t_now - 0.3) {
      char* w = arc_norm_prev(e->hist[i].text);
      size_t l = strlen(w);
      if (l && k + l + 2 < sizeof buf) {
        if (k) buf[k++] = ' ';
        memcpy(buf + k, w, l);
        k += l;
        buf[k] = 0;
      }
      arc_free(w);
    }
  }
  char** words;
  int nw = arc_split_words(buf, &words);
  int from = nw > 40 ? nw - 40 : 0;
  size_t cap = strlen(buf) + 2, o = 0;
  char* out = arc_malloc(cap);
  out[0] = 0;
  for (int i = from; i < nw; ++i) {
    size_t l = strlen(words[i]);
    if (o) out[o++] = ' ';
    memcpy(out + o, words[i], l);
    o += l;
    out[o] = 0;
  }
  arc_free_words(words, nw);
  return out;
}

static int prompt_ids(arc_engine* e, double t_now, int* out) {
  if (!e->has_ctx && !e->auto_prev) return 0;
  char* prev = NULL;
  if (e->auto_prev && !e->ctx_prev) prev = auto_prev_text(e, t_now);
  arc_ctx c = {e->ctx_dom, (const char* const*)e->ctx_kb, (const char* const*)e->ctx_ty, e->ctx_n,
               e->ctx_prev ? e->ctx_prev : prev};
  int n = arc_ctx_prompt(e->tok, &c, e->ctx_n ? "full" : "prev", 96, out);
  arc_free(prev);
  return n;
}

/* ------------------------------------------------------------------ transcript */

/* Decode one audio segment (Transcriber.decode_audio + utterance): text and tag spans. */
static int decode_segment(arc_engine* e, const float* seg, int n_in, double t_now, int beams, int bias, int words,
                          double t0, arc_utt_build* u) {
  int n = n_in < 3200 ? 3200 : n_in;
  float* w = arc_calloc(sizeof(float) * (size_t)n);
  if (!w) return -1;
  memcpy(w, seg, sizeof(float) * (size_t)n_in);
  int nn = n / 80 * 80;
  if (nn < 80) nn = 80;
  int T;
  float* H = arc_moon_encode(e->moon, w, nn, &T);
  arc_free(w);
  if (!H) return -1;
  int P[128];
  int np = prompt_ids(e, t_now, P);
  int max_len = 1 + (int)(e->len_mult * (4.0 + e->tokens_per_s * (double)n / ARC_SAMPLE_RATE)) + np;
  int ids[1024];
  arc_moon_procs pr = e->procs;
  pr.bias = bias && e->nb_seq ? &e->bias : NULL;
  int ni = arc_moon_decode(e->moon, H, T, P, np, max_len, beams, &pr, ids, 1024);
  if (ni < 0) {
    arc_free(H);
    return -1;
  }
  char* raw = arc_tok_decode(e->tok, ids, ni);
  char* txt = arc_loopfix(raw, 2);
  /* tag spans: readout words -> raw words -> loop-fixed words */
  char** rw;
  int nrw;
  arc_span* sp;
  int nsp;
  arc_readout(e->tok, ids, ni, &rw, &nrw, &sp, &nsp);
  char** raww;
  int nraw = arc_split_words(raw, &raww);
  arc_span* s1 = arc_malloc(sizeof(arc_span) * (size_t)(nsp + 1));
  int n1 = arc_map_spans(rw, nrw, sp, nsp, raww, nraw, s1);
  char** txw;
  int ntx = arc_split_words(txt, &txw);
  arc_span* s2 = arc_malloc(sizeof(arc_span) * (size_t)(n1 + 1));
  int n2 = strcmp(txt, raw) ? arc_map_spans(raww, nraw, s1, n1, txw, ntx, s2) : n1;
  if (!strcmp(txt, raw)) memcpy(s2, s1, sizeof(arc_span) * (size_t)n1);
  u->text = txt;
  u->spans = s2;
  u->n_spans = n2;
  u->words = txw;
  u->n_words = ntx;
  if (words && txt[0] && n_in >= 800) {
    /* Arc._words_for: align the text over this segment (unpadded: the decode's states are the same once the
     * segment is 0.2 s or longer), then tag_words */
    float* Ha = H;
    int Ta = T;
    if (n_in < 3200) Ha = arc_moon_encode(e->moon, seg, n_in / 80 * 80, &Ta);
    arc_aword* aw = NULL;
    int na = Ha ? arc_align_words(e->moon, e->tok, &e->align, Ha, Ta, n_in, txt, t0, &aw) : -1;
    if (Ha != H) arc_free(Ha);
    if (na < 0) {
      arc_free(H);
      return -1;
    }
    u->aw = arc_calloc(sizeof(arc_wordb) * (size_t)(na + 1));
    char** dst = arc_malloc(sizeof(char*) * (size_t)(na + 1));
    for (int k = 0; k < na; ++k) {
      u->aw[k] = (arc_wordb){aw[k].w, aw[k].start, aw[k].end, -1, 0};
      dst[k] = aw[k].w;
    }
    u->n_aw = na;
    arc_free(aw);
    if (n2 && na) {   /* tag_words: the spans over text.split() moved onto the aligned words */
      arc_span* s3 = arc_malloc(sizeof(arc_span) * (size_t)(n2 + 1));
      int n3 = arc_map_spans(txw, ntx, s2, n2, dst, na, s3);
      for (int k = 0; k < n3; ++k)
        for (int j = s3[k].a > 0 ? s3[k].a : 0; j < s3[k].b && j < na; ++j) {
          u->aw[j].tag = s3[k].tag;
          u->aw[j].tag_begin = j == s3[k].a;
        }
      arc_free(s3);
    }
    arc_free(dst);
  }
  arc_free(H);
  arc_free(raw);
  arc_free(s1);
  arc_free(sp);
  arc_free_words(rw, nrw);
  arc_free_words(raww, nraw);
  return 0;
}

arc_result* arc_transcribe(arc_engine* e, const float* pcm, size_t n, const arc_options* opt) {
  double t0 = arc_time_ms();
  g_err[0] = 0;
  arc_options o = {0};
  if (opt) o = *opt;
  int beams = o.beams > 0 ? o.beams : e->beams;
  /* utterances: Arc's segmenter on 100 ms frame levels, or the whole clip */
  int cap = 64, nu = 0;
  int(*seg)[2] = arc_malloc(sizeof(int[2]) * (size_t)cap);
  const int FR = 1600;
  int nf = (int)((n + FR - 1) / FR);
  if (o.one_utterance) {
    /* one utterance, but never longer than Arc decodes well (transcription.max_s, 15 s): a longer clip is cut where
     * the segmenter would cut it, at the quietest frame in the last cut_lookback_s before the limit. Decoded whole, a
     * 38 s clip ran off into repeats ("falling over and over ... Pink."). */
    int maxf = (int)(e->seg.max_s * 10 + 0.5f), look = (int)(e->seg.cut_lookback_s * 10 + 0.5f), f0 = 0;
    if (maxf < 10) maxf = 10;
    if (look < 1 || look > maxf) look = maxf / 5 > 0 ? maxf / 5 : 1;
    while (nf - f0 > maxf) {
      int cut = f0 + maxf - 1;
      double lo = 1e9;
      for (int f = f0 + maxf - look; f < f0 + maxf; ++f) {
        size_t a = (size_t)f * FR, b = a + FR < n ? a + FR : n;
        double db = arc_frame_db(pcm + a, (int)(b - a));
        if (db < lo) { lo = db; cut = f; }
      }
      if (nu == cap) {
        int(*ns)[2] = arc_malloc(sizeof(int[2]) * (size_t)cap * 2);
        memcpy(ns, seg, sizeof(int[2]) * (size_t)nu);
        arc_free(seg);
        seg = ns;
        cap *= 2;
      }
      seg[nu][0] = f0;
      seg[nu++][1] = cut;
      f0 = cut + 1;
    }
    if (nf > f0 && nu < cap) {
      seg[nu][0] = f0;
      seg[nu++][1] = nf - 1;
    }
  } else {
    arc_seg* s = arc_seg_new(&e->seg);
    int f0, f1;
    const char* why;
    for (int f = 0; f < nf; ++f) {
      size_t a = (size_t)f * FR, b = a + FR < n ? a + FR : n;
      if (arc_seg_feed(s, f, arc_frame_db(pcm + a, (int)(b - a)), &f0, &f1, &why)) {
        if (nu == cap) {
          int(*ns)[2] = arc_malloc(sizeof(int[2]) * (size_t)cap * 2);
          memcpy(ns, seg, sizeof(int[2]) * (size_t)nu);
          arc_free(seg);
          seg = ns;
          cap *= 2;
        }
        seg[nu][0] = f0;
        seg[nu++][1] = f1;
      }
    }
    if (arc_seg_flush(s, &f0, &f1, &why) && nu < cap) {
      seg[nu][0] = f0;
      seg[nu++][1] = f1;
    }
    arc_seg_free(s);
  }
  arc_result_build* R = arc_result_new(nu);
  int pad = (int)(e->pad_s * ARC_SAMPLE_RATE);
  for (int i = 0; i < nu; ++i) {
    long a = (long)seg[i][0] * FR, b = (long)(seg[i][1] + 1) * FR;
    /* the whole clip as one utterance decodes all of it; a long one cut into pieces decodes each piece (with the pad) */
    int whole = o.one_utterance && nu == 1;
    long lo = whole ? 0 : (a - pad > 0 ? a - pad : 0);
    long hi = whole ? (long)n : (b + pad < (long)n ? b + pad : (long)n);
    arc_utt_build* u = &R->u[i];
    u->f0 = seg[i][0];
    u->f1 = seg[i][1];
    u->start = (float)(seg[i][0] / 10.0);
    u->end = (float)((seg[i][1] + 1) / 10.0);
    if (decode_segment(e, pcm + lo, (int)(hi - lo), seg[i][0] / 10.0, beams, o.bias_names,
                       o.words || o.names || o.numbers, lo / (double)ARC_SAMPLE_RATE, u)) {
      arc_free(seg);
      arc_result_free(arc_result_finish(R, 0, 0));
      return NULL;
    }
    if (u->text[0]) {   /* history for auto_prev (last 64) */
      if (e->n_hist == 64) {
        arc_free(e->hist[0].text);
        memmove(e->hist, e->hist + 1, sizeof(e->hist[0]) * 63);
        e->n_hist--;
      }
      e->hist[e->n_hist].s = seg[i][0] / 10.0;
      e->hist[e->n_hist].e = (seg[i][1] + 1) / 10.0;
      e->hist[e->n_hist++].text = arc_strdup(u->text);
    }
  }
  arc_free(seg);
  if ((o.trunk || o.events || o.names || o.numbers) && arc_engine_run(e, pcm, n, &o, R)) {
    arc_result_free(arc_result_finish(R, 0, 0));
    return NULL;
  }
  return arc_result_finish(R, (float)((double)n / ARC_SAMPLE_RATE), (float)(arc_time_ms() - t0));
}
