/* No NFT2 backend in this build (make without ARC_ORT): arc_nft2_create fails with this message. A native C backend
 * would replace this file. Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include "arc_nft2_be.h"

const char* arc_nft2_be_name(void) { return "none"; }

arc_nft2_be* arc_nft2_be_open(const char* onnx_dir, const arc_nft2_dims* dims, int tap, int threads) {
  (void)onnx_dir;
  (void)dims;
  (void)tap;
  (void)threads;
  arc_seterr("libarc was built without an NFT2 encoder backend: rebuild with make ARC_ORT=/path/to/onnxruntime");
  return NULL;
}
void arc_nft2_be_free(arc_nft2_be* b) { (void)b; }
arc_nft2_cache* arc_nft2_be_cache_new(arc_nft2_be* b) {
  (void)b;
  return NULL;
}
void arc_nft2_be_cache_free(arc_nft2_cache* c) { (void)c; }
int arc_nft2_be_chunk(arc_nft2_be* b, arc_nft2_cache* c, int first, const float* x, int n, float* out, float* tap,
                      int cap, int* k) {
  (void)b, (void)c, (void)first, (void)x, (void)n, (void)out, (void)tap, (void)cap, (void)k;
  return -1;
}
int arc_nft2_be_joint_enc(arc_nft2_be* b, const float* H, int T, float* F) {
  (void)b, (void)H, (void)T, (void)F;
  return -1;
}
int arc_nft2_be_pred(arc_nft2_be* b, int y, float* hc, float* g) {
  (void)b, (void)y, (void)hc, (void)g;
  return -1;
}
int arc_nft2_be_joint(arc_nft2_be* b, const float* F, int n, const float* g, float* logits) {
  (void)b, (void)F, (void)n, (void)g, (void)logits;
  return -1;
}
