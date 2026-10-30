/* The Moonshine tokenizer (tokenizer.json): SentencePiece-style BPE with byte fallback, 32,000 pieces plus 768 added
 * special tokens (Arc's inline tags 32600..32635 and context markers 32640..). Encode as HF tokenizers does with
 * add_special_tokens=False; decode with skip_special_tokens=True.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#ifndef ARC_TOK_H
#define ARC_TOK_H

typedef struct arc_tok arc_tok;

arc_tok* arc_tok_load(const char* path);   /* NULL + arc_seterr */
void arc_tok_free(arc_tok* t);

/* Text -> ids (no BOS). Returns the count (may exceed cap: then only cap are written), -1 on error. */
int arc_tok_encode(const arc_tok* t, const char* text, int* ids, int cap);
/* Ids -> text, special tokens skipped, then trimmed (as Python's .strip()). Returns a malloc'd string (arc_free). */
char* arc_tok_decode(const arc_tok* t, const int* ids, int n);
/* One token's piece as stored ("▁Say", "<0x0A>", "<<END>>"), or NULL. */
const char* arc_tok_piece(const arc_tok* t, int id);
int arc_tok_is_special(const arc_tok* t, int id);
int arc_tok_size(const arc_tok* t);

#endif
