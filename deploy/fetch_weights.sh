#!/usr/bin/env bash
# Fetch the PUBLIC weights an image bundles into deploy/weights/ (git-ignored), checked against the model repos' checksums.
# Run at build time only: the server itself never downloads anything.
#
#   HF_TOKEN=hf_... deploy/fetch_weights.sh [mini] [tagger]      (default: mini)
#
#   mini    WhissleAI/arc-v1.1, folder mini/  -> deploy/weights/mini        (68 MB; the repo is gated with automatic
#           approval: accept its terms once on huggingface.co and use a read token)
#   tagger  WhissleAI/arc-tagger @ v0.4.1, onnx/ -> deploy/weights/arc-tagger (for Dockerfile.combined; public)
# Env: ARC_REVISION (default main; pin a commit for reproducible images), ARCTAG_REVISION (v0.4.1), DEST (deploy/weights).
# Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0 with the Whissle terms; see LICENSE.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${DEST:-$HERE/weights}"
ARC_REV="${ARC_REVISION:-main}"
TAG_REV="${ARCTAG_REVISION:-v0.4.1}"
WHAT=("$@")
[ ${#WHAT[@]} -eq 0 ] && WHAT=(mini)
AUTH=()
[ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")
sha() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
get() {  # get REPO REV PATH OUT
  mkdir -p "$(dirname "$4")"
  curl -fsSL --retry 3 ${AUTH[@]+"${AUTH[@]}"} -o "$4.part" "https://huggingface.co/$1/resolve/$2/$3" && mv "$4.part" "$4"
}

for w in "${WHAT[@]}"; do
  case "$w" in
    mini)
      out="$DEST/mini"; mkdir -p "$out"
      get WhissleAI/arc-v1.1 "$ARC_REV" MANIFEST.sha256 "$out/.manifest" || {
        echo "cannot read WhissleAI/arc-v1.1 (gated): accept its terms on huggingface.co and set HF_TOKEN" >&2; exit 1; }
      grep ' \./mini/' "$out/.manifest" | while read -r sum path; do
        rel="${path#./mini/}"
        echo "fetching mini/$rel"
        get WhissleAI/arc-v1.1 "$ARC_REV" "mini/$rel" "$out/$rel"
        echo "$sum  $rel" >> "$out/.sums"
      done
      (cd "$out" && sha -c .sums && rm -f .sums .manifest)
      ;;
    tagger)
      out="$DEST/arc-tagger"; mkdir -p "$out"
      for f in SHA256SUMS arctag_meta.json tokenizer.json arctag.onnx arctag_int8.onnx; do
        echo "fetching arc-tagger onnx/$f"
        get WhissleAI/arc-tagger "$TAG_REV" "onnx/$f" "$out/$f"
      done
      (cd "$out" && sha -c SHA256SUMS)
      ;;
    *) echo "unknown weights '$w' (mini | tagger)" >&2; exit 2;;
  esac
done
du -sh "$DEST"/* 2>/dev/null || true
