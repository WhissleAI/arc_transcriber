#!/bin/sh
# Build and install libarc (Arc Mini in C: library, header, arc-transcribe, arc-stream) on a Raspberry Pi, tuned for
# the board, and fetch the model. Run it from a clone of the repo, on the Pi (64-bit Raspberry Pi OS, Ubuntu, or
# another aarch64 Linux with apt):
#
#   git clone https://github.com/WhissleAI/arc_transcriber && cd arc_transcriber
#   raspberry-pi/install.sh               # build (tuned for this Pi), unit tests, install to ~/.local, fetch Arc Mini
#
# Environment: PREFIX (default ~/.local), MODELS (default $PREFIX/share/arc), CPU (override the detected -mcpu:
# cortex-a76, cortex-a72, native, or "" for portable). The model repo is gated: accept its terms on
# https://huggingface.co/WhissleAI/arc-v1.1 and log in (`hf auth login`, or export HF_TOKEN) first.
# Flags: --no-test, --no-model.
set -eu
test=1 model=1
for a in "$@"; do
  case "$a" in
    --no-test) test=0 ;;
    --no-model) model=0 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done
here=$(cd "$(dirname "$0")" && pwd)
c=$(dirname "$here")/c
PREFIX=${PREFIX:-$HOME/.local}
MODELS=${MODELS:-$PREFIX/share/arc}

board=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo "unknown board")
if [ -z "${CPU+x}" ]; then
  case "$board" in
    *"Raspberry Pi 5"*|*"Raspberry Pi 500"*|*"Compute Module 5"*) CPU=cortex-a76 ;;
    *"Raspberry Pi 4"*|*"Raspberry Pi 400"*|*"Compute Module 4"*) CPU=cortex-a72 ;;
    *) CPU= ;;
  esac
fi
os=$( (. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") || uname -s)
echo "board: $board ($(uname -m), $os); tuning: ${CPU:-portable}; install prefix: $PREFIX"
[ "$(uname -m)" = aarch64 ] || echo "note: not aarch64: a portable build (32-bit Raspberry Pi OS is not a target)"

if command -v vcgencmd >/dev/null 2>&1 && t=$(vcgencmd get_throttled 2>/dev/null | sed 's/.*=//') && [ -n "$t" ]; then
  if [ $((t & 0x10001)) -ne 0 ]; then
    echo "WARNING: under-voltage detected since boot (get_throttled=$t). Use the official supply (Pi 5: 27 W, 5 V 5 A;"
    echo "         Pi 4: 15 W, 5 V 3 A); from a laptop's USB port the Pi throttles under load."
  fi
fi

need=""
for t in cc make; do command -v $t >/dev/null 2>&1 || need="$need $t"; done
if [ -n "$need" ]; then
  if command -v apt-get >/dev/null 2>&1; then sudo apt-get update && sudo apt-get install -y gcc make
  else echo "install$need with your package manager, then run this again" >&2; exit 1; fi
fi

make -C "$c" -j"$(nproc)" CPU="$CPU" >/dev/null
[ $test = 1 ] && make -C "$c" CPU="$CPU" test
make -C "$c" CPU="$CPU" PREFIX="$PREFIX" install >/dev/null
echo "installed: $PREFIX/bin/arc-transcribe, $PREFIX/bin/arc-stream, $PREFIX/lib/libarc.a, $PREFIX/include/arc.h"

if [ $model = 1 ]; then
  if [ -f "$MODELS/mini/config.json" ]; then
    echo "model: $MODELS/mini (already there)"
  elif command -v hf >/dev/null 2>&1; then
    hf download WhissleAI/arc-v1.1 --include "mini/*" --include LICENSE --include NOTICE --local-dir "$MODELS" >/dev/null
    echo "model: $MODELS/mini"
  else
    echo "model: install the Hugging Face CLI (pip install -U huggingface_hub), log in, then run:"
    echo "  hf download WhissleAI/arc-v1.1 --include 'mini/*' --include LICENSE --include NOTICE --local-dir $MODELS"
  fi
  [ $test = 1 ] && [ -f "$MODELS/mini/config.json" ] && "$c/build/test_stream" "$MODELS/mini" "$here/../examples/example_synth.wav"
fi
cat <<EOT

Try it:
  $PREFIX/bin/arc-transcribe $MODELS/mini $here/../examples/example_synth.wav
  $PREFIX/bin/arc-stream $MODELS/mini $here/../examples/example_synth.wav --mode heads
  $here/bench.sh          # speed, latency, memory and thermals on this Pi
  examples: cc $here/examples/stream.c -I$PREFIX/include $PREFIX/lib/libarc.a -lm -pthread -o stream
            arecord -q -f S16_LE -r 16000 -c 1 -t raw | ./stream $MODELS/mini --heads
EOT
