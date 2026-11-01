#!/bin/sh
# Build libarc and arc-speech (Apple SpeechTranscriber + Arc heads) for this Mac: macos/build/arc-speech.
# Needs macOS 26 and the Xcode 26 command-line tools.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
make -C "$here/../c" -j"$(sysctl -n hw.ncpu)" build/libarc.a >/dev/null
mkdir -p "$here/build"
swiftc -O -import-objc-header "$here/../c/include/arc.h" "$here/arc_speech.swift" "$here/../c/build/libarc.a" \
  -o "$here/build/arc-speech"
echo "built $here/build/arc-speech"
