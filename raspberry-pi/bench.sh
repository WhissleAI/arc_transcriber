#!/bin/sh
# Speed, latency, memory and thermals of Arc streams on a Raspberry Pi (or any Linux machine): arc-stream in
# heads-only and full mode at 1, 2 and 4 threads, the SoC temperature, clock and throttling flags before and after
# each run. Nothing is played or recorded.
#
#   raspberry-pi/bench.sh [file.wav ...]       # default: examples/example_synth.wav
#
# Environment: ARC_MODEL (default ~/.local/share/arc/mini), ARC_STREAM (default c/build/arc-stream), PARTIAL_MS
# (full mode, default 1000), COOL (wait until the SoC is below this many degrees C before each run, default 62; 0: off),
# OUT (log directory, default bench-<date>).
set -eu
here=$(cd "$(dirname "$0")" && pwd)
bin=${ARC_STREAM:-$here/../c/build/arc-stream}
model=${ARC_MODEL:-$HOME/.local/share/arc/mini}
[ -x "$bin" ] || { echo "no arc-stream: build first (raspberry-pi/README.md)" >&2; exit 1; }
[ -f "$model/config.json" ] || { echo "no model at $model (set ARC_MODEL)" >&2; exit 1; }
[ $# -gt 0 ] || set -- "$here/../examples/example_synth.wav"
out=${OUT:-bench-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$out"
have_vc=0
command -v vcgencmd >/dev/null 2>&1 && vcgencmd measure_temp >/dev/null 2>&1 && have_vc=1

temp() {
  if [ $have_vc = 1 ]; then vcgencmd measure_temp | sed 's/[^0-9.]//g'
  elif [ -r /sys/class/thermal/thermal_zone0/temp ]; then awk '{printf "%.1f", $1 / 1000}' /sys/class/thermal/thermal_zone0/temp
  else echo n/a; fi
}
# get_throttled bits: 0 under-voltage now, 1 clock capped now, 2 throttled now, 3 soft temperature limit now;
# 16-19 the same "since boot". A row measured while bits 0-3 are set is power- or heat-limited.
throttled() {
  if [ $have_vc = 1 ]; then vcgencmd get_throttled | sed 's/.*=//'; return; fi
  for f in /sys/devices/platform/soc/soc:firmware/get_throttled /sys/devices/platform/soc:firmware/get_throttled; do
    [ -r "$f" ] && { echo "0x$(cat "$f")"; return; }
  done
  echo n/a
}
cool() {
  c=${COOL:-62}
  [ "$c" = 0 ] && return 0
  while t=$(temp) && [ "$t" != n/a ] && [ "${t%.*}" -ge "$c" ]; do sleep 5; done
}

echo "# $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || uname -m), $(uname -r), $(nproc) CPUs; throttled $(throttled)" | tee "$out/info.txt"
printf '%-6s %-7s %7s %8s %9s %9s %9s %9s %10s %s\n' mode threads rtf cpu/s final_ms tags_ms prov_ms rss_mb temp_c throttled
for mode in heads full; do
  for t in 1 2 4; do
    [ "$t" -le "$(nproc)" ] || continue
    cool
    t0=$(temp)
    "$bin" "$model" "$@" --mode $mode --threads $t --partial-ms "${PARTIAL_MS:-1000}" --quiet > "$out/$mode-t$t.jsonl"
    s=$(tail -1 "$out/$mode-t$t.jsonl")
    v() { echo "$s" | sed "s/.*\"$1\": \([-0-9.]*\).*/\1/"; }
    printf '%-6s %-7s %7s %8s %9s %9s %9s %9s %10s %s\n' $mode $t "$(v rtf)" "$(v cpu_per_audio_s)" "$(v final_ms_p50)" \
      "$(v tags_ms_p50)" "$(v prov_tags_ms_p50)" "$(v peak_rss_mb)" "$t0-$(temp)" "$(throttled)" | tee -a "$out/table.txt"
  done
done
echo "logs: $out (one JSON line per file and a summary line per run)"
