// arc-speech: Apple's on-device SpeechTranscriber (macOS 26 SpeechAnalyzer) as the external transcriber of an Arc
// stream: Arc runs heads-only (no transcript decoder loaded) through the C API, Apple's words (with their audio times)
// go in with arc_stream_words, and Arc's utterances come out with the words, tags, read-out and events.
//
//   arc-speech MINI_DIR file.wav [more.wav ...] [--json] [--tail-s 2] [--record DIR] [--locale en-US] [--no-finalize]
//
// When Arc's segmenter closes an utterance (ARC_OUT_ENDPOINT) the transcriber is asked to finalise up to 0.3 s past its
// end (SpeechAnalyzer.finalize(through:)); --no-finalize leaves it to finalise on its own.
// The WAV is fed in 100 ms buffers at real time (as a microphone would); nothing is played or recorded. For every
// file: Apple's first words and final text, Arc's utterance records, and latencies from the end of speech (from the
// file's levels). --record DIR writes Apple's results as arc-stream --words-dir replay files (DIR/<name>.jsonl), so
// the same transcriber output can be replayed into Arc on another machine (a Raspberry Pi).
// Build: macos/build.sh. Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
import AVFoundation
import Foundation
import Speech

struct Opts {
  var model = ""
  var files: [String] = []
  var json = false
  var tail = 2.0
  var record: String? = nil
  var locale = "en-US"
  var noFinalize = false
}

func parse() -> Opts {
  var o = Opts()
  var a = Array(CommandLine.arguments.dropFirst())
  guard a.count >= 2 else {
    FileHandle.standardError.write("usage: arc-speech MINI_DIR file.wav [...] [--json] [--tail-s 2] [--record DIR] [--locale en-US]\n".data(using: .utf8)!)
    exit(2)
  }
  o.model = a.removeFirst()
  while !a.isEmpty {
    let x = a.removeFirst()
    switch x {
    case "--json": o.json = true
    case "--tail-s": o.tail = Double(a.removeFirst()) ?? 2
    case "--record": o.record = a.removeFirst()
    case "--locale": o.locale = a.removeFirst()
    case "--no-finalize": o.noFinalize = true
    default: o.files.append(x)
    }
  }
  return o
}

/// 16 kHz mono float samples of a WAV (converted if needed).
func readAudio(_ path: String) throws -> [Float] {
  let f = try AVAudioFile(forReading: URL(fileURLWithPath: path))
  let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
  let inBuf = AVAudioPCMBuffer(pcmFormat: f.processingFormat, frameCapacity: AVAudioFrameCount(f.length))!
  try f.read(into: inBuf)
  if f.processingFormat.sampleRate == 16000 && f.processingFormat.channelCount == 1
    && f.processingFormat.commonFormat == .pcmFormatFloat32 {
    return Array(UnsafeBufferPointer(start: inBuf.floatChannelData![0], count: Int(inBuf.frameLength)))
  }
  let conv = AVAudioConverter(from: f.processingFormat, to: fmt)!
  let cap = AVAudioFrameCount(Double(inBuf.frameLength) * 16000 / f.processingFormat.sampleRate) + 1024
  let out = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: cap)!
  var done = false
  _ = conv.convert(to: out, error: nil) { _, st in
    if done { st.pointee = .endOfStream; return nil }
    done = true
    st.pointee = .haveData
    return inBuf
  }
  return Array(UnsafeBufferPointer(start: out.floatChannelData![0], count: Int(out.frameLength)))
}

/// Speech start / end: first / last 100 ms frame within 35 dB of the loudest (and above -55 dBFS), as arc-stream.
func speechBounds(_ x: [Float]) -> (Double, Double) {
  let nf = (x.count + 1599) / 1600
  var db = [Double](repeating: -120, count: nf)
  for f in 0..<nf {
    let lo = f * 1600, hi = min(lo + 1600, x.count)
    var s = 0.0
    for i in lo..<hi { s += Double(x[i]) * Double(x[i]) }
    db[f] = 20 * log10(sqrt(s / Double(hi - lo)) + 1e-6)
  }
  let thr = max((db.max() ?? -120) - 35, -55)
  var a = 0, b = nf - 1
  while a < nf && db[a] < thr { a += 1 }
  while b > a && db[b] < thr { b -= 1 }
  return (Double(a) / 10, Double(b + 1) / 10)
}

/// Apple's result as words with times: runs carrying audioTimeRange, joined into words at spaces.
func words(of text: AttributedString) -> [(String, Double, Double)] {
  var out: [(String, Double, Double)] = []
  var cur = "", t0 = -1.0, t1 = -1.0
  func flush() {
    let w = cur.trimmingCharacters(in: .whitespaces)
    if !w.isEmpty { out.append((w, t0, t1)) }
    cur = ""; t0 = -1; t1 = -1
  }
  for run in text.runs {
    let piece = String(text[run.range].characters)
    let tr = run.audioTimeRange
    for (i, part) in piece.split(separator: " ", omittingEmptySubsequences: false).enumerated() {
      if i > 0 { flush() }
      if part.isEmpty { continue }
      cur += part
      if let tr {
        if t0 < 0 { t0 = tr.start.seconds }
        t1 = tr.end.seconds
      }
    }
  }
  flush()
  return out
}

func now() -> Double { Date().timeIntervalSince1970 * 1000 }

/// CPU seconds of every process (pid -> (name, seconds)), to see what Apple's recogniser costs outside this process.
func cpuTimes() -> [Int32: (String, Double)] {
  let p = Process()
  p.executableURL = URL(fileURLWithPath: "/bin/ps")
  p.arguments = ["-A", "-o", "pid=,time=,comm="]
  let pipe = Pipe()
  p.standardOutput = pipe
  try? p.run()
  let data = pipe.fileHandleForReading.readDataToEndOfFile()
  p.waitUntilExit()
  var r: [Int32: (String, Double)] = [:]
  for line in String(decoding: data, as: UTF8.self).split(separator: "\n") {
    let f = line.split(separator: " ", maxSplits: 2, omittingEmptySubsequences: true)
    guard f.count == 3, let pid = Int32(f[0]) else { continue }
    var secs = 0.0
    for (i, c) in f[1].split(separator: ":").reversed().enumerated() { secs += (Double(c) ?? 0) * pow(60, Double(i)) }
    r[pid] = (String(f[2]), secs)
  }
  return r
}

/// What one file's run collects (touched only on `q`).
final class Run: @unchecked Sendable {
  var t0 = 0.0, speechStart = 0.0, speechEnd = 0.0
  var appleFirst = -1.0, appleFinal = -1.0, arcFinal = -1.0, arcTags = -1.0
  var appleText: [String] = []
  var arcText: [String] = []
  var log: [String] = []
  var record: [String] = []
  var json = false
  var analyzer: SpeechAnalyzer? = nil
  var endpoints = 0
  func wallAt(_ audio: Double) -> Double { t0 + audio * 1000 }
}

let q = DispatchQueue(label: "arc")   // every arc_* call happens here
let o = parse()

var err = [CChar](repeating: 0, count: 512)
guard let engine = arc_create_ex(o.model, 2, Int32(ARC_NO_DECODER), &err, err.count) else {
  print("arc: \(String(cString: err))"); exit(1)
}

func esc(_ s: String) -> String {
  let d = try! JSONSerialization.data(withJSONObject: [s], options: [])
  return String(String(decoding: d, as: UTF8.self).dropFirst().dropLast())
}

let onOutput: arc_stream_fn = { out, user in
  guard let out, let user else { return }
  let r = Unmanaged<Run>.fromOpaque(user).takeUnretainedValue()
  let o = out.pointee
  let t = now()
  if o.kind == ARC_OUT_ENDPOINT, let a = r.analyzer, let u = o.utt?.pointee {
    // Arc's segmenter closed an utterance: ask the transcriber for its final words up to there (not from inside this
    // callback: it runs on the Arc queue, and the results arrive through it)
    r.endpoints += 1
    let through = CMTime(seconds: Double(u.end) + 0.3, preferredTimescale: 1000)
    Task { try? await a.finalize(through: through) }
    if !r.json { r.log.append(String(format: "  arc endpoint #%d %.2f-%.2f %5.0f ms", o.id, u.start, u.end, t - r.wallAt(r.speechEnd))) }
    return
  }
  guard let u = o.utt?.pointee, o.kind == ARC_OUT_FINAL || o.kind == ARC_OUT_TAGS else { return }
  let text = String(cString: u.text)
  let lat = t - r.wallAt(r.speechEnd)
  if o.kind == ARC_OUT_FINAL {
    if !text.isEmpty { r.arcText.append(text) }
    if Double(u.end) >= r.speechEnd - 0.3 || r.arcFinal < 0 { r.arcFinal = lat }
  }
  if (o.kind == ARC_OUT_TAGS || o.tags_final != 0) && (Double(u.end) >= r.speechEnd - 0.3 || r.arcTags < 0) { r.arcTags = lat }
  let intent = u.intent.label.map { String(cString: $0) } ?? "-"
  let act = u.act.label.map { String(cString: $0) } ?? "-"
  if r.json {
    r.log.append("{\"kind\": \"\(o.kind == ARC_OUT_FINAL ? "final" : "tags")\", \"id\": \(o.id), \"start\": \(u.start), \"end\": \(u.end), \"text\": \(esc(text)), \"device_directed\": \(u.device_directed), \"directed_head\": \(u.directed_head), \"intent\": \(esc(intent)), \"act\": \(esc(act)), \"tags_final\": \(o.tags_final), \"latency_ms\": \(Int(lat))}")
  } else {
    r.log.append(String(format: "  arc %@ #%d %.2f-%.2f %5.0f ms  %@  [dd %.2f, %@, %@%@]", o.kind == ARC_OUT_FINAL ? "final" : "tags ",
                        o.id, u.start, u.end, lat, text.isEmpty ? "(no text)" : text, u.device_directed, intent, act,
                        o.kind == ARC_OUT_FINAL && o.tags_final == 0 ? ", tags pending" : ""))
  }
}

func runFile(_ path: String) async throws -> Run {
  let x = try readAudio(path)
  let r = Run()
  r.json = o.json
  (r.speechStart, r.speechEnd) = speechBounds(x)
  let tail = Int(o.tail * 16000)
  let all = x + [Float](repeating: 0, count: tail)

  let locale = Locale(identifier: o.locale)
  let tr = SpeechTranscriber(locale: locale, transcriptionOptions: [], reportingOptions: [.volatileResults, .fastResults],
                             attributeOptions: [.audioTimeRange])
  guard let fmt = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [tr]) else { throw NSError(domain: "no format", code: 1) }
  let (inputs, cont) = AsyncStream<AnalyzerInput>.makeStream()
  let analyzer = SpeechAnalyzer(modules: [tr])
  try await analyzer.prepareToAnalyze(in: fmt)
  if !o.noFinalize { r.analyzer = analyzer }

  var cfg = arc_stream_config()
  cfg.mode = ARC_STREAM_EXTERNAL
  cfg.on_output = onOutput
  cfg.user = Unmanaged.passUnretained(r).toOpaque()
  let stream: OpaquePointer = q.sync { arc_stream_open(engine, &cfg) }

  // Apple's results into Arc as they come
  let results = Task {
    for try await res in tr.results {
      let t = now()
      let ws = words(of: res.text)
      let plain = ws.map { $0.0 }.joined(separator: " ")
      q.sync {
        if !plain.isEmpty && r.appleFirst < 0 { r.appleFirst = t - r.wallAt(r.speechStart) }
        if res.isFinal {
          if !plain.isEmpty { r.appleText.append(plain) }
          if res.range.end.seconds >= r.speechEnd - 0.3 && !plain.isEmpty { r.appleFinal = t - r.wallAt(r.speechEnd) }
        }
        let audioAt = (t - r.t0) / 1000
        if o.record != nil {
          let wj = ws.map { "{\"text\": \(esc($0.0)), \"start\": \(String(format: "%.3f", $0.1)), \"end\": \(String(format: "%.3f", $0.2))}" }.joined(separator: ", ")
          r.record.append("{\"at\": \(String(format: "%.3f", audioAt)), \"final\": \(res.isFinal ? 1 : 0), \"words\": [\(wj)]}")
        }
        var cw = ws.map { strdup($0.0) }
        var aw = (0..<ws.count).map { arc_ext_word(text: cw[$0], start: Float(ws[$0].1), end: Float(ws[$0].2)) }
        arc_stream_words(stream, &aw, Int32(aw.count), res.isFinal ? 1 : 0)
        for p in cw { free(p) }
        cw.removeAll()
        if !r.json {
          r.log.append(String(format: "  apple %@ %.2f-%.2f %5.0f ms  %@", res.isFinal ? "final" : "words", res.range.start.seconds,
                              res.range.end.seconds, t - r.wallAt(res.isFinal ? r.speechEnd : r.speechStart), plain))
        }
      }
    }
  }
  try await analyzer.start(inputSequence: inputs)

  // 100 ms buffers at real time: Arc and Apple get each one at the same moment
  let step = 1600
  r.t0 = now()
  var pos = 0
  while pos < all.count {
    let k = min(step, all.count - pos)
    let due = r.t0 + Double(pos + k) / 16.0
    let wait = due - now()
    if wait > 0 { try await Task.sleep(nanoseconds: UInt64(wait * 1e6)) }
    let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(k))!
    buf.frameLength = AVAudioFrameCount(k)
    if let i16 = buf.int16ChannelData {
      for i in 0..<k { i16[0][i] = Int16(max(-1, min(1, all[pos + i])) * 32767) }
    } else if let f32 = buf.floatChannelData {
      for i in 0..<k { f32[0][i] = all[pos + i] }
    }
    all.withUnsafeBufferPointer { p in q.sync { _ = arc_stream_push(stream, p.baseAddress! + pos, k) } }
    cont.yield(AnalyzerInput(buffer: buf))
    pos += k
  }
  cont.finish()
  try await analyzer.finalizeAndFinishThroughEndOfInput()
  _ = try await results.value
  q.sync {
    arc_stream_finish(stream)
    arc_stream_close(stream)
  }
  if let dir = o.record {
    try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    let name = ((path as NSString).lastPathComponent as NSString).deletingPathExtension
    try? (r.record.joined(separator: "\n") + "\n").write(toFile: "\(dir)/\(name).jsonl", atomically: true, encoding: .utf8)
  }
  return r
}

let sem = DispatchSemaphore(value: 0)
Task {
  let c0 = cpuTimes()
  var ru0 = rusage(); getrusage(RUSAGE_SELF, &ru0)
  var audio = 0.0
  var af: [Double] = [], afin: [Double] = [], arcf: [Double] = [], arct: [Double] = []
  for f in o.files {
    do {
      let r = try await runFile(f)
      audio += Double((try? AVAudioFile(forReading: URL(fileURLWithPath: f)).length) ?? 0) / 16000
      if !o.json { print(f); r.log.forEach { print($0) } }
      if r.appleFirst >= 0 { af.append(r.appleFirst) }
      if r.appleFinal >= 0 { afin.append(r.appleFinal) }
      if r.arcFinal >= 0 { arcf.append(r.arcFinal) }
      if r.arcTags >= 0 { arct.append(r.arcTags) }
      print("{\"file\": \(esc(f)), \"speech\": [\(r.speechStart), \(r.speechEnd)], \"apple_first_ms\": \(Int(r.appleFirst)), \"apple_final_ms\": \(Int(r.appleFinal)), \"arc_final_ms\": \(Int(r.arcFinal)), \"arc_tags_ms\": \(Int(r.arcTags)), \"apple_text\": \(esc(r.appleText.joined(separator: " "))), \"text\": \(esc(r.arcText.joined(separator: " ")))}")
    } catch { print("{\"file\": \(esc(f)), \"error\": \(esc("\(error)"))}") }
  }
  var ru = rusage(); getrusage(RUSAGE_SELF, &ru)
  let mine = Double(ru.ru_utime.tv_sec - ru0.ru_utime.tv_sec + ru.ru_stime.tv_sec - ru0.ru_stime.tv_sec)
    + Double(ru.ru_utime.tv_usec - ru0.ru_utime.tv_usec + ru.ru_stime.tv_usec - ru0.ru_stime.tv_usec) / 1e6
  let c1 = cpuTimes()
  var others: [(String, Double)] = []
  for (pid, v) in c1 where pid != getpid() {
    let d = v.1 - (c0[pid]?.1 ?? 0)
    if d > 0.5 { others.append((v.0, d)) }
  }
  others.sort { $0.1 > $1.1 }
  func p50(_ a: [Double]) -> Int { a.isEmpty ? -1 : Int(a.sorted()[a.count / 2]) }
  func p90(_ a: [Double]) -> Int { a.isEmpty ? -1 : Int(a.sorted()[min(a.count - 1, Int(Double(a.count) * 0.9))]) }
  let top = others.prefix(6).map { "{\"process\": \(esc(($0.0 as NSString).lastPathComponent)), \"cpu_s\": \(String(format: "%.1f", $0.1))}" }.joined(separator: ", ")
  print("{\"summary\": true, \"files\": \(o.files.count), \"audio_s\": \(String(format: "%.1f", audio)), \"apple_first_ms_p50\": \(p50(af)), \"apple_first_ms_p90\": \(p90(af)), \"apple_final_ms_p50\": \(p50(afin)), \"apple_final_ms_p90\": \(p90(afin)), \"arc_final_ms_p50\": \(p50(arcf)), \"arc_final_ms_p90\": \(p90(arcf)), \"arc_tags_ms_p50\": \(p50(arct)), \"arc_tags_ms_p90\": \(p90(arct)), \"cpu_this_process_per_audio_s\": \(String(format: "%.4f", mine / max(audio, 1e-9))), \"peak_rss_mb\": \(String(format: "%.1f", Double(ru.ru_maxrss) / 1048576)), \"other_processes_cpu\": [\(top)]}")
  sem.signal()
}
sem.wait()
arc_destroy(engine)
