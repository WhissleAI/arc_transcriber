"""Arc EXTERNAL stream with a cloud recogniser's words: a file streamed at real-time pace, printed live.

    pip install "arc-transcriber[integrations] @ git+https://github.com/WhissleAI/arc_transcriber"
    export DEEPGRAM_API_KEY=...            # or ASSEMBLYAI_API_KEY / SPEECHMATICS_API_KEY
    python examples/cloud_asr.py deepgram examples/example_synth.wav --domain nba

The one-line version of the same thing is `python -m arc_audio.integrations.deepgram FILE`.
"""
import argparse
import asyncio

from arc_audio import Arc, load_audio
from arc_audio.integrations import Session
from arc_audio.integrations.session import file_chunks


async def main(a):
    arc = Arc.from_pretrained("WhissleAI/arc-v1.1", backbone="mini", device="cpu", threads=4)
    wav = load_audio(a.audio)
    async with Session(arc, a.provider, domain=a.domain, language=a.language) as s:
        async for chunk in file_chunks(wav):                   # a microphone or a call works the same way
            new = await s.feed(chunk)
            for u in new["transcript"]:                        # an utterance is final: provider words + Arc's read-out
                print(f"[{u['start']:6.2f}] {u['text']}")
            for u in new["utterances"]:
                dd = u["tags"].get("discourse.assistant.device_directed")
                act = (u["groups"].get("discourse.act") or {}).get("label")
                print(f"         arc: directed {'-' if dd is None else f'{dd:.2f}'}, act {act}")
            for e in new["events"]:
                if "command_complete" in str(e.get("tag", "")):
                    print(f"         arc: end of turn at {e.get('t')} s")
        out = await s.finish()
    print(out["asr"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("provider", choices=["deepgram", "assemblyai", "speechmatics"])
    ap.add_argument("audio")
    ap.add_argument("--domain", default="unknown")
    ap.add_argument("--language", default="en")
    asyncio.run(main(ap.parse_args()))
