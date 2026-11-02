"""Cloud streaming recognisers as the word source of Arc's EXTERNAL stream mode (Arc's trunk and heads give the tags,
directedness, end of turn and acts; the provider gives the words).

    python -m arc_audio.integrations.deepgram clip.wav       # DEEPGRAM_API_KEY
    python -m arc_audio.integrations.assemblyai clip.wav     # ASSEMBLYAI_API_KEY
    python -m arc_audio.integrations.speechmatics clip.wav   # SPEECHMATICS_API_KEY

Needs `pip install "arc-transcriber[integrations]"` (websockets). See README.md in this folder.
"""
from .base import MissingKey, Options, Parsed, ProviderError, StreamingASR, replay, scrub
from .registry import PROVIDERS, get
from .session import Session, run_file, wire

__all__ = ["Options", "Parsed", "ProviderError", "MissingKey", "StreamingASR", "Session", "run_file", "wire", "replay",
           "scrub", "PROVIDERS", "get"]
