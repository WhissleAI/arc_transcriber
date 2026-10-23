"""arc_transcriber: a command-line transcriber built on Arc v1 (arc_audio). The weights download from Hugging Face on first use.

    from arc_transcriber import load
    arc = load()                               # Arc.from_pretrained("WhissleAI/arc-v1", revision="v1.0.1")
"""
__version__ = "1.0.1"
DEFAULT_MODEL = "WhissleAI/arc-v1"
DEFAULT_REVISION = "v1.0.1"


def load(model=DEFAULT_MODEL, revision=DEFAULT_REVISION, device="cpu", threads=None):
    from arc_audio import Arc
    return Arc.from_pretrained(model, revision=revision, device=device, threads=threads)
