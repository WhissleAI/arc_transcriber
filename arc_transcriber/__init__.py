"""arc_transcriber: a command-line transcriber built on Arc (arc_audio). The weights download from Hugging Face on first use.

    from arc_transcriber import load
    arc = load()                               # Arc Mini: Arc.from_pretrained("WhissleAI/arc-v1.1", backbone="mini", revision="v1.2.0")
    arc = load(backbone="full", device="cuda") # Arc Transcriber (full, Parakeet encoder)
"""
__version__ = "1.2.0"
DEFAULT_MODEL = "WhissleAI/arc-v1.1"
DEFAULT_REVISION = "v1.2.0"


def load(model=DEFAULT_MODEL, revision=DEFAULT_REVISION, device="cpu", threads=None, backbone="mini"):
    from arc_audio import Arc
    return Arc.from_pretrained(model, backbone=backbone, revision=revision, device=device, threads=threads)
