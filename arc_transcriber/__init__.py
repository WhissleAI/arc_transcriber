"""arc_transcriber: a command-line transcriber built on Arc v1.1 (arc_audio). The weights download from Hugging Face on first use.

    from arc_transcriber import load
    arc = load()                               # edge backbone: Arc.from_pretrained("WhissleAI/arc-v1.1", revision="v1.1.0")
    arc = load(backbone="parakeet", device="cuda")
"""
__version__ = "1.1.0"
DEFAULT_MODEL = "WhissleAI/arc-v1.1"
DEFAULT_REVISION = "v1.1.0"


def load(model=DEFAULT_MODEL, revision=DEFAULT_REVISION, device="cpu", threads=None, backbone="edge"):
    from arc_audio import Arc
    return Arc.from_pretrained(model, backbone=backbone, revision=revision, device=device, threads=threads)
