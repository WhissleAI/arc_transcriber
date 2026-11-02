"""provider name -> adapter class. Adding a provider: subclass base.StreamingASR in a new module and list it here."""
import importlib

PROVIDERS = {
    "deepgram": ("deepgram", "Deepgram"),
    "assemblyai": ("assemblyai", "AssemblyAI"),
    "speechmatics": ("speechmatics", "Speechmatics"),
}


def get(name):
    if isinstance(name, type):
        return name
    try:
        mod, cls = PROVIDERS[name.lower()]
    except KeyError:
        raise ValueError(f"unknown provider {name!r}: one of {', '.join(PROVIDERS)}") from None
    return getattr(importlib.import_module(f"{__package__}.{mod}"), cls)
