"""The model document: a certified description plus facts from named providers.

describe() stays the authority for structure. Everything the user interface
needed beyond it is added here, and ``origins`` says where each kind of fact
came from so a reader can tell declared structure from tooling and convention.
"""

SCHEMA = "graphion.inspection.document/0"

ORIGINS = {
    "source": "tooling: Python source lookup on the definition",
    "doc": "tooling: the definition's own docstring",
    "nbytes": "derived: resolved shape and dtype in the description",
    "rate": "library: graphion.dsp stream_ports convention",
    "display": "tooling: hint registry keyed by definition",
    "producer": "derived: connection records followed to the direct producer",
}

# Which port a display block shows by default, keyed by definition identity.
# The core has no annotation registry, so the tooling layer keeps this one.
DISPLAY_HINTS = {
    ("graphion.dsp.blocks.constellation_viewer", "ConstellationViewer"): {"port": "points", "view": "constellation"},
    ("examples.blocks.spectrum", "Spectrum"): {"port": "psd", "view": "array"},
}


def address(path, name):
    """Spell the address of a named member of a block."""
    return f"{path}::{name}"


def split_address(text):
    """Return (path, name) for an address."""
    path, _, name = text.rpartition("::")
    return path, name


def producers(description):
    """Map every port address to its direct producer by following connection records.

    A forwarded input or output resolves through the hierarchy to the output
    that computes it. An external input with no source resolves to itself.
    """
    sources = {
        address(c["destination"]["path"], c["destination"]["port"]): address(c["source"]["path"], c["source"]["port"])
        for c in description["connections"]
    }
    out = {}
    for block in description["blocks"]:
        for declaration in block["declarations"]:
            if declaration["kind"] not in ("In", "Out"):
                continue
            start = current = address(block["path"], declaration["name"])
            # The bound keeps a miswired cycle of forwarding records from looping forever.
            for _ in range(len(sources) + 1):
                if current not in sources:
                    break
                current = sources[current]
            out[start] = current
    return out


def compose(description, facts, capabilities, name):
    """Assemble the document served to clients."""
    resolved = producers(description)
    for path, entry in facts.items():
        for port, port_facts in entry["ports"].items():
            port_facts["producer"] = resolved.get(address(path, port))
    return {
        "schema": SCHEMA,
        "model": name,
        "description": description,
        "facts": facts,
        "origins": ORIGINS,
        "capabilities": sorted(capabilities),
    }
