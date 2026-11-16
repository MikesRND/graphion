"""A host-side diagram of the running BER model, drawn from describe()."""

from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np

if TYPE_CHECKING:
    from matplotlib.backend_bases import FigureManagerBase


class BerDiagram:
    """Display the BER composite's recorded blocks, ports and connections.

    Placement is specific to BerTestbench; labels and edges come from its
    structural description. This is a topology view, not an execution trace.
    Inspecting/updating the figure never advances the model. The optional
    constellation viewer is placed beside the block it taps.
    """

    POSITIONS: ClassVar[dict[str, tuple[float, float]]] = {
        "root/prbs": (0, 3.2),
        "root/mapper": (3.5, 3.2),
        "root/tx": (7, 3.2),
        "root/chan": (10.5, 3.2),
        "root/rx": (10.5, 0),
        "root/demap": (7, 0),
        "root/counter": (3.5, 0),
        "root/delay": (0, 0),
    }
    OPTIONAL_POSITIONS: ClassVar[dict[str, tuple[float, float]]] = {"root/viewer": (14, 0)}

    def __init__(self, *, headless=False):
        self._headless = headless
        self._closed = False
        self._model = None
        self.description = None
        if headless:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure

            self.figure = Figure(figsize=(15, 6.5))
            FigureCanvasAgg(self.figure)
        else:
            import matplotlib.pyplot as plt

            if plt.get_backend().lower() in ("agg", "svg", "pdf", "ps", "pgf", "template", "cairo"):
                raise RuntimeError("--diagram needs a GUI Matplotlib backend and a graphical desktop")
            self.figure = plt.figure(figsize=(15, 6.5))
            manager = cast("FigureManagerBase", self.figure.canvas.manager)
            manager.set_window_title("Graphion · BER model")
        self.axes = self.figure.add_axes((0.02, 0.14, 0.96, 0.76))
        self.title = self.figure.suptitle("BerTestbench · model diagram", fontsize=16)
        self.status = self.figure.text(0.5, 0.06, "", ha="center", fontsize=11)
        self.figure.canvas.mpl_connect("close_event", self._on_close)

    def _on_close(self, event):
        self._closed = True

    @staticmethod
    def _label(record):
        name = record["path"].split("/")[-1]
        config = []
        for key in ("order", "factor", "phase", "delay", "depth", "skip", "snr_db"):
            value = (record["parameters"] | record["controls"]).get(key)
            if value is not None and value["kind"] == "literal":
                literal = value["value"]
                config.append(f"{key}={literal:.2f}" if isinstance(literal, float) else f"{key}={literal}")
        outputs = []
        for port in record["declarations"]:
            if port["kind"] == "Out":
                shape = " × ".join(str(axis["value"]) for axis in port["shape"])
                dtype = np.dtype(port["dtype"]["value"]).name
                outputs.append(f"{port['name']}: {dtype} [{shape}]")
        return name, record["definition"]["qualname"], ", ".join(config), "\n".join(outputs)

    def _draw_model(self, model):
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

        description = model.describe()
        if description["schema_version"] != 2 or description["topology"] != "complete":
            raise ValueError("BER diagram needs a complete schema-v2 model description")
        blocks = description["blocks"][1:]
        paths = {block["path"] for block in blocks}
        optional = paths - self.POSITIONS.keys()
        if (
            not self.POSITIONS.keys() <= paths
            or not optional <= self.OPTIONAL_POSITIONS.keys()
            or any(b["context"] is None for b in blocks)
        ):
            raise ValueError("BER diagram needs a configured BerTestbench")
        positions = self.POSITIONS | {path: self.OPTIONAL_POSITIONS[path] for path in optional}
        self.description = description
        self.axes.clear()
        self.axes.set(xlim=(-1.6, max(x for x, _ in positions.values()) + 1.6), ylim=(-1.2, 4.4), aspect="equal")
        self.axes.set_axis_off()
        for record in blocks:
            x, y = positions[record["path"]]
            reference = record["path"] == "root/delay"
            self.axes.add_patch(
                FancyBboxPatch(
                    (x - 1.35, y - 0.85),
                    2.7,
                    1.7,
                    boxstyle="round,pad=0.02",
                    facecolor="#fff5e6" if reference else "#edf4fb",
                    edgecolor="#b7791f" if reference else "#35678d",
                    linewidth=1.3,
                )
            )
            name, kind, config, outputs = self._label(record)
            self.axes.text(x, y + 0.57, name, ha="center", va="center", weight="bold", fontsize=12)
            self.axes.text(x, y + 0.28, kind, ha="center", va="center", fontsize=10)
            self.axes.text(x, y + 0.02, config, ha="center", va="center", fontsize=9, color="#435467")
            self.axes.text(x, y - 0.42, outputs, ha="center", va="center", fontsize=8.5, linespacing=1.4)
        for edge in description["connections"]:
            source, destination = edge["source"], edge["destination"]
            x1, y1 = positions[source["path"]]
            x2, y2 = positions[destination["path"]]
            horizontal = y1 == y2
            dx = np.sign(x2 - x1) * 1.38 if horizontal else 0
            dy = np.sign(y2 - y1) * 0.88 if not horizontal else 0
            reference = "root/delay" in (source["path"], destination["path"])
            color = "#b7791f" if reference else "#35678d"
            self.axes.add_patch(
                FancyArrowPatch(
                    (x1 + dx, y1 + dy),
                    (x2 - dx, y2 - dy),
                    arrowstyle="-|>",
                    mutation_scale=14,
                    linewidth=1.4,
                    color=color,
                )
            )
            self.axes.text(
                (x1 + x2) / 2 + (0 if horizontal else 0.12),
                (y1 + y2) / 2 + (0.16 if horizontal else 0),
                f"{source['port']} → {destination['port']}",
                fontsize=9,
                color=color,
                ha="center" if horizontal else "left",
                va="bottom" if horizontal else "center",
            )
        self._model = model

    def update(self, model, modulation, frame, frames, *, service_events=True):
        """Show the actual point's model and completed-frame count; service GUI events.

        A caller whose own GUI event loop is already running passes
        service_events=False, so that updating only requests a redraw.
        """
        if self._closed:
            return
        if model is not self._model:
            self._draw_model(model)
            self.title.set_text(f"BerTestbench · {modulation} · Eb/N0 {model.ebn0_db:g} dB")
        state = "Point complete" if frame == frames else "Running"
        self.status.set_text(f"{state} · frame {frame:,} / {frames:,} · {frame / frames:.0%}")
        self.figure.canvas.draw_idle()
        if service_events and not self._headless:
            import matplotlib.pyplot as plt

            plt.show(block=False)
            self.figure.canvas.flush_events()

    def finish(self):
        if not self._closed:
            self.status.set_text("Sweep complete · model shown for the final point")
            self.figure.canvas.draw_idle()

    def save(self, path):
        self.figure.savefig(path, dpi=160)

    def close(self):
        if not self._headless:
            import matplotlib.pyplot as plt

            plt.close(self.figure)
        self._closed = True
