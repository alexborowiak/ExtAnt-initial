"""Module introspection and small shared helpers.

The interesting thing here is `list_plots`, which reads a module and reports
every plot in it with a one-line summary, so you don't have to remember what
you wrote six months ago.
"""

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

LEVELS = ("axes", "figure")


# --------------------------------------------------------------------------
# Marking plots
# --------------------------------------------------------------------------


def plot(level="figure"):
    """Mark a function as a plot, so `list_plots` finds it.

    Marking is explicit rather than guessed from the name, because a name-based
    rule ("starts with plot_") both misses things and picks up helpers, and it
    goes stale silently. One decorator line per function is cheaper than
    debugging why a figure never appears in the index.

    Parameters
    ----------
    level : str
        'axes' for a function that draws on an axes you pass in, 'figure' for
        one that builds and returns its own figure. Recorded so the listing can
        tell you how to call it. Usable bare (``@plot``) for the default.

    Returns
    -------
    callable
        The function, unchanged apart from a `_plot_level` attribute.
    """
    if callable(level):
        level._plot_level = "figure"
        return level

    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")

    def decorate(fn):
        fn._plot_level = level
        return fn

    return decorate


# --------------------------------------------------------------------------
# Reading a module
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PlotInfo:
    """One row of a plot index."""

    name: str
    module: str
    level: str
    summary: str

    def __str__(self):
        return f"{self.name} ({self.level}) — {self.summary}"


def summary_line(obj):
    """First paragraph of a docstring, collapsed to a single line.

    Uses the first *paragraph* rather than splitting on '.', because a full
    stop is not a reliable sentence boundary in these docstrings: 'e.g.',
    '0.5' and 'p10 - p90' all contain one. The numpydoc convention this
    codebase follows puts a one-line summary first and a blank line after it,
    so the paragraph is the summary.

    `inspect.getdoc` rather than `obj.__doc__`, because it dedents the text and
    falls back to an inherited docstring when a method has none of its own.
    """
    doc = inspect.getdoc(obj)
    if not doc:
        return ""
    return " ".join(doc.split("\n\n")[0].split())


def list_plots(module, level=None, marked_only=None):
    """Every plot defined in a module, with its one-line summary.

    Parameters
    ----------
    module : module or str
        A module object, or a dotted name to import.
    level : str or None
        Restrict to 'axes' or 'figure' plots.
    marked_only : bool or None
        Only report functions carrying the `@plot` decorator. The default,
        None, means: use the marks if the module has any, otherwise fall back
        to every public function in it. That way a module nobody has decorated
        yet still lists something useful.

    Returns
    -------
    list of PlotInfo
        Sorted by name.

    Notes
    -----
    Only functions *defined* in the module are reported. Without that filter
    the index for a figures module would also list everything it imported,
    which is how these listings usually end up untrustworthy.
    """
    if isinstance(module, str):
        module = importlib.import_module(module)

    functions = [
        fn
        for _, fn in inspect.getmembers(module, inspect.isfunction)
        if fn.__module__ == module.__name__ and not fn.__name__.startswith("_")
    ]

    if marked_only is None:
        marked_only = any(hasattr(fn, "_plot_level") for fn in functions)

    infos = []
    for fn in functions:
        fn_level = getattr(fn, "_plot_level", None)
        if marked_only and fn_level is None:
            continue
        if level is not None and fn_level != level:
            continue
        infos.append(
            PlotInfo(
                name=fn.__name__,
                module=module.__name__,
                level=fn_level or "?",
                summary=summary_line(fn),
            )
        )

    return sorted(infos, key=lambda info: info.name)


def plot_index(*modules, level=None, width=100):
    """Formatted index of the plots in one or more modules.

    Called with no arguments, indexes every drawing module in this package that
    imports cleanly, skipping any whose dependencies are missing.

    Returns
    -------
    str
    """
    if not modules:
        modules = []
        for name in ("maps", "timeseries", "distributions", "figures"):
            try:
                modules.append(importlib.import_module(f".{name}", __package__))
            except ImportError:
                continue

    lines = []
    for module in modules:
        infos = list_plots(module, level=level)
        if not infos:
            continue
        name = module if isinstance(module, str) else module.__name__
        lines += [name, "-" * len(name)]
        pad = max(len(info.name) for info in infos)
        for info in infos:
            head = f"  {info.name:<{pad}}  {info.level:<6}  "
            lines.append(head + _wrap(info.summary, width - len(head), len(head)))
        lines.append("")

    return "\n".join(lines).rstrip()


def _wrap(text, width, indent):
    words, lines, current = text.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    lines.append(current)
    return ("\n" + " " * indent).join(lines)


def print_plots(*modules, level=None, width=100):
    """Print `plot_index` to stdout."""
    print(plot_index(*modules, level=level, width=width))


# --------------------------------------------------------------------------
# Small shared helpers, used by more than one drawing module
# --------------------------------------------------------------------------


def period_label(period):
    """Human label for a year selection, which may be a slice or a scalar."""
    if isinstance(period, slice):
        return f"{period.start}–{period.stop}"
    return str(period)


def shared_ylim(arrays, pad=0.05, include=()):
    """Y limits spanning every array, so buildup frames line up exactly.

    Compute this once from all the series and reuse it for every frame. Letting
    each frame autoscale is what makes a buildup jump around on screen. It can
    also change the y tick label widths, which under constrained layout moves
    the axes as well, though that depends on the numbers.

    Parameters
    ----------
    arrays : sequence of array_like
        All the series that will ever appear, revealed or not.
    pad : float
        Margin above and below, as a fraction of the data range.
    include : array_like
        Extra values that must stay inside the limits, e.g. a baseline drawn as
        a horizontal line rather than a series.

    Returns
    -------
    (float, float)
    """
    parts = [np.asarray(a).ravel() for a in arrays]
    if len(np.asarray(include).ravel()):
        parts.append(np.asarray(include, dtype=float).ravel())

    values = np.concatenate(parts)
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError("no finite values to take limits from")

    lo, hi = values.min(), values.max()
    margin = pad * (hi - lo)
    return lo - margin, hi + margin


def save_frame(fig, directory, index, name, close=True, facecolor="white", **kwargs):
    """Write one buildup frame at the figure's own size; return its path.

    Do not pass bbox_inches='tight': trimming to the drawn content changes the
    canvas between frames, which is the jitter the fixed geometry exists to
    prevent.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{index:02d}_{name}.png"
    fig.savefig(path, facecolor=facecolor, **kwargs)
    if close:
        plt.close(fig)
    return path
