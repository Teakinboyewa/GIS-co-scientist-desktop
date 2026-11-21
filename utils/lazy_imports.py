"""
Load heavy libraries on first use instead of at startup.

The app imports every agent when it starts, and those modules used to import
networkx, matplotlib, scikit-learn and IPython at the top, which cost about
3-4 seconds of the desktop app's startup even when a session never draws a
chart, fits a model or runs in a notebook.

    nx = lazy_module("networkx")            # then use nx.DiGraph() as usual
    display = lazy_function("IPython.display", "display")

The real module is imported the first time an attribute is used, and the
behaviour from then on is unchanged.
"""

import importlib
import importlib.util
import os


class LazyModule:
    """A stand-in for a module that imports it on first attribute access."""

    def __init__(self, name, before_import=None):
        object.__setattr__(self, "_lazy_name", name)
        object.__setattr__(self, "_lazy_module", None)
        object.__setattr__(self, "_lazy_before", before_import)

    def _load(self):
        mod = object.__getattribute__(self, "_lazy_module")
        if mod is None:
            before = object.__getattribute__(self, "_lazy_before")
            if before is not None:
                before()
            mod = importlib.import_module(object.__getattribute__(self, "_lazy_name"))
            object.__setattr__(self, "_lazy_module", mod)
        return mod

    def __getattr__(self, attr):
        return getattr(self._load(), attr)

    def __setattr__(self, attr, value):
        setattr(self._load(), attr, value)

    def __dir__(self):
        return dir(self._load())

    def __repr__(self):
        name = object.__getattribute__(self, "_lazy_name")
        loaded = object.__getattribute__(self, "_lazy_module") is not None
        return f"<lazy module {name!r}{' (loaded)' if loaded else ''}>"


def lazy_module(name):
    return LazyModule(name)


def lazy_function(module, name):
    """A function (or class used only by calling it, like IPython's HTML)
    that imports ``module`` when first called."""

    def call(*args, **kwargs):
        return getattr(importlib.import_module(module), name)(*args, **kwargs)

    call.__name__ = call.__qualname__ = name
    call.__doc__ = f"Calls {module}.{name}, importing {module} on first use."
    return call


def is_installed(name):
    """Whether a top-level package can be imported, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _use_agg():
    import matplotlib
    matplotlib.use("Agg")


def lazy_pyplot():
    """matplotlib.pyplot, loaded on first use, drawing off-screen.

    The figure modules used to call matplotlib.use("Agg") at import, so the
    server (which draws on worker threads) never picks a desktop GUI backend.
    MPLBACKEND gives any other early importer of matplotlib the same
    instruction, and matplotlib.use("Agg") runs before pyplot loads here."""
    os.environ["MPLBACKEND"] = "Agg"
    return LazyModule("matplotlib.pyplot", before_import=_use_agg)
