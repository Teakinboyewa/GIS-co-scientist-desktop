"""
Runtime guard that forces file-write APIs in LLM-generated code to land
inside the per-task OUTPUT_DIR. Absolute paths pointing outside OUTPUT_DIR
are transparently rewritten to os.path.join(OUTPUT_DIR, basename(path)).

Relative paths are left alone — the executor chdir's to OUTPUT_DIR before
exec, so they already resolve inside the task folder.

Usage:
    from utils.output_path_guard import install_output_path_guard

    with install_output_path_guard(output_dir):
        exec(code, exec_globals)

The patches are applied process-wide for the duration of the `with` block
and reverted on exit, so concurrent executions must be serialized (which
they already are — the assembly code runs on a single worker thread).
"""
import os
import builtins
import contextlib


def _looks_final(path_str):
    """Heuristic: path is intended as a 'final' deliverable, not intermediate."""
    low = path_str.lower().replace("\\", "/")
    if "final_outputs/" in low or "/final_outputs" in low:
        return True
    base = os.path.basename(low)
    return base.startswith("final_") or "_final." in base or base.startswith("final.")


def _redirect(path, output_dir, final_output_dir=None):
    """Return a safe path under output_dir (or final_output_dir) if `path` would escape it."""
    if path is None:
        return path
    try:
        path_str = os.fspath(path)
    except TypeError:
        return path
    if not isinstance(path_str, str) or not path_str:
        return path
    if not os.path.isabs(path_str):
        # Relative paths resolve against cwd (output_dir). Leave alone.
        return path
    abs_out = os.path.abspath(output_dir)
    abs_path = os.path.abspath(path_str)
    if abs_path == abs_out or abs_path.startswith(abs_out + os.sep):
        # Already inside OUTPUT_DIR (covers FINAL_OUTPUT_DIR too since it's nested).
        return path
    # Pick destination: final deliverables go to FINAL_OUTPUT_DIR, intermediates to OUTPUT_DIR.
    target_dir = output_dir
    if final_output_dir and _looks_final(path_str):
        target_dir = final_output_dir
    new_path = os.path.join(target_dir, os.path.basename(path_str))
    try:
        os.makedirs(os.path.dirname(new_path) or target_dir, exist_ok=True)
    except OSError:
        pass
    print(
        f"[OutputGuard] Redirected write: {path_str!r} -> {new_path!r}",
        flush=True,
    )
    return new_path


def _redirect_positional_or_kwarg(args, kwargs, output_dir, final_output_dir, kwarg_names):
    """Rewrite args[0] (if str/PathLike) or the first matching kwarg."""
    if args and isinstance(args[0], (str, os.PathLike)):
        return (_redirect(args[0], output_dir, final_output_dir),) + args[1:], kwargs
    for key in kwarg_names:
        if key in kwargs and isinstance(kwargs[key], (str, os.PathLike)):
            kwargs = dict(kwargs)
            kwargs[key] = _redirect(kwargs[key], output_dir, final_output_dir)
            break
    return args, kwargs


@contextlib.contextmanager
def install_output_path_guard(output_dir, final_output_dir=None):
    """Patch common save APIs for the duration of the context.

    Paths outside OUTPUT_DIR are rewritten into OUTPUT_DIR (intermediates) or
    FINAL_OUTPUT_DIR (if the path looks like a final deliverable — basename
    starts with 'final_' or path contains '/final_outputs/').
    """
    patches = []  # (owner, attr, original)

    def _patch(owner, attr, wrapper_factory):
        if not hasattr(owner, attr):
            return
        original = getattr(owner, attr)
        try:
            wrapper = wrapper_factory(original)
            setattr(owner, attr, wrapper)
            patches.append((owner, attr, original))
        except (AttributeError, TypeError):
            pass

    # ---- matplotlib ----
    try:
        import matplotlib.pyplot as _plt
        from matplotlib.figure import Figure as _Figure

        def _plt_savefig_wrapper(orig):
            def wrapped(*args, **kwargs):
                args, kwargs = _redirect_positional_or_kwarg(
                    args, kwargs, output_dir, final_output_dir, ("fname",)
                )
                return orig(*args, **kwargs)
            return wrapped

        def _fig_savefig_wrapper(orig):
            def wrapped(self, *args, **kwargs):
                args, kwargs = _redirect_positional_or_kwarg(
                    args, kwargs, output_dir, final_output_dir, ("fname",)
                )
                return orig(self, *args, **kwargs)
            return wrapped

        _patch(_plt, "savefig", _plt_savefig_wrapper)
        _patch(_Figure, "savefig", _fig_savefig_wrapper)
    except ImportError:
        pass

    # ---- geopandas ----
    try:
        from geopandas import GeoDataFrame as _GeoDataFrame

        def _gpd_to_file_wrapper(orig):
            def wrapped(self, *args, **kwargs):
                args, kwargs = _redirect_positional_or_kwarg(
                    args, kwargs, output_dir, final_output_dir, ("filename",)
                )
                return orig(self, *args, **kwargs)
            return wrapped

        _patch(_GeoDataFrame, "to_file", _gpd_to_file_wrapper)
    except ImportError:
        pass

    # ---- pandas ----
    try:
        from pandas import DataFrame as _DataFrame

        def _df_wrapper(orig):
            def wrapped(self, *args, **kwargs):
                args, kwargs = _redirect_positional_or_kwarg(
                    args, kwargs, output_dir, final_output_dir,
                    ("path_or_buf", "path", "excel_writer", "buf", "fname"),
                )
                return orig(self, *args, **kwargs)
            return wrapped

        for attr in ("to_csv", "to_parquet", "to_excel",
                     "to_json", "to_feather", "to_pickle",
                     "to_hdf", "to_html"):
            _patch(_DataFrame, attr, _df_wrapper)
    except ImportError:
        pass

    # ---- folium ----
    try:
        from folium import Map as _FoliumMap

        def _folium_save_wrapper(orig):
            def wrapped(self, *args, **kwargs):
                args, kwargs = _redirect_positional_or_kwarg(
                    args, kwargs, output_dir, final_output_dir, ("outfile",)
                )
                return orig(self, *args, **kwargs)
            return wrapped

        _patch(_FoliumMap, "save", _folium_save_wrapper)
    except ImportError:
        pass

    # ---- plotly (module-level io + Figure methods) ----
    try:
        import plotly.io as _pio

        def _pio_wrapper(orig):
            def wrapped(fig, *args, **kwargs):
                args, kwargs = _redirect_positional_or_kwarg(
                    args, kwargs, output_dir, final_output_dir, ("file",)
                )
                return orig(fig, *args, **kwargs)
            return wrapped

        _patch(_pio, "write_html", _pio_wrapper)
        _patch(_pio, "write_image", _pio_wrapper)
        _patch(_pio, "write_json", _pio_wrapper)
    except ImportError:
        pass

    try:
        from plotly.basedatatypes import BaseFigure as _BaseFigure

        def _plotly_fig_wrapper(orig):
            def wrapped(self, *args, **kwargs):
                args, kwargs = _redirect_positional_or_kwarg(
                    args, kwargs, output_dir, final_output_dir, ("file",)
                )
                return orig(self, *args, **kwargs)
            return wrapped

        for attr in ("write_html", "write_image", "write_json"):
            _patch(_BaseFigure, attr, _plotly_fig_wrapper)
    except ImportError:
        pass

    # ---- PIL ----
    try:
        from PIL import Image as _PILImage

        def _pil_save_wrapper(orig):
            def wrapped(self, *args, **kwargs):
                args, kwargs = _redirect_positional_or_kwarg(
                    args, kwargs, output_dir, final_output_dir, ("fp",)
                )
                return orig(self, *args, **kwargs)
            return wrapped

        _patch(_PILImage.Image, "save", _pil_save_wrapper)
    except ImportError:
        pass

    # ---- rasterio.open (write modes only) ----
    try:
        import rasterio as _rasterio

        def _rio_open_wrapper(orig):
            def wrapped(fp, mode="r", *args, **kwargs):
                if isinstance(fp, (str, os.PathLike)) and any(
                    m in mode for m in ("w", "a", "r+")
                ):
                    fp = _redirect(fp, output_dir, final_output_dir)
                return orig(fp, mode, *args, **kwargs)
            return wrapped

        _patch(_rasterio, "open", _rio_open_wrapper)
    except ImportError:
        pass

    # ---- builtin open() for write/append/exclusive modes ----
    _original_open = builtins.open

    def _open_wrapper(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and any(
            m in mode for m in ("w", "a", "x")
        ):
            file = _redirect(file, output_dir, final_output_dir)
        return _original_open(file, mode, *args, **kwargs)

    builtins.open = _open_wrapper
    patches.append((builtins, "open", _original_open))

    try:
        yield
    finally:
        for owner, attr, original in reversed(patches):
            try:
                setattr(owner, attr, original)
            except (AttributeError, TypeError):
                pass
