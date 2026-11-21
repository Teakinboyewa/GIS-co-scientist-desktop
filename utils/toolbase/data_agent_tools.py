import os
import geopandas as gpd
import zipfile
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List


def sniff_format(path: str) -> Tuple[str, str]:
    """Determine file format from extension."""
    ext = Path(path).suffix.lower()

    if ext in ['.shp', '.geojson', '.gpkg', '.gdb']:
        return "vector", ext
    elif ext in ['.csv', '.xlsx', '.xls', '.parquet']:
        return "table", ext
    elif ext == '.zip':
        return "zip", ext
    else:
        return "unknown", ext



def read_vector(path: str, geom_col="geometry", n_preview=2) -> Dict[str, Any]:
    """
    Load a GeoDataFrame and return lightweight metadata + geometry checks.
    Supports:
      - regular file paths
      - zipped shapefiles via "zip://<path>.zip"
      - zipped shapefiles without member: will auto-pick the first .shp inside the zip
    """
    info: Dict[str, Any] = {"ok": False, "n_rows": None, "n_cols": None, "crs": None, "head": None, "error": None}

    raw = str(path)
    is_zip_url = raw.startswith("zip://")

    # Normalize local path and existence check
    if is_zip_url:
        local = raw[len("zip://"):]
        local_path = Path(local)
    else:
        local_path = Path(raw)

    if not local_path.exists():
        info["error"] = "File not found"
        return info

    # Build a GDAL/Fiona-readable path
    gdal_path = None
    if is_zip_url:
        # If user already specified an inner member like zip://...!file.shp, use as-is
        if "!" in raw:
            gdal_path = "zip://" + local_path.as_posix()
        else:
            # No member provided: try to detect the first .shp inside the zip
            try:
                with zipfile.ZipFile(local_path, "r") as zf:
                    shp_members = [m for m in zf.namelist() if m.lower().endswith(".shp")]
                if shp_members:
                    # Use the first .shp found
                    gdal_path = f"zip://{local_path.as_posix()}!{shp_members[0]}"
                else:
                    # Fall back to plain zip://<file> (may still work if GDAL can infer a layer)
                    gdal_path = "zip://" + local_path.as_posix()
            except Exception as e:
                info["error"] = f"Zip inspection failed: {e}"
                return info
    else:
        gdal_path = local_path.as_posix()

    try:
        gdf = gpd.read_file(gdal_path)
        info["ok"] = True
        info["n_rows"], info["n_cols"] = gdf.shape
        info["columns"] = list(gdf.columns)
        info["crs"] = str(getattr(gdf, "crs", None))

        # pick geometry column (explicit or active)
        active_geom = getattr(gdf, "geometry", None)
        active_geom_name = active_geom.name if active_geom is not None else None
        geom_name = geom_col if geom_col in gdf.columns else active_geom_name
        if geom_name and geom_name in gdf.columns:
            geom = gdf[geom_name]
            invalid_cnt = int((~geom.is_valid).fillna(True).sum())
            empty_cnt = int((geom.is_empty).fillna(False).sum())
            info["geometry"] = {"geom_col": geom_name, "invalid": invalid_cnt, "empty": empty_cnt}
        else:
            info["geometry"] = {
                "geom_col": geom_col, "invalid": None, "empty": None,
                "note": "Geometry column not found; skipped geometry checks."
            }

        # preview (omit geometry column)
        drop_cols = [geom_name] if geom_name in gdf.columns else []
        info["head"] = (
            gdf.head(n_preview)
            .drop(columns=drop_cols, errors="ignore")
            .to_dict(orient="list")
        )

    except Exception as e:
        info["error"] = str(e)

    return info


def read_table(path: str, key_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """Read tabular data and return metadata."""
    try:
        import pandas as pd

        ext = Path(path).suffix.lower()
        if ext == '.csv':
            df = pd.read_csv(path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(path)
        elif ext == '.parquet':
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported table format: {ext}")

        result = {
            "ok": True,
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "columns": list(df.columns),
            "head": df.head(2).to_dict(orient="list"),
        }

        # Check key columns if provided
        if key_cols:
            result["keys"] = {}
            for k in key_cols:
                if k in df.columns:
                    miss = int(df[k].isna().sum())
                    dups = int(df[k].duplicated().sum())
                    result["keys"][k] = {
                        "present": True,
                        "missing": miss,
                        "duplicates": dups
                    }
                else:
                    result["keys"][k] = {"present": False}

        return result
    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }


def try_reproject(path: str, target_crs: str) -> Optional[str]:
    """Attempt to reproject vector data to target CRS."""
    try:
        import geopandas as gpd

        gdf = gpd.read_file(path)
        if str(gdf.crs) == target_crs:
            return None  # Already in target CRS

        # Reproject
        gdf_reproj = gdf.to_crs(target_crs)

        # Save reprojected file
        out_path = str(Path(path).with_suffix('')) + f"_{target_crs.replace(':', '_')}.gpkg"
        gdf_reproj.to_file(out_path, driver="GPKG")

        return out_path
    except Exception:
        return None