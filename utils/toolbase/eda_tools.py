"""
EDA tools for the agentic EDA loop.

Uses the same LangChain @tool pattern as debugger.py.
Tools are closures built via _build_eda_tools(df, output_dir) so they
can access and modify the working DataFrame.
"""

import os
import json
import numpy as np
import pandas as pd
from langchain_core.tools import tool


def _build_eda_tools(state):
    """
    Build EDA tools as closures over a mutable state dict.

    state keys:
        df          – the working DataFrame/GeoDataFrame (None until switch_dataset is called)
        output_dir  – path to save plots and cleaned data
        all_datasets – dict of all loaded datasets {name: {"df": DataFrame, "path": str}}
    """

    def _require_active_dataset():
        """Return the active DataFrame or None if no dataset is selected yet."""
        df = state.get("df")
        if df is None:
            return None
        return df

    NO_DATASET_MSG = "No dataset selected yet. Use `switch_dataset('<dataset_name>')` first to select a working dataset. Available datasets: {}"

    def _no_dataset_error():
        available = list(state.get("all_datasets", {}).keys())
        return NO_DATASET_MSG.format(available)

    # ── Column Selection Tool ─────────────────────────────────────────

    @tool
    def select_relevant_columns(selections: str) -> str:
        """Select and categorize columns relevant to the research question.
        Call this FIRST before other analyses.
        IMPORTANT: Only select columns that are DIRECTLY relevant to answering the
        research question. Do NOT select every column — be selective.
        This tool searches across ALL loaded datasets automatically.

        selections: A JSON string mapping your own category labels to column names.
        Use whatever categories make sense for the research question. Use ONLY column names from the actual datasets.

        The JSON keys are your category labels (any names that fit your analysis),
        and the values are lists of actual column names from any loaded dataset.

        Example format: select_relevant_columns('{"category1": ["actual_col1"], "category2": ["actual_col2","actual_col3"]}')"""

        # Build a map of all columns across all datasets
        all_datasets = state.get("all_datasets", {})
        col_to_datasets = {}  # column_name -> list of dataset names that have it
        for ds_name, ds_info in all_datasets.items():
            for col in ds_info["df"].columns:
                col_to_datasets.setdefault(str(col), []).append(ds_name)

        all_available_cols = set(col_to_datasets.keys())

        try:
            categories = json.loads(selections)
        except json.JSONDecodeError:
            return "Error: selections must be a valid JSON string. Example: '{\"category\": [\"col1\", \"col2\"]}'"

        if not isinstance(categories, dict):
            return "Error: selections must be a JSON object mapping category names to column lists."

        selected = {}
        missing = []
        total_selected = 0
        result_lines = []

        for category, cols in categories.items():
            if not isinstance(cols, list):
                cols = [cols]
            valid = []
            for c in cols:
                c = str(c).strip()
                if c in all_available_cols:
                    valid.append(c)
                else:
                    missing.append(c)
            selected[category] = valid
            total_selected += len(valid)
            if valid:
                # Show which dataset each column comes from if multiple datasets
                if len(all_datasets) > 1:
                    col_sources = [f"{c} ({col_to_datasets[c][0]})" for c in valid]
                    result_lines.append(f"  {category}: {col_sources}")
                else:
                    result_lines.append(f"  {category}: {valid}")

        state["selected_columns"] = selected

        header = [f"Selected {total_selected} relevant columns across {len(selected)} categories:"]
        if missing:
            result_lines.append(f"  WARNING - Not found in any dataset: {missing}")
        result_lines.append(f"  Total available columns across all datasets: {len(all_available_cols)}")
        return "\n".join(header + result_lines)

    # ── Data Assessment Tools ─────────────────────────────────────────

    @tool
    def assess_missing_data(columns: str = "") -> str:
        """Check missing value counts and percentages for specified columns.
        Pass column names as comma-separated string, or leave empty to check all.
        Example: assess_missing_data('col1, col2') or assess_missing_data('')"""
        if _require_active_dataset() is None:
            return _no_dataset_error()
        df = state["df"]
        cols = [c.strip() for c in columns.split(",") if c.strip()] if columns else list(df.columns)
        cols = [c for c in cols if c in df.columns]
        total = len(df)
        records = []
        for c in cols:
            miss = int(df[c].isna().sum())
            if miss > 0:
                records.append({"column": c, "missing": miss, "pct": round(miss / total * 100, 2)})
        if not records:
            return f"No missing values found in {len(cols)} column(s) checked."
        records.sort(key=lambda r: r["missing"], reverse=True)
        lines = [f"Missing data ({len(records)} of {len(cols)} columns, {total} rows):"]
        for r in records:
            lines.append(f"  {r['column']}: {r['missing']} ({r['pct']}%)")
        return "\n".join(lines)

    @tool
    def descriptive_statistics(columns: str) -> str:
        """Get summary statistics (count, mean, std, min, max, quartiles) for numeric columns.
        Pass column names as comma-separated string.
        Example: descriptive_statistics('%Poverty, %Black, MedianIncome')"""
        if _require_active_dataset() is None:
            return _no_dataset_error()
        df = state["df"]
        cols = [c.strip() for c in columns.split(",") if c.strip() and c.strip() in df.columns]
        if not cols:
            return f"No valid columns found. Available: {list(df.columns)}"
        desc = df[cols].describe().round(4).to_string()
        return f"Descriptive statistics:\n{desc}"

    @tool
    def check_distribution(column: str) -> str:
        """Test normality and save histogram for a column. Returns skewness, kurtosis,
        Shapiro-Wilk test result, and saves histogram plot.
        Example: check_distribution('%Poverty')"""
        if _require_active_dataset() is None:
            return _no_dataset_error()
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats

        df = state["df"]
        if column not in df.columns:
            return f"Column '{column}' not found. Available: {list(df.columns)}"
        series = df[column].dropna()
        if len(series) < 8:
            return f"Too few non-null values ({len(series)}) for distribution analysis."

        skew = round(float(series.skew()), 4)
        kurt = round(float(series.kurtosis()), 4)
        sample = series.sample(min(5000, len(series)), random_state=42)
        try:
            stat, p_val = stats.shapiro(sample)
            normality = f"Shapiro-Wilk: stat={round(stat, 4)}, p={round(p_val, 6)}"
            is_normal = "likely normal" if p_val > 0.05 else "not normal"
        except Exception:
            normality = "Shapiro-Wilk: could not compute"
            is_normal = "unknown"

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.hist(series, bins=min(50, max(10, len(series) // 20)), edgecolor="black", alpha=0.7)
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        plot_path = os.path.join(state["output_dir"], f"dist_{column.replace('%','pct_')}.png")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=100)
        plt.close(fig)

        return (
            f"Distribution of '{column}' (n={len(series)}):\n"
            f"  Skewness: {skew}, Kurtosis: {kurt}\n"
            f"  {normality} -> {is_normal}\n"
            f"  Histogram saved: {plot_path}"
        )

    @tool
    def detect_outliers(column: str, method: str = "iqr") -> str:
        """Detect outliers using IQR or z-score method. Does NOT remove them.
        method: 'iqr' (1.5*IQR rule) or 'zscore' (|z| > 3).
        Example: detect_outliers('%Poverty', 'iqr')"""
        if _require_active_dataset() is None:
            return _no_dataset_error()
        df = state["df"]
        if column not in df.columns:
            return f"Column '{column}' not found. Available: {list(df.columns)}"
        series = df[column].dropna()

        if method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (series < lower) | (series > upper)
        else:
            z = np.abs((series - series.mean()) / series.std())
            mask = z > 3
            lower, upper = series.mean() - 3 * series.std(), series.mean() + 3 * series.std()

        n_outliers = int(mask.sum())
        pct = round(n_outliers / len(series) * 100, 2)
        return (
            f"Outliers in '{column}' ({method}):\n"
            f"  {n_outliers} outliers ({pct}%) out of {len(series)} values\n"
            f"  Bounds: [{round(lower, 4)}, {round(upper, 4)}]\n"
            f"  Data range: [{round(float(series.min()), 4)}, {round(float(series.max()), 4)}]"
        )

    @tool
    def frequency_analysis(column: str, top_n: int = 15) -> str:
        """Get value counts for a categorical column.
        Example: frequency_analysis('LandUse', 10)"""
        if _require_active_dataset() is None:
            return _no_dataset_error()
        df = state["df"]
        if column not in df.columns:
            return f"Column '{column}' not found. Available: {list(df.columns)}"
        vc = df[column].value_counts()
        total = len(df[column].dropna())
        n_unique = len(vc)
        lines = [f"Frequency of '{column}' ({n_unique} unique, {int(df[column].isna().sum())} missing):"]
        for val, cnt in vc.head(top_n).items():
            lines.append(f"  {val}: {cnt} ({round(cnt / total * 100, 1)}%)")
        if n_unique > top_n:
            lines.append(f"  ... and {n_unique - top_n} more categories")
        return "\n".join(lines)

    # ── Analysis Tools ────────────────────────────────────────────────

    @tool
    def correlation_analysis(columns: str, method: str = "pearson") -> str:
        """Compute correlation matrix and save heatmap. Flags highly correlated pairs (|r|>0.7).
        columns: comma-separated numeric column names. method: pearson/spearman/kendall.
        Example: correlation_analysis('%Poverty, %Black, MedianIncome', 'pearson')"""
        if _require_active_dataset() is None:
            return _no_dataset_error()
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = state["df"]
        cols = [c.strip() for c in columns.split(",") if c.strip() and c.strip() in df.columns]
        cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(cols) < 2:
            return f"Need at least 2 numeric columns. Valid numeric: {[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]}"

        corr = df[cols].corr(method=method).round(4)

        high_corrs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr.iloc[i, j]
                if abs(val) > 0.7:
                    high_corrs.append((cols[i], cols[j], val))
        high_corrs.sort(key=lambda x: abs(x[2]), reverse=True)

        fig, ax = plt.subplots(figsize=(max(8, len(cols)), max(6, len(cols) * 0.8)))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(cols, fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"Correlation Matrix ({method})")
        fig.tight_layout()
        plot_path = os.path.join(state["output_dir"], f"correlation_{method}.png")
        fig.savefig(plot_path, dpi=100)
        plt.close(fig)

        result = f"Correlation matrix ({method}, {len(cols)} cols) saved: {plot_path}\n"
        if high_corrs:
            result += "Highly correlated pairs (|r| > 0.7):\n"
            for c1, c2, val in high_corrs[:10]:
                result += f"  {c1} <-> {c2}: {val}\n"
        else:
            result += "No highly correlated pairs (|r| > 0.7)."
        return result

    @tool
    def bivariate_analysis(x_column: str, y_column: str) -> str:
        """Scatter plot (numeric vs numeric) or box plot (categorical vs numeric). Saves plot.
        Example: bivariate_analysis('%Black', '%Poverty')"""
        if _require_active_dataset() is None:
            return _no_dataset_error()
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = state["df"]
        for c in [x_column, y_column]:
            if c not in df.columns:
                return f"Column '{c}' not found. Available: {list(df.columns)}"

        x_numeric = pd.api.types.is_numeric_dtype(df[x_column])
        y_numeric = pd.api.types.is_numeric_dtype(df[y_column])

        fig, ax = plt.subplots(figsize=(8, 5))
        if x_numeric and y_numeric:
            clean = df[[x_column, y_column]].dropna()
            ax.scatter(clean[x_column], clean[y_column], alpha=0.4, s=10)
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            corr_val = clean[x_column].corr(clean[y_column])
            ax.set_title(f"{x_column} vs {y_column} (r={corr_val:.4f})")
            result = f"Scatter: {x_column} vs {y_column}, r={corr_val:.4f}, n={len(clean)}"
        else:
            cat_col = x_column if not x_numeric else y_column
            num_col = y_column if not x_numeric else x_column
            groups = df.groupby(cat_col)[num_col].apply(list).to_dict()
            labels = list(groups.keys())[:15]
            data = [groups[l] for l in labels]
            ax.boxplot(data, labels=labels)
            ax.set_xlabel(cat_col)
            ax.set_ylabel(num_col)
            ax.set_title(f"{num_col} by {cat_col}")
            plt.xticks(rotation=45, ha="right")
            result = f"Box plot: {num_col} by {cat_col} ({len(labels)} groups)"

        fig.tight_layout()
        safe_x = x_column.replace("%", "pct_")
        safe_y = y_column.replace("%", "pct_")
        plot_path = os.path.join(state["output_dir"], f"bivariate_{safe_x}_vs_{safe_y}.png")
        fig.savefig(plot_path, dpi=100)
        plt.close(fig)
        return result + f"\nPlot saved: {plot_path}"

    @tool
    def spatial_autocorrelation(column: str) -> str:
        """Compute Global Moran's I for spatial autocorrelation. Requires spatial data with geometry.
        Example: spatial_autocorrelation('%Poverty')"""
        if _require_active_dataset() is None:
            return _no_dataset_error()
        import geopandas as gpd
        from libpysal.weights import Queen
        from esda.moran import Moran

        df = state["df"]
        if column not in df.columns:
            return f"Column '{column}' not found. Available: {list(df.columns)}"
        if not isinstance(df, gpd.GeoDataFrame):
            return "Spatial autocorrelation requires a GeoDataFrame with geometry."

        gdf = df.dropna(subset=[column])
        if len(gdf) < 10:
            return f"Too few non-null rows ({len(gdf)}) for spatial autocorrelation."

        w = Queen.from_dataframe(gdf)
        w.transform = "r"
        mi = Moran(gdf[column].values, w)

        return (
            f"Global Moran's I for '{column}':\n"
            f"  I = {mi.I:.4f}\n"
            f"  Expected I = {mi.EI:.4f}\n"
            f"  p-value = {mi.p_sim:.6f}\n"
            f"  z-score = {mi.z_sim:.4f}\n"
            f"  {'Significant spatial autocorrelation' if mi.p_sim < 0.05 else 'No significant spatial autocorrelation'}"
        )

    @tool
    def plot_spatial_distribution(column: str, scheme: str = "quantiles") -> str:
        """Create a choropleth map of a variable. Requires geometry data.
        scheme: quantiles, equal_interval, natural_breaks.
        Example: plot_spatial_distribution('%Poverty', 'quantiles')"""
        if _require_active_dataset() is None:
            return _no_dataset_error()
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import geopandas as gpd

        df = state["df"]
        if column not in df.columns:
            return f"Column '{column}' not found. Available: {list(df.columns)}"
        if not isinstance(df, gpd.GeoDataFrame):
            return "Spatial plot requires a GeoDataFrame with geometry."

        scheme_map = {"quantiles": "Quantiles", "equal_interval": "EqualInterval", "natural_breaks": "NaturalBreaks"}
        mc_scheme = scheme_map.get(scheme, "Quantiles")

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        df.plot(column=column, scheme=mc_scheme, cmap="YlOrRd", legend=True,
                edgecolor="black", linewidth=0.2, ax=ax, missing_kwds={"color": "lightgrey"})
        ax.set_title(f"Spatial Distribution of {column} ({scheme})")
        ax.set_axis_off()
        fig.tight_layout()
        safe_name = column.replace("%", "pct_")
        plot_path = os.path.join(state["output_dir"], f"map_{safe_name}_{scheme}.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        return f"Choropleth map of '{column}' saved: {plot_path}"

    @tool
    def group_comparison(numeric_column: str, group_column: str) -> str:
        """Compare a numeric variable across groups. Runs t-test (2 groups) or ANOVA (3+).
        Example: group_comparison('%Poverty', 'Region')"""
        if _require_active_dataset() is None:
            return _no_dataset_error()
        from scipy import stats

        df = state["df"]
        for c in [numeric_column, group_column]:
            if c not in df.columns:
                return f"Column '{c}' not found. Available: {list(df.columns)}"

        groups = df.groupby(group_column)[numeric_column].apply(lambda x: x.dropna().tolist()).to_dict()
        valid_groups = {k: v for k, v in groups.items() if len(v) >= 2}

        if len(valid_groups) < 2:
            return f"Need at least 2 groups with 2+ values. Found {len(valid_groups)}."

        group_data = list(valid_groups.values())
        group_names = list(valid_groups.keys())
        lines = [f"Group comparison: {numeric_column} by {group_column} ({len(group_names)} groups)"]
        for name, data in zip(group_names, group_data):
            lines.append(f"  {name}: n={len(data)}, mean={np.mean(data):.4f}, std={np.std(data):.4f}")

        if len(group_names) == 2:
            stat, p_val = stats.ttest_ind(group_data[0], group_data[1])
            test_name = "t-test"
        else:
            stat, p_val = stats.f_oneway(*group_data)
            test_name = "ANOVA"

        lines.append(f"  {test_name}: stat={stat:.4f}, p={p_val:.6f}")
        lines.append(f"  {'Significant difference' if p_val < 0.05 else 'No significant difference'} (a=0.05)")
        return "\n".join(lines)

    # ── Preprocessing Tools ───────────────────────────────────────────

    @tool
    def handle_missing_values(columns: str, strategy: str) -> str:
        """Handle missing values in specified columns. MODIFIES the working dataset.
        columns: comma-separated column names. strategy: mean/median/mode/drop_rows/zero/ffill.
        Example: handle_missing_values('%Poverty, %Black', 'median')"""
        if _require_active_dataset() is None:
            return _no_dataset_error()
        df = state["df"]
        cols = [c.strip() for c in columns.split(",") if c.strip() and c.strip() in df.columns]
        if not cols:
            return f"No valid columns. Available: {list(df.columns)}"

        before = int(df[cols].isna().sum().sum())

        if strategy == "drop_rows":
            df = df.dropna(subset=cols).copy()
        elif strategy == "zero":
            df[cols] = df[cols].fillna(0)
        elif strategy == "ffill":
            df[cols] = df[cols].ffill()
        else:
            for c in cols:
                if strategy == "mean" and pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = df[c].fillna(df[c].mean())
                elif strategy == "median" and pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = df[c].fillna(df[c].median())
                elif strategy == "mode":
                    mode_val = df[c].mode()
                    if len(mode_val) > 0:
                        df[c] = df[c].fillna(mode_val.iloc[0])

        after = int(df[cols].isna().sum().sum())
        state["df"] = df
        return (
            f"Missing values handled ({strategy}) for {cols}:\n"
            f"  Before: {before} missing -> After: {after} missing\n"
            f"  Dataset shape: {df.shape}"
        )

    @tool
    def remove_outliers(column: str, method: str = "iqr") -> str:
        """Remove outlier rows from the dataset. MODIFIES the working dataset.
        method: 'iqr' or 'zscore'.
        Example: remove_outliers('%Poverty', 'iqr')"""
        if _require_active_dataset() is None:
            return _no_dataset_error()
        df = state["df"]
        if column not in df.columns:
            return f"Column '{column}' not found. Available: {list(df.columns)}"
        series = df[column].dropna()
        before = len(df)

        if method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            mask = (df[column] >= q1 - 1.5 * iqr) & (df[column] <= q3 + 1.5 * iqr)
        else:
            z = np.abs((df[column] - series.mean()) / series.std())
            mask = z <= 3

        mask = mask | df[column].isna()
        df = df[mask].copy()
        state["df"] = df
        return f"Removed {before - len(df)} outlier rows from '{column}' ({method}). Shape: {before} -> {len(df)}"

    @tool
    def transform_variable(column: str, method: str) -> str:
        """Apply transformation to a column (creates new column with suffix). MODIFIES dataset.
        method: log/sqrt/log1p/standardize/minmax.
        Example: transform_variable('%Poverty', 'log1p')"""
        if _require_active_dataset() is None:
            return _no_dataset_error()
        df = state["df"]
        if column not in df.columns:
            return f"Column '{column}' not found. Available: {list(df.columns)}"

        new_col = f"{column}_{method}"
        if method == "log":
            df[new_col] = np.log(df[column].clip(lower=1e-10))
        elif method == "sqrt":
            df[new_col] = np.sqrt(df[column].clip(lower=0))
        elif method == "log1p":
            df[new_col] = np.log1p(df[column].clip(lower=0))
        elif method == "standardize":
            mean, std = df[column].mean(), df[column].std()
            df[new_col] = (df[column] - mean) / std if std > 0 else 0
        elif method == "minmax":
            mn, mx = df[column].min(), df[column].max()
            df[new_col] = (df[column] - mn) / (mx - mn) if mx > mn else 0

        state["df"] = df
        return f"Created '{new_col}' via {method}. mean={df[new_col].mean():.4f}, std={df[new_col].std():.4f}"

    # ── General-Purpose Code Execution ───────────────────────────────

    @tool
    def execute_code(code: str) -> str:
        """Execute custom Python code on the working dataset when no existing tool can do the job.
        Use this as a fallback for any data operation not covered by the other tools
        (e.g., filtering rows, creating new columns, reshaping, pivoting, complex transformations).

        The code has access to:
          - `df`: the current working DataFrame (modify it in-place or reassign)
          - `output_dir`: path to save any outputs (plots, files)
          - `pd`, `np`, `os`, `json`: pre-imported

        If your code modifies df, assign the result back to `df` — it will be saved automatically.
        Print any results you want returned.

        Example: execute_code("df = df[df['StateAbbr'] == 'PA']\\nprint(f'Filtered to {len(df)} PA rows')")
        Example: execute_code("df['density'] = df['population'] / df['ALAND']\\nprint('Created density column')")"""
        if _require_active_dataset() is None:
            return _no_dataset_error()
        import io
        import contextlib

        df = state["df"]
        output_dir = state["output_dir"]

        # Set up execution namespace with useful imports and the working data
        exec_globals = {
            "df": df,
            "output_dir": output_dir,
            "pd": pd,
            "np": np,
            "os": os,
            "json": json,
        }

        stdout_capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(code, exec_globals)
        except Exception as e:
            return f"Error executing code: {e}"

        # If df was reassigned in the code, update state
        if exec_globals.get("df") is not None:
            state["df"] = exec_globals["df"]

        output = stdout_capture.getvalue().strip()
        new_shape = state["df"].shape
        result = output if output else "Code executed successfully."
        result += f"\nDataset shape: {new_shape[0]} rows x {new_shape[1]} cols"
        return result

    # ── Spatial Scale Aggregation ────────────────────────────────────

    @tool
    def aggregate_spatial_scale(group_by: str, agg_config: str, weight_column: str = "") -> str:
        """Aggregate the working dataset to a coarser geographic scale. MODIFIES the working dataset.
        Use when data is at a finer geographic level than the research question requires.

        group_by: Column to group by for aggregation.
        agg_config: JSON string mapping each column to its aggregation method.
                    Supported methods: "sum", "mean", "weighted_mean", "first", "mode", "min", "max".
                    Use "weighted_mean" for columns that should be averaged using the weight_column.
                    You decide the correct method for each column based on what it represents.
                    Columns not listed are dropped. The group_by column is kept automatically.
        weight_column: Column used as weights for any "weighted_mean" columns.
                       Required if any column uses "weighted_mean". The weight column itself
                       is summed in the output (total weight per group).

        Example: aggregate_spatial_scale('county_FIPS', '{"median_income": "weighted_mean", "population": "sum", "state_name": "first"}', 'population')"""
        if _require_active_dataset() is None:
            return _no_dataset_error()

        df = state["df"]
        if group_by not in df.columns:
            return f"Group-by column '{group_by}' not found. Available: {list(df.columns)}"

        try:
            config = json.loads(agg_config)
        except json.JSONDecodeError:
            return "Error: agg_config must be valid JSON. Example: '{\"col1\": \"sum\", \"col2\": \"weighted_mean\"}'"

        if not isinstance(config, dict):
            return "Error: agg_config must be a JSON object mapping column names to methods."

        # Validate columns and separate weighted_mean columns
        missing = [c for c in config if c not in df.columns]
        if missing:
            return f"Columns not found: {missing}. Available: {list(df.columns)}"

        weighted_cols = [c for c, m in config.items() if m == "weighted_mean"]
        if weighted_cols and not weight_column:
            return "Error: weight_column is required when using 'weighted_mean' method."
        if weight_column and weight_column not in df.columns:
            return f"Weight column '{weight_column}' not found. Available: {list(df.columns)}"

        # Skip geometry columns
        geo_col = None
        try:
            import geopandas as gpd
            if isinstance(df, gpd.GeoDataFrame) and df.geometry is not None:
                geo_col = df.geometry.name
        except Exception:
            pass

        before_shape = df.shape

        # Build standard aggregation dict (non-weighted methods)
        agg_dict = {}
        for col, method in config.items():
            if method == "weighted_mean" or col == geo_col:
                continue
            if method == "mode":
                agg_dict[col] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
            else:
                agg_dict[col] = method

        # Sum the weight column itself
        if weight_column and weight_column not in agg_dict:
            agg_dict[weight_column] = "sum"

        # Perform groupby for standard aggs
        if agg_dict:
            result_df = df.groupby(group_by, as_index=False).agg(agg_dict)
        else:
            result_df = df[[group_by]].drop_duplicates().reset_index(drop=True)

        # Compute weighted means separately
        for col in weighted_cols:
            clean = df[[group_by, col, weight_column]].dropna(subset=[col, weight_column])
            weighted = clean.groupby(group_by).apply(
                lambda g, c=col: np.average(g[c], weights=g[weight_column]) if len(g) > 0 else np.nan,
                include_groups=False
            ).reset_index(name=col)
            result_df = result_df.merge(weighted, on=group_by, how="left")

        state["df"] = result_df

        # Build summary
        lines = [f"Aggregated from {before_shape[0]} rows to {len(result_df)} groups (by '{group_by}')."]
        if weight_column:
            lines.append(f"  Weight column: {weight_column} (summed per group)")
        method_groups = {}
        for col, method in config.items():
            method_groups.setdefault(method, []).append(col)
        for method, cols in method_groups.items():
            lines.append(f"  {method}: {cols}")
        lines.append(f"  Result shape: {result_df.shape[0]} rows x {result_df.shape[1]} cols")
        return "\n".join(lines)

    # ── Utility Tools ─────────────────────────────────────────────────

    @tool
    def switch_dataset(dataset_name: str) -> str:
        """Select a dataset as the active working DataFrame. You MUST call this before using any analysis or preprocessing tool.
        Call this to switch between datasets when exploring or working with multiple datasets.
        Example: switch_dataset('demographics.csv')"""
        all_ds = state.get("all_datasets", {})
        if not all_ds:
            return "No datasets are loaded."
        if dataset_name not in all_ds:
            return f"Dataset '{dataset_name}' not found. Available: {list(all_ds.keys())}"
        state["df"] = all_ds[dataset_name]["df"]
        df = state["df"]
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items() if str(dtype) != "geometry"}
        return (
            f"Switched to '{dataset_name}': {df.shape[0]} rows x {df.shape[1]} cols.\n"
            f"Columns & dtypes: {json.dumps(dtypes)}"
        )

    @tool
    def find_join_columns(other_dataset: str, sample_size: int = 100) -> str:
        """Find potential join columns between the current dataset and another by comparing actual values.
        Use this BEFORE merge_datasets when column names differ between datasets.
        Returns columns with overlapping values that could serve as join keys.
        Example: find_join_columns('pennsylvania_data.gpkg')"""
        if _require_active_dataset() is None:
            return _no_dataset_error()
        all_ds = state.get("all_datasets", {})
        if other_dataset not in all_ds:
            return f"Dataset '{other_dataset}' not found. Available: {list(all_ds.keys())}"

        df = state["df"]
        other_df = all_ds[other_dataset]["df"]
        results = []

        # Get candidate columns (non-numeric or integer-like that could be IDs)
        def _id_candidates(frame):
            candidates = []
            for col in frame.columns:
                if str(frame[col].dtype) == "geometry":
                    continue
                nunique = frame[col].nunique()
                # Good join keys: many unique values relative to row count, or string/int type
                if nunique >= min(10, len(frame) * 0.01):
                    candidates.append(col)
            return candidates

        left_candidates = _id_candidates(df)
        right_candidates = _id_candidates(other_df)

        matches = []
        for lcol in left_candidates:
            left_vals = set(df[lcol].dropna().astype(str).head(sample_size * 10))
            if not left_vals:
                continue
            for rcol in right_candidates:
                right_vals = set(other_df[rcol].dropna().astype(str).head(sample_size * 10))
                if not right_vals:
                    continue
                overlap = left_vals & right_vals
                if overlap:
                    overlap_pct = round(len(overlap) / min(len(left_vals), len(right_vals)) * 100, 1)
                    if overlap_pct >= 5:  # At least 5% overlap
                        matches.append({
                            "left_col": lcol, "right_col": rcol,
                            "overlap_count": len(overlap), "overlap_pct": overlap_pct,
                            "sample_matches": list(overlap)[:5],
                            "left_dtype": str(df[lcol].dtype),
                            "right_dtype": str(other_df[rcol].dtype),
                        })

        matches.sort(key=lambda x: x["overlap_pct"], reverse=True)

        if not matches:
            return (
                f"No matching join columns found between current dataset and '{other_dataset}'.\n"
                f"  Current columns: {list(df.columns)}\n"
                f"  Other columns: {list(other_df.columns)}\n"
                "Consider using prepare_join_key to transform a column first."
            )

        lines = [f"Potential join columns between current dataset and '{other_dataset}':"]
        for m in matches[:10]:
            lines.append(
                f"  {m['left_col']} ({m['left_dtype']}) <-> {m['right_col']} ({m['right_dtype']}): "
                f"{m['overlap_count']} matches ({m['overlap_pct']}%), "
                f"e.g. {m['sample_matches']}"
            )
        return "\n".join(lines)

    @tool
    def prepare_join_key(column: str, operation: str, params: str = "") -> str:
        """Transform a column to make it compatible for merging with another dataset.
        MODIFIES the working dataset.

        Operations:
          rename:<new_name>     — rename the column
          to_string             — cast to string
          to_int                — cast to integer (drops non-numeric)
          zero_pad:<width>      — left-pad with zeros to fixed width (e.g. '5' for FIPS codes)
          extract:<pattern>     — extract substring using regex (e.g. '\\d{5}$' for last 5 digits)
          strip_prefix:<prefix> — remove a prefix (e.g. '0500000US')
          concat:<other_col>:<sep> — concatenate two columns with separator

        Examples:
          prepare_join_key('CountyFIPS', 'to_string')
          prepare_join_key('CountyFIPS', 'zero_pad', '5')
          prepare_join_key('FIPS', 'rename', 'GEOID')
          prepare_join_key('LocationID', 'strip_prefix', '0500000US')
          prepare_join_key('StateFIPS', 'concat', 'CountyFIPS:')"""
        if _require_active_dataset() is None:
            return _no_dataset_error()
        df = state["df"]
        if column not in df.columns:
            return f"Column '{column}' not found. Available: {list(df.columns)}"

        if operation == "to_string":
            df[column] = df[column].astype(str)
        elif operation == "to_int":
            df[column] = pd.to_numeric(df[column], errors="coerce")
            dropped = int(df[column].isna().sum())
            if dropped:
                return f"Cast '{column}' to numeric. {dropped} values could not be converted (set to NaN)."
        elif operation.startswith("rename"):
            new_name = params if params else operation.split(":", 1)[-1] if ":" in operation else ""
            if not new_name:
                return "Error: provide new column name. Example: prepare_join_key('col', 'rename', 'new_name')"
            df.rename(columns={column: new_name}, inplace=True)
            state["df"] = df
            return f"Renamed '{column}' -> '{new_name}'. Columns: {list(df.columns)}"
        elif operation.startswith("zero_pad"):
            width = int(params) if params else int(operation.split(":", 1)[-1]) if ":" in operation else 5
            df[column] = df[column].astype(str).str.zfill(width)
        elif operation.startswith("extract"):
            import re
            pattern = params if params else operation.split(":", 1)[-1] if ":" in operation else ""
            if not pattern:
                return "Error: provide regex pattern. Example: prepare_join_key('col', 'extract', '\\\\d{5}$')"
            df[column] = df[column].astype(str).str.extract(f"({pattern})", expand=False)
        elif operation.startswith("strip_prefix"):
            prefix = params if params else operation.split(":", 1)[-1] if ":" in operation else ""
            if not prefix:
                return "Error: provide prefix to strip. Example: prepare_join_key('col', 'strip_prefix', '0500000US')"
            df[column] = df[column].astype(str).str.replace(prefix, "", n=1, regex=False)
        elif operation.startswith("concat"):
            parts = params.split(":") if params else operation.split(":", 1)[-1].split(":") if ":" in operation else []
            if len(parts) < 1:
                return "Error: provide other column and optional separator. Example: prepare_join_key('col', 'concat', 'other_col:')"
            other_col = parts[0]
            sep = parts[1] if len(parts) > 1 else ""
            if other_col not in df.columns:
                return f"Column '{other_col}' not found. Available: {list(df.columns)}"
            new_col = f"{column}_{other_col}"
            df[new_col] = df[column].astype(str) + sep + df[other_col].astype(str)
            state["df"] = df
            return f"Created '{new_col}' by concatenating '{column}' + '{sep}' + '{other_col}'. Sample: {list(df[new_col].head(3))}"
        else:
            return f"Unknown operation '{operation}'. Available: rename, to_string, to_int, zero_pad, extract, strip_prefix, concat"

        state["df"] = df
        sample = list(df[column].dropna().head(5))
        return f"Transformed '{column}' via {operation}. Sample values: {sample}"

    @tool
    def merge_datasets(other_dataset: str, on: str = "", how: str = "inner",
                       left_on: str = "", right_on: str = "") -> str:
        """Merge the current working dataset with another loaded dataset.
        Only use when the research question requires variables from both datasets.

        Use 'on' when both datasets share the same column name for joining.
        Use 'left_on' and 'right_on' when join columns have different names.

        on: join column name (same in both datasets).
        left_on: join column in current dataset. right_on: join column in other dataset.
        how: inner/left/right/outer.
        Example: merge_datasets('geo.gpkg', on='GEOID', how='left')
        Example: merge_datasets('geo.gpkg', left_on='CountyFIPS', right_on='GEOID', how='left')"""
        if _require_active_dataset() is None:
            return _no_dataset_error()
        all_ds = state.get("all_datasets", {})
        if other_dataset not in all_ds:
            return f"Dataset '{other_dataset}' not found. Available: {list(all_ds.keys())}"
        df = state["df"]
        other_df = all_ds[other_dataset]["df"]
        before = df.shape

        if on:
            join_cols = [c.strip() for c in on.split(",")]
            for c in join_cols:
                if c not in df.columns:
                    return f"Column '{c}' not in current dataset. Available: {list(df.columns)}"
                if c not in other_df.columns:
                    return f"Column '{c}' not in '{other_dataset}'. Available: {list(other_df.columns)}"
            df = df.merge(other_df, on=join_cols, how=how)
            join_desc = f"on={join_cols}"
        elif left_on and right_on:
            if left_on not in df.columns:
                return f"Column '{left_on}' not in current dataset. Available: {list(df.columns)}"
            if right_on not in other_df.columns:
                return f"Column '{right_on}' not in '{other_dataset}'. Available: {list(other_df.columns)}"
            df = df.merge(other_df, left_on=left_on, right_on=right_on, how=how)
            join_desc = f"left_on={left_on}, right_on={right_on}"
        else:
            return "Error: provide either 'on' (same column name) or both 'left_on' and 'right_on' (different names)."

        state["df"] = df
        return (
            f"Merged with '{other_dataset}' ({join_desc}, {how}).\n"
            f"  Before: {before[0]} rows x {before[1]} cols -> After: {df.shape[0]} rows x {df.shape[1]} cols"
        )

    @tool
    def save_cleaned_data(filename: str = "cleaned_data.csv") -> str:
        """Save the current working dataset to disk. Call after preprocessing is done.
        Example: save_cleaned_data('cleaned_data.gpkg')"""
        if _require_active_dataset() is None:
            return _no_dataset_error()
        import geopandas as gpd

        df = state["df"]
        path = os.path.join(state["output_dir"], filename)

        if isinstance(df, gpd.GeoDataFrame) and not filename.endswith(".csv"):
            df.to_file(path, driver="GPKG")
        elif isinstance(df, gpd.GeoDataFrame) and filename.endswith(".csv"):
            df.drop(columns=[df.geometry.name], errors="ignore").to_csv(path, index=False)
            gpkg_path = path.replace(".csv", ".gpkg")
            df.to_file(gpkg_path, driver="GPKG")
            state["cleaned_data_path"] = gpkg_path
            return f"Saved to {path} (CSV) and {gpkg_path} (GeoPackage). Shape: {df.shape}"
        else:
            df.to_csv(path, index=False)

        state["cleaned_data_path"] = path
        return f"Saved cleaned data to {path}. Shape: {df.shape}"

    # ── User Approval Tool ───────────────────────────────────────────

    @tool
    def request_user_approval(question: str, options: str) -> str:
        """Ask the user for approval or clarification before proceeding.
        This PAUSES the EDA and waits for the user to respond.

        Use this when:
        - A column needed for the research question is not available and you want to suggest a proxy/alternative
        - You are unsure which column best represents a concept in the research question
        - You need to confirm a major preprocessing decision (e.g., dropping many rows)
        - The merge between datasets is ambiguous (multiple possible join keys)

        question: Clear question explaining the situation and what you need from the user.
        options: Comma-separated list of options for the user to choose from (2-5 options).

        Example: request_user_approval("Column 'poverty_rate' not found. The closest match is '%Poverty'. Should I use it?", "Yes use %Poverty, No skip this variable, Let me specify the column")
        Example: request_user_approval("Found two possible join keys: LocationID and CountyFIPS. Which should I use to merge with the spatial dataset?", "Use LocationID, Use CountyFIPS, Try both and compare")"""

        approval_queue = state.get("_approval_queue")
        response_queue = state.get("_response_queue")

        if not approval_queue or not response_queue:
            return f"User approval not available in this context. Proceeding with best judgment for: {question}"

        opts = [o.strip() for o in options.split(",") if o.strip()]
        if len(opts) < 2:
            opts = ["Yes", "No"]

        # Send approval request to frontend (blocks the agentic loop)
        approval_queue.put({
            "type": "eda_approval",
            "question": question,
            "options": opts,
        })

        # Block and wait for user response
        response = response_queue.get()

        # Handle file upload responses — load the new dataset into EDA state
        if isinstance(response, dict) and response.get("type") == "file_upload":
            file_path = response["file_path"]
            file_name = response.get("file_name", os.path.basename(file_path))
            audit_record = response.get("audit", {})

            # Load the dataset
            import geopandas as gpd
            ext = os.path.splitext(file_path)[1].lower()
            try:
                if ext in (".gpkg", ".shp", ".geojson"):
                    new_df = gpd.read_file(file_path)
                elif ext == ".csv":
                    new_df = pd.read_csv(file_path)
                elif ext in (".xlsx", ".xls"):
                    new_df = pd.read_excel(file_path)
                elif ext == ".zip":
                    from pathlib import Path
                    posix = Path(file_path).as_posix()
                    try:
                        new_df = gpd.read_file(f"zip://{posix}")
                    except Exception:
                        new_df = pd.read_csv(file_path)
                else:
                    return f"User uploaded file '{file_name}' but format '{ext}' is not supported."
            except Exception as e:
                return f"User uploaded file '{file_name}' but failed to load: {e}"

            # Add to available datasets
            all_ds = state.get("all_datasets", {})
            all_ds[file_name] = {"df": new_df, "path": file_path}
            state["all_datasets"] = all_ds

            # Build column info for LLM
            dtypes_info = {col: str(dtype) for col, dtype in new_df.dtypes.items()
                          if str(dtype) != "geometry"}

            return (
                f"User uploaded new dataset: '{file_name}' ({new_df.shape[0]} rows x {new_df.shape[1]} cols).\n"
                f"Columns & dtypes: {json.dumps(dtypes_info)}\n"
                f"Dataset is now available. Use `switch_dataset('{file_name}')` to work with it, "
                f"or `find_join_columns('{file_name}')` to find join keys for merging."
            )

        return f"User responded: {response}"

    # ── Return all tools ──────────────────────────────────────────────
    return [
        # Column selection (call first)
        select_relevant_columns,
        # Assessment
        assess_missing_data,
        descriptive_statistics,
        check_distribution,
        detect_outliers,
        frequency_analysis,
        # Analysis
        correlation_analysis,
        bivariate_analysis,
        spatial_autocorrelation,
        plot_spatial_distribution,
        group_comparison,
        # Preprocessing
        handle_missing_values,
        remove_outliers,
        transform_variable,
        # General-purpose fallback
        execute_code,
        # Spatial scale
        aggregate_spatial_scale,
        # Utility / Merge
        switch_dataset,
        find_join_columns,
        prepare_join_key,
        merge_datasets,
        save_cleaned_data,
        # User interaction
        request_user_approval,
    ]
