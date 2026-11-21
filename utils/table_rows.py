"""The rows of an output table, as text for Result Analysis and the Manuscript.

These stages used to see an output table as df.head(5).to_string() cut at
1500 characters. to_string pads every column to its widest value, so a
20-column results table lost everything after its third row: a regression
table listing OLS, spatial lag and spatial error models showed only the OLS
rows, and the manuscript reported that the spatial models gave no estimates.

Result tables (model coefficients, test statistics, rankings) are small, and
every row is a finding, so they are passed whole as compact CSV. Large tables
(per-feature data, raw observations) keep a short preview that says it is one.
"""

# Tables with at most this many rows are passed whole.
FULL_TABLE_MAX_ROWS = 60
# Preview size for larger tables.
PREVIEW_ROWS = 5


def table_rows_text(df, max_chars=8000, total_rows=None):
    """Rows of *df* as CSV text, saying explicitly when rows are left out.

    total_rows is the file's true row count when *df* holds only part of it.
    """
    df = df.drop(columns="geometry", errors="ignore")
    n = len(df) if total_rows is None else total_rows
    rows = df if n <= FULL_TABLE_MAX_ROWS and len(df) == n else df.head(PREVIEW_ROWS)
    text = rows.to_csv(index=False, float_format="%.6g")
    shown = len(rows)
    if len(text) > max_chars:
        text = text[:max_chars]
        text = text[:text.rfind("\n") + 1] if "\n" in text else text
        shown = max(text.count("\n") - 1, 0)
    if shown < n:
        text += f"[showing the first {shown} of {n} rows]\n"
    else:
        text += f"[all {n} rows]\n"
    return text
