"""
Post-process a DOCX file to convert any remaining LaTeX equation source
(left as plain text in paragraphs) into native Word OMML equations —
editable, not images.

Primary path: pandoc subprocess (LaTeX -> OMML via a DOCX intermediate).
Manual fallback: best-effort plain-text OMML wrapper used when pandoc is
unavailable.

This module is a safety-net step run AFTER the main markdown -> DOCX
conversion in ``report_converter.py``. The expected primary path
(Pandoc-driven md -> DOCX) already produces OMML equations directly;
this module catches anything that slipped through, e.g. ``\\[ ... \\]``
style delimiters or equations in cached / legacy manuscripts.

Supported delimiter styles (priority order — $$/$ tried before \\[/\\
because the manuscript prompts now instruct the LLM to use dollar-style
math):

    $$ ... $$     display math (double-dollar)        — group 1
    $ ... $       inline math (single-dollar)         — group 2
    \\[ ... \\]   standard display math (legacy)      — group 3
    \\ ... \\     bare backslash, with/without spaces — group 4
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from typing import Callable, Optional

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree


# OMML namespace (Office Math Markup Language)
OMML_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


# Equation delimiter pattern. Order matters: $$ / $ are tried before
# \[ / \ so the LLM's preferred style (per the manuscript prompts) wins
# priority. The inline ($) alternative requires at least one
# LaTeX-distinctive character in the inner content (\, ^, _, {, }) so
# prose currency like "cost $5 to $10" is left alone.
EQ_PATTERN = re.compile(
    r'\$\$\s*(.*?)\s*\$\$'                              # 1: $$ ... $$
    r'|'
    r'\$([^\$\n]*?[\\^_{}][^\$\n]*?)\$'                 # 2: $ ... $ (LaTeX-y inner)
    r'|'
    r'\\\[\s*(.*?)\s*\\\]'                              # 3: \[ ... \]
    r'|'
    r'\\(?!\[)(.*?)(?<!\[)\\',                          # 4: \ ... \
    re.DOTALL,
)


def _extract_latex(match: re.Match) -> str:
    """Return the first non-None capture group, stripped of whitespace."""
    for i in range(1, 5):
        g = match.group(i)
        if g is not None:
            return g.strip()
    return ""


def _latex_to_omml_via_pandoc(latex: str, pandoc: str) -> Optional[etree._Element]:
    """Use pandoc to convert LaTeX -> OMML. Returns None on any failure."""
    tex_doc = (
        r"\documentclass{article}"
        r"\begin{document}"
        f"\\[{latex}\\]"
        r"\end{document}"
    )
    tex_path = None
    docx_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".tex", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(tex_doc)
            tex_path = f.name

        result = subprocess.run(
            [pandoc, tex_path, "-f", "latex", "-t", "docx", "--mathml", "-o", "-"],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None

        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as df:
            df.write(result.stdout)
            docx_path = df.name

        tmp_doc = Document(docx_path)
        for para in tmp_doc.paragraphs:
            for elem in para._element.iter():
                if elem.tag == f"{{{OMML_NS}}}oMathPara":
                    return elem
                if elem.tag == f"{{{OMML_NS}}}oMath":
                    return elem
        return None
    finally:
        for p in (tex_path, docx_path):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass


def _latex_to_omml_manual(latex: str) -> etree._Element:
    """Fallback: wrap raw LaTeX as a plain OMML run (not typeset)."""
    o_math_para = OxmlElement("m:oMathPara")
    o_math = OxmlElement("m:oMath")
    r = OxmlElement("m:r")
    t = OxmlElement("m:t")
    t.text = latex
    r.append(t)
    o_math.append(r)
    o_math_para.append(o_math)
    return o_math_para


def _replace_paragraph_with_equation(para, latex: str, pandoc: Optional[str]):
    """Replace a paragraph's contents with a centered OMML equation.

    NOTE: This destroys all other text/runs in the paragraph. It is intended
    for paragraphs that consist of a single equation (the typical shape for
    display math after a markdown -> DOCX conversion).
    """
    omml = None
    if pandoc:
        try:
            omml = _latex_to_omml_via_pandoc(latex, pandoc)
        except Exception:
            omml = None
    if omml is None:
        omml = _latex_to_omml_manual(latex)

    p_elem = para._p
    for child in list(p_elem):
        if child.tag != qn("w:pPr"):
            p_elem.remove(child)

    pPr = p_elem.find(qn("w:pPr"))
    if pPr is None:
        pPr = OxmlElement("w:pPr")
        p_elem.insert(0, pPr)
    jc = OxmlElement("w:jc")
    jc.set(qn("w:val"), "center")
    pPr.append(jc)
    p_elem.append(omml)


def render_latex_equations_in_docx(
    docx_path: str,
    output_path: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
) -> int:
    """Convert any remaining LaTeX equation source in a DOCX into OMML equations.

    Args:
        docx_path: Path to the input DOCX.
        output_path: Where to save. Defaults to ``docx_path`` (in-place edit).
        log: Optional callback for status messages (e.g. ``task_manager.log``).

    Returns:
        Number of equations replaced.
    """
    output_path = output_path or docx_path
    pandoc = shutil.which("pandoc")
    if log and not pandoc:
        log("[EQ] pandoc not on PATH — using manual OMML wrapper (no typesetting).")

    doc = Document(docx_path)
    replaced = 0

    for para in doc.paragraphs:
        text = para.text
        # Fast skip — paragraph cannot contain any of our delimiters
        if "$" not in text and "\\" not in text:
            continue
        match = EQ_PATTERN.search(text)
        if not match:
            continue
        latex = _extract_latex(match)
        if not latex:
            continue
        try:
            _replace_paragraph_with_equation(para, latex, pandoc)
            replaced += 1
            if log:
                preview = latex if len(latex) <= 70 else latex[:70] + "..."
                log(f"[EQ] Converted: {preview}")
        except Exception as e:
            if log:
                log(f"[EQ] Failed on '{latex[:40]}...': {e}")

    if replaced or output_path != docx_path:
        doc.save(output_path)

    if log:
        log(f"[EQ] {replaced} equation(s) converted; saved to {output_path}")
    return replaced


# Allow running as a script: `python -m agents.result_analysis_agent.equation_renderer in.docx out.docx`
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m agents.result_analysis_agent.equation_renderer "
              "<input.docx> [output.docx]")
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else in_path
    n = render_latex_equations_in_docx(in_path, out_path, log=print)
    print(f"Done. {n} equation(s) replaced.")
