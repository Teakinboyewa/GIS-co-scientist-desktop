"""
Converts markdown reports (result presentation / discussion) to DOCX and PDF.

DOCX path: tries Pandoc first (renders LaTeX `$...$` / `$$...$$` math as native
Word OMML equations); falls back to python-docx with a regex-based math handler
that italicises the LaTeX source.

PDF path: uses reportlab with the same regex-based math handler.

Images referenced via markdown syntax ![alt](path) are embedded directly.
"""

import os
import re
import shutil
import subprocess
import tempfile
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table as RLTable, TableStyle, PageBreak, Preformatted,
)
from reportlab.lib import colors


# ── Markdown Parsing Helpers ─────────────────────────────────────────────────

_RE_HEADING = re.compile(r'^(#{1,6})\s+(.*)')
_RE_IMAGE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
_RE_BOLD = re.compile(r'\*\*(.+?)\*\*')
_RE_ITALIC = re.compile(r'\*(.+?)\*')
_RE_TABLE_ROW = re.compile(r'^\|(.+)\|$')
_RE_TABLE_SEP = re.compile(r'^\|[\s:|-]+\|$')
_RE_BULLET = re.compile(r'^(\s*)[-*+]\s+(.*)')
_RE_NUMBERED = re.compile(r'^(\s*)\d+\.\s+(.*)')

# Math (used by the regex fallback path; Pandoc handles math natively).
# Display math: $$...$$ on a single line (also detected at the block level
# for the multi-line case via _RE_DISPLAY_MATH_OPEN/CLOSE).
_RE_DISPLAY_MATH_LINE = re.compile(r'^\s*\$\$(.+?)\$\$\s*$')
_RE_DISPLAY_MATH_OPEN = re.compile(r'^\s*\$\$\s*$')
# Inline math: $...$ — must not collide with $$...$$. To avoid false matches
# on prose currency ("$5 to $10"), require the inner content to contain at
# least one LaTeX-distinctive character (backslash, caret, underscore, brace).
# The LLM is instructed in section_prompts.py to use these for real math.
_RE_INLINE_MATH = re.compile(
    r'(?<!\$)\$(?!\$)([^\$\n]*?[\\^_{}][^\$\n]*?)(?<!\$)\$(?!\$)'
)
_RE_DISPLAY_MATH_INLINE = re.compile(r'\$\$([^\$]+?)\$\$')


def _math_to_italic(text):
    """Strip $...$ / $$...$$ delimiters and wrap inner LaTeX as markdown italic.

    This is the regex-fallback rendering: equations come out as italicised
    LaTeX source rather than typeset math. Used for the python-docx and
    reportlab paths when Pandoc is unavailable.

    Plain dollar amounts in prose ("cost $5 to $10") are left untouched
    because the inline regex requires at least one LaTeX-distinctive
    character in the inner content.
    """
    if not text or '$' not in text:
        return text
    # Display math first ($$...$$ — unambiguous, accept any inner content)
    text = _RE_DISPLAY_MATH_INLINE.sub(lambda m: f'*{m.group(1).strip()}*', text)
    # Inline math (only matches when inner contains \ ^ _ { or })
    text = _RE_INLINE_MATH.sub(lambda m: f'*{m.group(1).strip()}*', text)
    return text


def _is_image_valid(path):
    """Check if an image path exists and has a supported extension."""
    if not os.path.exists(path):
        return False
    ext = os.path.splitext(path)[1].lower()
    return ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')


# ── DOCX Post-Processing: Embed Images ─────────────────────────────────────

def _embed_docx_images(docx_path, max_width_inches=5.5):
    """
    Post-process a DOCX file: find any remaining markdown-style image
    placeholders ``![alt](path)`` left as plain text and replace them
    with actual embedded images + centered italic captions.

    This catches images that _md_to_docx missed (e.g. images inside
    list items like ``- ![Figure 1: ...](path)``).
    """
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    doc = Document(docx_path)
    replaced = 0

    for para in doc.paragraphs:
        full_text = "".join(run.text for run in para.runs)
        matches = _RE_IMAGE.findall(full_text)
        if not matches:
            continue

        alt_text, img_path = matches[0]
        img_path = img_path.strip()
        if not _is_image_valid(img_path):
            continue

        # Clear the text placeholder
        for run in list(para.runs):
            run._element.getparent().remove(run._element)

        # Center the paragraph and embed the image
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        try:
            from PIL import Image as PILImage
            with PILImage.open(img_path) as img:
                w_px, _ = img.size
                dpi = img.info.get("dpi", (96, 96))[0] or 96
                w_in = w_px / dpi
            display_w = Inches(min(w_in, max_width_inches))

            run = para.add_run()
            run.add_picture(img_path, width=display_w)
        except Exception:
            # Fallback: use fixed width
            try:
                run = para.add_run()
                run.add_picture(img_path, width=Inches(max_width_inches))
            except Exception:
                para.add_run(f"[Image: {os.path.basename(img_path)}]")
                continue

        # Insert a centered italic caption after the image paragraph
        if alt_text.strip():
            body = doc.element.body
            idx = list(body).index(para._element)

            cap_p = OxmlElement("w:p")
            cap_pPr = OxmlElement("w:pPr")
            cap_jc = OxmlElement("w:jc")
            cap_jc.set(qn("w:val"), "center")
            cap_pPr.append(cap_jc)
            cap_p.append(cap_pPr)

            cap_r = OxmlElement("w:r")
            cap_rPr = OxmlElement("w:rPr")
            cap_rPr.append(OxmlElement("w:i"))
            cap_sz = OxmlElement("w:sz")
            cap_sz.set(qn("w:val"), "18")  # 9pt
            cap_rPr.append(cap_sz)
            cap_r.append(cap_rPr)

            cap_t = OxmlElement("w:t")
            cap_t.text = alt_text.strip()
            cap_t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            cap_r.append(cap_t)
            cap_p.append(cap_r)

            body.insert(idx + 1, cap_p)

        replaced += 1

    if replaced:
        doc.save(docx_path)

    return replaced


def _inline_md_to_html(text):
    """Convert inline markdown (bold, italic, math) to HTML tags for reportlab Paragraph."""
    # Convert math to italic markdown first so it flows through the same path.
    text = _math_to_italic(text)
    text = _RE_BOLD.sub(r'<b>\1</b>', text)
    text = _RE_ITALIC.sub(r'<i>\1</i>', text)
    # Escape any remaining angle brackets that aren't our tags
    return text


# ── DOCX Conversion ─────────────────────────────────────────────────────────

def _add_formatted_runs(paragraph, text):
    """Parse inline markdown (bold, italic, math) and add as formatted runs."""
    # Convert any inline math into italic markdown before bold/italic splitting.
    text = _math_to_italic(text)
    parts = _RE_BOLD.split(text)
    for i, part in enumerate(parts):
        if i % 2 == 1:
            italic_parts = _RE_ITALIC.split(part)
            for j, ip in enumerate(italic_parts):
                if ip:
                    run = paragraph.add_run(ip)
                    run.bold = True
                    if j % 2 == 1:
                        run.italic = True
        else:
            italic_parts = _RE_ITALIC.split(part)
            for j, ip in enumerate(italic_parts):
                if ip:
                    run = paragraph.add_run(ip)
                    if j % 2 == 1:
                        run.italic = True


def _md_to_docx(md_content, output_path):
    """Convert markdown string to a DOCX document with embedded images."""
    doc = Document()

    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)

    lines = md_content.split('\n')
    i = 0
    in_code_block = False
    table_rows = []

    while i < len(lines):
        line = lines[i]

        # Code blocks
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            i += 1
            continue

        if in_code_block:
            p = doc.add_paragraph()
            p.style = doc.styles['Normal']
            run = p.add_run(line)
            run.font.name = 'Courier New'
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(40, 40, 40)
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            i += 1
            continue

        # Display math block: $$...$$ on a single line, or multi-line $$ ... $$
        single_dm = _RE_DISPLAY_MATH_LINE.match(line)
        if single_dm:
            _add_display_math_docx(doc, single_dm.group(1).strip())
            i += 1
            continue
        if _RE_DISPLAY_MATH_OPEN.match(line):
            j = i + 1
            buf = []
            while j < len(lines) and not _RE_DISPLAY_MATH_OPEN.match(lines[j]):
                buf.append(lines[j])
                j += 1
            _add_display_math_docx(doc, '\n'.join(buf).strip())
            # Skip past the closing $$ line (if found)
            i = j + 1 if j < len(lines) else j
            continue

        # Empty line
        if not line.strip():
            if table_rows:
                _flush_table(doc, table_rows)
                table_rows = []
            i += 1
            continue

        # Table rows
        table_match = _RE_TABLE_ROW.match(line.strip())
        if table_match:
            if not _RE_TABLE_SEP.match(line.strip()):
                cells = [c.strip() for c in table_match.group(1).split('|')]
                table_rows.append(cells)
            i += 1
            continue
        elif table_rows:
            _flush_table(doc, table_rows)
            table_rows = []

        # Headings
        heading_match = _RE_HEADING.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            doc.add_heading(text, level=min(level, 4))
            i += 1
            continue

        # Images (standalone line)
        img_match = _RE_IMAGE.search(line)
        if img_match and line.strip().startswith('!'):
            alt = img_match.group(1)
            img_path = img_match.group(2)
            if _is_image_valid(img_path):
                try:
                    doc.add_picture(img_path, width=Inches(5.5))
                    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                except Exception:
                    doc.add_paragraph(f"[Image: {alt} — {img_path}]")
            else:
                doc.add_paragraph(f"[Image: {alt} — {img_path}]")
            if alt:
                cap = doc.add_paragraph(alt)
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                if cap.runs:
                    cap.runs[0].italic = True
                    cap.runs[0].font.size = Pt(9)
            i += 1
            continue

        # Bullet lists
        bullet_match = _RE_BULLET.match(line)
        if bullet_match:
            text = bullet_match.group(2)
            p = doc.add_paragraph(style='List Bullet')
            _add_formatted_runs(p, text)
            i += 1
            continue

        # Numbered lists
        num_match = _RE_NUMBERED.match(line)
        if num_match:
            text = num_match.group(2)
            p = doc.add_paragraph(style='List Number')
            _add_formatted_runs(p, text)
            i += 1
            continue

        # Regular paragraph
        p = doc.add_paragraph()
        _add_formatted_runs(p, line)
        i += 1

    if table_rows:
        _flush_table(doc, table_rows)

    doc.save(output_path)
    return output_path


def _add_display_math_docx(doc, latex_source):
    """Render a display-math block in the python-docx fallback path.

    Word doesn't have a native equation API exposed via python-docx, so we
    render the LaTeX source as a centered italic paragraph. This is the
    degraded fallback — Pandoc, when available, produces real OMML equations.
    """
    if not latex_source:
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(latex_source)
    run.italic = True
    run.font.name = 'Cambria Math'
    run.font.size = Pt(11)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)


def _flush_table(doc, rows):
    """Add a table to the DOCX document from collected rows."""
    if not rows:
        return
    num_cols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=num_cols, style='Light Grid Accent 1')

    for ri, row in enumerate(rows):
        for ci, cell_text in enumerate(row):
            if ci < num_cols:
                cell = table.cell(ri, ci)
                cell.text = cell_text
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_before = Pt(1)
                    paragraph.paragraph_format.space_after = Pt(1)
                    for run in paragraph.runs:
                        run.font.size = Pt(9)
                        if ri == 0:
                            run.bold = True


# ── PDF Conversion (reportlab) ──────────────────────────────────────────────

def _build_pdf_styles():
    """Build custom paragraph styles for the PDF."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        'MDH1', parent=styles['Heading1'],
        fontSize=18, spaceAfter=10, spaceBefore=16,
        textColor=HexColor('#1a1a2e'),
        borderWidth=0, borderColor=HexColor('#1a1a2e'),
    ))
    styles.add(ParagraphStyle(
        'MDH2', parent=styles['Heading2'],
        fontSize=15, spaceAfter=8, spaceBefore=14,
        textColor=HexColor('#16213e'),
    ))
    styles.add(ParagraphStyle(
        'MDH3', parent=styles['Heading3'],
        fontSize=12, spaceAfter=6, spaceBefore=10,
        textColor=HexColor('#0f3460'),
    ))
    styles.add(ParagraphStyle(
        'MDH4', parent=styles['Heading4'],
        fontSize=11, spaceAfter=4, spaceBefore=8,
        textColor=HexColor('#333333'),
    ))
    styles.add(ParagraphStyle(
        'MDBullet', parent=styles['Normal'],
        fontSize=10, leftIndent=20, bulletIndent=10,
        spaceBefore=2, spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        'MDCode', parent=styles['Code'],
        fontSize=8, fontName='Courier',
        backColor=HexColor('#f4f4f4'),
        spaceBefore=2, spaceAfter=2,
        leftIndent=10,
    ))
    styles.add(ParagraphStyle(
        'MDCaption', parent=styles['Normal'],
        fontSize=9, alignment=TA_CENTER,
        textColor=HexColor('#555555'),
        spaceBefore=2, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        'MDMath', parent=styles['Normal'],
        fontSize=11, alignment=TA_CENTER,
        fontName='Helvetica-Oblique',
        spaceBefore=6, spaceAfter=6,
    ))

    return styles


def _add_display_math_pdf(story, styles, latex_source):
    """Render a display-math block in the reportlab fallback path.

    LaTeX source is shown as a centered italic paragraph. Pandoc handles real
    OMML for the DOCX path; reportlab has no native math typesetting.
    """
    if not latex_source:
        return
    # Escape angle brackets so reportlab doesn't try to parse them as tags.
    safe = latex_source.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    story.append(Paragraph(safe, styles['MDMath']))


def _md_to_pdf(md_content, output_path):
    """Convert markdown string to PDF using reportlab with embedded images."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.8 * inch,
        rightMargin=0.8 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
    )

    styles = _build_pdf_styles()
    story = []
    available_width = letter[0] - 1.6 * inch  # page width minus margins

    lines = md_content.split('\n')
    i = 0
    in_code_block = False
    code_lines = []
    table_rows = []

    heading_map = {1: 'MDH1', 2: 'MDH2', 3: 'MDH3', 4: 'MDH4'}

    while i < len(lines):
        line = lines[i]

        # Code blocks
        if line.strip().startswith('```'):
            if in_code_block:
                # End code block — flush
                if code_lines:
                    code_text = '\n'.join(code_lines)
                    story.append(Preformatted(code_text, styles['MDCode']))
                    story.append(Spacer(1, 6))
                code_lines = []
            in_code_block = not in_code_block
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Display math block: $$...$$ on a single line, or multi-line $$ ... $$
        single_dm = _RE_DISPLAY_MATH_LINE.match(line)
        if single_dm:
            _add_display_math_pdf(story, styles, single_dm.group(1).strip())
            i += 1
            continue
        if _RE_DISPLAY_MATH_OPEN.match(line):
            j = i + 1
            buf = []
            while j < len(lines) and not _RE_DISPLAY_MATH_OPEN.match(lines[j]):
                buf.append(lines[j])
                j += 1
            _add_display_math_pdf(story, styles, '\n'.join(buf).strip())
            i = j + 1 if j < len(lines) else j
            continue

        # Empty line
        if not line.strip():
            if table_rows:
                _flush_pdf_table(story, table_rows, styles, available_width)
                table_rows = []
            story.append(Spacer(1, 4))
            i += 1
            continue

        # Table rows
        table_match = _RE_TABLE_ROW.match(line.strip())
        if table_match:
            if not _RE_TABLE_SEP.match(line.strip()):
                cells = [c.strip() for c in table_match.group(1).split('|')]
                table_rows.append(cells)
            i += 1
            continue
        elif table_rows:
            _flush_pdf_table(story, table_rows, styles, available_width)
            table_rows = []

        # Headings
        heading_match = _RE_HEADING.match(line)
        if heading_match:
            level = min(len(heading_match.group(1)), 4)
            text = heading_match.group(2).strip()
            style_name = heading_map.get(level, 'MDH4')
            story.append(Paragraph(_inline_md_to_html(text), styles[style_name]))
            if level == 1:
                # Add a line under H1
                from reportlab.platypus import HRFlowable
                story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#1a1a2e')))
            i += 1
            continue

        # Images
        img_match = _RE_IMAGE.search(line)
        if img_match and line.strip().startswith('!'):
            alt = img_match.group(1)
            img_path = img_match.group(2)
            if _is_image_valid(img_path):
                try:
                    img = RLImage(img_path)
                    # Scale to fit within available width
                    aspect = img.imageHeight / img.imageWidth if img.imageWidth else 1
                    img_width = min(available_width, img.imageWidth)
                    img_height = img_width * aspect
                    # Cap height
                    max_height = 4 * inch
                    if img_height > max_height:
                        img_height = max_height
                        img_width = img_height / aspect
                    img.drawWidth = img_width
                    img.drawHeight = img_height
                    img.hAlign = 'CENTER'
                    story.append(img)
                except Exception:
                    story.append(Paragraph(f"[Image: {alt} — {img_path}]", styles['Normal']))
            else:
                story.append(Paragraph(f"[Image: {alt} — {img_path}]", styles['Normal']))
            if alt:
                story.append(Paragraph(alt, styles['MDCaption']))
            i += 1
            continue

        # Bullet lists
        bullet_match = _RE_BULLET.match(line)
        if bullet_match:
            text = bullet_match.group(2)
            html_text = f"• {_inline_md_to_html(text)}"
            story.append(Paragraph(html_text, styles['MDBullet']))
            i += 1
            continue

        # Numbered lists
        num_match = _RE_NUMBERED.match(line)
        if num_match:
            text = num_match.group(2)
            # Extract the number from the original line
            num = re.match(r'\s*(\d+)\.', line).group(1)
            html_text = f"{num}. {_inline_md_to_html(text)}"
            story.append(Paragraph(html_text, styles['MDBullet']))
            i += 1
            continue

        # Regular paragraph
        html_text = _inline_md_to_html(line)
        story.append(Paragraph(html_text, styles['Normal']))
        i += 1

    # Flush remaining
    if table_rows:
        _flush_pdf_table(story, table_rows, styles, available_width)
    if code_lines:
        story.append(Preformatted('\n'.join(code_lines), styles['MDCode']))

    doc.build(story)
    return output_path


def _flush_pdf_table(story, rows, styles, available_width):
    """Convert collected table rows into a reportlab Table."""
    if not rows:
        return

    num_cols = max(len(r) for r in rows)
    col_width = available_width / num_cols

    # Convert cells to Paragraphs
    table_data = []
    for ri, row in enumerate(rows):
        pdf_row = []
        for ci in range(num_cols):
            cell_text = row[ci] if ci < len(row) else ''
            cell_text = _inline_md_to_html(cell_text)
            if ri == 0:
                cell_text = f'<b>{cell_text}</b>'
            pdf_row.append(Paragraph(cell_text, styles['Normal']))
        table_data.append(pdf_row)

    table = RLTable(table_data, colWidths=[col_width] * num_cols)
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e8e8e8')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#999999')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#f9f9f9')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ])
    table.setStyle(table_style)
    story.append(table)
    story.append(Spacer(1, 8))


# ── Pandoc DOCX path (primary) ──────────────────────────────────────────────

def _pandoc_available():
    """Return the path to a usable pandoc executable, or None."""
    return shutil.which('pandoc')


def _md_to_docx_pandoc(md_content, output_path):
    """Convert markdown to DOCX via Pandoc.

    Pandoc's `markdown` reader has the `tex_math_dollars` extension enabled
    by default, so `$...$` and `$$...$$` are converted into native Word
    equations (OMML) automatically. Image paths in the markdown should
    already be absolute (callers run `_resolve_image_paths` first).

    Returns the output path on success. Raises on any failure so the caller
    can fall back to the python-docx path.
    """
    pandoc = _pandoc_available()
    if not pandoc:
        raise RuntimeError("pandoc executable not found on PATH")

    # Write the (already image-resolved) markdown to a temp file alongside
    # the output so any relative paths still resolve from a sane CWD.
    out_dir = os.path.dirname(os.path.abspath(output_path)) or '.'
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.md', delete=False, dir=out_dir, encoding='utf-8'
    ) as tmp:
        tmp.write(md_content)
        tmp_path = tmp.name

    try:
        cmd = [
            pandoc,
            tmp_path,
            '-o', output_path,
            '--from=markdown+tex_math_dollars+pipe_tables+raw_tex',
            '--to=docx',
            '--standalone',
        ]
        result = subprocess.run(
            cmd,
            cwd=out_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"pandoc exited with code {result.returncode}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        if not os.path.exists(output_path):
            raise RuntimeError("pandoc reported success but no output file was created")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return output_path


# ── Public API ───────────────────────────────────────────────────────────────

def _resolve_image_paths(md_content, ctx):
    """Resolve filename-only image paths to absolute paths.

    The figure catalog tells the LLM to use bare filenames (e.g. ``map.png``).
    Before DOCX/PDF conversion we need to expand them to full paths so the
    converter can find the image files on disk.

    Search order, first-hit wins:
      1. Path as-given (already absolute or relative to cwd).
      2. `<output_dir>/final_outputs/` — curated final deliverables.
      3. `<output_dir>/figures/auto/` — PNGs auto-rendered from gpkg/csv
         sources by `figure_renderer.realize_data_to_viz`. This is where
         catalog-phase images land; missing this directory is the reason
         only execution-phase (task-root) images used to appear in the
         compiled manuscript.
      4. `<output_dir>/figures/`, `plots/`, `images/`, `maps/` — conventional
         subdirs mirrored by the orphan-pickup scan in `figure_catalog_v2`.
      5. `<output_dir>/` — task root (execution-phase images).
      6. Recursive fallback: walk `<output_dir>` for a file with the same
         basename. This is the safety net for anything the fixed list missed.
    """
    from utils.config import resolve_output_dir
    output_dir = resolve_output_dir(ctx)

    search_dirs = [
        os.path.join(output_dir, "final_outputs"),
        os.path.join(output_dir, "figures", "auto"),
        os.path.join(output_dir, "figures"),
        os.path.join(output_dir, "plots"),
        os.path.join(output_dir, "images"),
        os.path.join(output_dir, "maps"),
        output_dir,
    ]

    # Cache the recursive walk so we do at most one filesystem traversal
    # per conversion even when many figures need resolution.
    _basename_index: dict = {}

    def _build_index():
        if _basename_index:
            return
        for root, _dirs, files in os.walk(output_dir):
            # Skip noisy irrelevant dirs
            parts = set(os.path.relpath(root, output_dir).split(os.sep))
            if parts & {"__pycache__", ".ipynb_checkpoints", "generated_code"}:
                continue
            for fn in files:
                _basename_index.setdefault(fn, os.path.join(root, fn))

    def _resolve(match):
        alt = match.group(1)
        img_path = match.group(2)
        if os.path.exists(img_path):
            return match.group(0)
        base = os.path.basename(img_path)
        for d in search_dirs:
            candidate = os.path.join(d, base)
            if os.path.exists(candidate):
                return f"![{alt}]({candidate.replace(os.sep, '/')})"
        # Recursive fallback (cached)
        _build_index()
        hit = _basename_index.get(base)
        if hit:
            return f"![{alt}]({hit.replace(os.sep, '/')})"
        return match.group(0)

    return _RE_IMAGE.sub(_resolve, md_content)


def convert_md_report(md_path, md_content, ctx, task_manager, report_type="presentation"):
    """
    Generate DOCX and PDF versions of a markdown report.

    Args:
        md_path: Path to the saved .md file
        md_content: Raw markdown string
        ctx: Pipeline context dict
        task_manager: TaskManager instance for logging
        report_type: "presentation" or "discussion"

    Returns:
        dict with 'docx' and 'pdf' paths (or None on failure)
    """
    # Resolve filename-only image paths to absolute paths for embedding
    md_content = _resolve_image_paths(md_content, ctx)

    base = os.path.splitext(md_path)[0]
    docx_path = base + ".docx"
    pdf_path = base + ".pdf"
    results = {"docx": None, "pdf": None}

    # DOCX — try Pandoc first (renders LaTeX math as native OMML equations),
    # fall back to python-docx + regex math handler on any failure.
    docx_engine = None
    try:
        _md_to_docx_pandoc(md_content, docx_path)
        docx_engine = "pandoc"
    except Exception as pandoc_err:
        task_manager.log(
            ctx,
            f"[{report_type.upper()}] Pandoc DOCX path unavailable, falling back to "
            f"python-docx: {pandoc_err}"
        )
        try:
            _md_to_docx(md_content, docx_path)
            docx_engine = "python-docx"
        except Exception as e:
            task_manager.log(ctx, f"[{report_type.upper()}] DOCX generation failed: {e}")
            docx_engine = None

    if docx_engine:
        try:
            # Post-process: embed any remaining ![alt](path) text placeholders.
            # Safe for both engines — only acts on plain-text placeholders.
            n_embedded = _embed_docx_images(docx_path)
            results["docx"] = docx_path
            ctx.setdefault("artifacts", {})[f"result_{report_type}_docx"] = docx_path
            extra = f" ({n_embedded} images post-embedded)" if n_embedded else ""
            task_manager.log(
                ctx,
                f"[{report_type.upper()}] DOCX saved to {docx_path} via {docx_engine}{extra}"
            )
        except Exception as e:
            task_manager.log(
                ctx,
                f"[{report_type.upper()}] DOCX post-processing failed: {e}"
            )

        # Final post-processing: convert any remaining LaTeX equation source
        # ($$...$$, $...$, \[...\], \...\) into native Word OMML equations.
        # No-op when the Pandoc path was used (Pandoc already produced OMML).
        try:
            from agents.result_analysis_agent.equation_renderer import (
                render_latex_equations_in_docx,
            )
            n_eq = render_latex_equations_in_docx(
                docx_path,
                log=lambda msg: task_manager.log(ctx, msg),
            )
            if n_eq:
                task_manager.log(
                    ctx,
                    f"[{report_type.upper()}] Rendered {n_eq} LaTeX equation(s) as OMML"
                )
        except Exception as e:
            task_manager.log(
                ctx,
                f"[{report_type.upper()}] LaTeX equation post-processing failed (non-fatal): {e}"
            )

    # PDF
    try:
        _md_to_pdf(md_content, pdf_path)
        results["pdf"] = pdf_path
        ctx.setdefault("artifacts", {})[f"result_{report_type}_pdf"] = pdf_path
        task_manager.log(ctx, f"[{report_type.upper()}] PDF saved to {pdf_path}")
    except Exception as e:
        task_manager.log(ctx, f"[{report_type.upper()}] PDF generation failed: {e}")

    return results
