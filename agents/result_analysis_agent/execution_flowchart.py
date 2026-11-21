"""
Interactive execution flowchart: RQ → Objectives → Steps with click-to-view code.
Uses custom SVG with popup overlays (same format as research plan overview).
"""

import os
import json
import html as html_lib
import colorsys


def _lighter_shade(hex_color, amount=0.25):
    r, g, b = int(hex_color[1:3], 16) / 255, int(hex_color[3:5], 16) / 255, int(hex_color[5:7], 16) / 255
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    lr, lg, lb = colorsys.hls_to_rgb(h, min(l + amount, 0.92), s)
    return f"#{int(lr*255):02x}{int(lg*255):02x}{int(lb*255):02x}"


def _wrap_text(text, max_chars):
    """Wrap text into lines for SVG labels."""
    if not text:
        return [""]
    words = text.split()
    lines = []
    current = ""
    for w in words:
        if current and len(current) + len(w) + 1 > max_chars:
            lines.append(current)
            current = w
        else:
            current = f"{current} {w}" if current else w
    if current:
        lines.append(current)
    return lines[:3]  # max 3 lines


def generate_execution_flowchart(ctx, output_path=None):
    """
    Generate an interactive HTML flowchart showing the full research execution.
    Uses SVG with popup overlays matching the research plan overview style.
    Clicking a step node shows the final code used to execute it.

    Args:
        ctx: Pipeline context with artifacts
        output_path: Where to save the HTML (default: Outputs/execution_flowchart.html)

    Returns:
        Path to saved HTML file.
    """
    from utils.config import resolve_output_dir

    if not output_path:
        output_path = os.path.join(resolve_output_dir(ctx), "execution_flowchart.html")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plan = ctx.get("artifacts", {}).get("research_plan_final", {})
    step_results = ctx.get("artifacts", {}).get("step_execution_results", {})
    rq = plan.get("research_question", ctx.get("rq", ""))
    title = plan.get("title", "Research Execution")
    objectives = plan.get("objectives", [])

    colors_list = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    # Status colors
    STATUS_COLORS = {
        "completed": {"bg": "#28a745", "text": "white", "label": "OK"},
        "error":     {"bg": "#dc3545", "text": "white", "label": "ERR"},
        "pending":   {"bg": "#6c757d", "text": "white", "label": "?"},
    }

    nodes = []
    edges = []
    details = {}  # node_id → HTML string for popup

    # Layout parameters
    col_spacing = 280
    n_obj = len(objectives) if objectives else 1
    total_width = (n_obj - 1) * col_spacing
    start_x = 500 - total_width // 2
    rq_y = 50
    obj_y = 180
    step_start_y = 310
    row_spacing = 100

    max_steps = max((len(o.get("steps", [])) for o in objectives), default=1)
    # Account for executed steps that may exceed planned steps
    for obj_idx, obj in enumerate(objectives):
        obj_num = obj.get("number", obj_idx + 1)
        obj_key = f"objective_{obj_num}"
        executed = step_results.get(obj_key, [])
        planned = obj.get("steps", [])
        max_steps = max(max_steps, len(executed), len(planned))

    # Extra space for post-execution stages (result collation, manuscript)
    post_stage_start_y = step_start_y + max_steps * row_spacing + 60
    post_stage_spacing = 100
    svg_height = post_stage_start_y + 2 * post_stage_spacing + 80
    svg_width = max(1000, total_width + 400)

    # ── RQ node ──
    rq_width = min(svg_width - 100, 600)
    nodes.append({
        "id": "RQ", "x": svg_width // 2, "y": rq_y,
        "w": rq_width, "h": 60,
        "label": rq[:80], "color": "#2C3E50",
        "text_color": "white", "font_size": 13
    })
    details["RQ"] = f"<h3>Research Question</h3><p>{html_lib.escape(rq)}</p>"

    # ── Objectives and Steps ──
    for obj_idx, obj in enumerate(objectives):
        obj_num = obj.get("number", obj_idx + 1)
        obj_key = f"objective_{obj_num}"
        obj_title = obj.get("title", "")
        hypothesis = obj.get("hypothesis", "")
        planned_steps = obj.get("steps", [])
        color = colors_list[(obj_num - 1) % len(colors_list)]
        x = start_x + obj_idx * col_spacing
        obj_id = f"obj_{obj_num}"

        # Objective node
        nodes.append({
            "id": obj_id, "x": x, "y": obj_y,
            "w": 240, "h": 55,
            "label": f"Obj {obj_num}: {obj_title[:38]}", "color": color,
            "text_color": "white", "font_size": 11
        })
        edges.append({"from": "RQ", "to": obj_id, "color": "#555"})

        details[obj_id] = (
            f"<h3>Objective {obj_num}</h3>"
            f"<p><b>Title:</b> {html_lib.escape(obj_title)}</p>"
            f"<p><b>Hypothesis:</b> {html_lib.escape(hypothesis)}</p>"
        )

        # Steps
        executed = step_results.get(obj_key, [])
        exec_lookup = {s.get("step_index"): s for s in executed}
        num_steps = max(len(planned_steps), len(executed))

        # Extract depends_on
        step_deps = {}
        for i in range(num_steps):
            step_idx = i + 1
            if i < len(planned_steps):
                ps = planned_steps[i]
                if isinstance(ps, dict):
                    step_deps[step_idx] = ps.get("depends_on", [])
                else:
                    step_deps[step_idx] = []
            else:
                step_deps[step_idx] = []

        prev_id = obj_id
        step_id_map = {}  # step_idx → node_id string

        for i in range(num_steps):
            step_idx = i + 1
            sy = step_start_y + i * row_spacing

            # Get execution info
            exec_info = exec_lookup.get(step_idx, {})
            inner = exec_info.get("result", {}) if isinstance(exec_info.get("result"), dict) else {}
            status = exec_info.get("status", "pending")
            desc = exec_info.get("step_description", "")
            if not desc and i < len(planned_steps):
                ps = planned_steps[i]
                desc = ps if isinstance(ps, str) else ps.get("description", "")
            agent = exec_info.get("agent_name", "")
            error_msg = exec_info.get("error_message", "")
            code = inner.get("code", "")
            code_file = inner.get("code_file", "")
            output_preview = (inner.get("output", "") or "")[:500]
            created_files = inner.get("created_files", [])

            # Determine status key
            if status == "completed" and not error_msg:
                status_key = "completed"
            elif error_msg:
                status_key = "error"
            else:
                status_key = "pending"

            sc = STATUS_COLORS[status_key]
            status_icon = sc["label"]

            step_id = f"obj{obj_num}_s{step_idx}"
            step_id_map[step_idx] = step_id

            # Use lighter shade of objective color for step bg, overlaid with status border
            step_bg = _lighter_shade(color) if status_key == "pending" else (
                _lighter_shade("#28a745", 0.3) if status_key == "completed" else _lighter_shade("#dc3545", 0.3)
            )
            step_border = sc["bg"]

            op_label = ""
            if i < len(planned_steps) and isinstance(planned_steps[i], dict):
                op_label = planned_steps[i].get("operation", "")

            step_label = f"[{status_icon}] S{obj_num}.{step_idx}"
            if op_label:
                step_label += f": {op_label[:22]}"
            elif desc:
                step_label += f": {desc[:22]}"
            if agent:
                step_label += f" ({agent})"

            nodes.append({
                "id": step_id, "x": x, "y": sy,
                "w": 230, "h": 50,
                "label": step_label, "color": step_bg,
                "text_color": "#222", "font_size": 10,
                "shape": "parallelogram",
                "border_color": step_border
            })

            # Build popup HTML for this step
            status_color = "#28a745" if status_key == "completed" else ("#dc3545" if status_key == "error" else "#6c757d")
            status_text = "Completed" if status_key == "completed" else ("Error" if status_key == "error" else "Pending")

            card_phase = f"exec_obj{obj_num}_step{step_idx}"
            popup_html = (
                f'<h3 style="border-bottom-color:{color}; display:flex; align-items:center; justify-content:space-between; gap:8px;">'
                f'<span style="cursor:pointer;" onclick="goToStepCard(\'{card_phase}\')">'
                f'{html_lib.escape(obj_key.replace("_", " ").upper())} | Step {step_idx}</span>'
                f'<button onclick="goToStepCard(\'{card_phase}\')" '
                f'style="font-size:0.65em; padding:3px 10px; border:1px solid #4a90d9; border-radius:4px; '
                f'background:#f0f6ff; color:#2a6cb6; cursor:pointer; white-space:nowrap;">'
                f'Go to Card</button></h3>'
                f'<div class="meta-grid">'
                f'<div class="meta-item"><span class="meta-label">Status</span>'
                f'<span class="meta-value" style="color:{status_color};font-weight:bold;">{status_text}</span></div>'
                f'<div class="meta-item"><span class="meta-label">Agent</span>'
                f'<span class="meta-value">{html_lib.escape(agent or "N/A")}</span></div>'
                f'<div class="meta-item" style="grid-column:span 2"><span class="meta-label">Description</span>'
                f'<span class="meta-value">{html_lib.escape(desc)}</span></div>'
                f'</div>'
            )

            if code_file:
                popup_html += (
                    f'<div class="meta-item" style="margin-bottom:12px">'
                    f'<span class="meta-label">Code File</span>'
                    f'<span class="meta-value">{html_lib.escape(code_file)}</span></div>'
                )

            if code:
                popup_html += (
                    f'<div class="code-section"><h4>Executed Code</h4>'
                    f'<pre>{html_lib.escape(code)}</pre></div>'
                )

            if output_preview:
                popup_html += (
                    f'<div class="code-section output-section"><h4>Output</h4>'
                    f'<pre>{html_lib.escape(output_preview)}</pre></div>'
                )

            if error_msg:
                popup_html += (
                    f'<div class="code-section error-section"><h4>Error</h4>'
                    f'<pre>{html_lib.escape(error_msg)}</pre></div>'
                )

            if created_files:
                files_html = "".join(
                    f'<span class="file-tag">{html_lib.escape(str(f))}</span>' for f in created_files
                )
                popup_html += (
                    f'<div class="code-section"><h4>Created Files</h4>'
                    f'<div class="files-list">{files_html}</div></div>'
                )

            details[step_id] = popup_html

        # Add dependency edges
        for step_idx in range(1, num_steps + 1):
            deps = step_deps.get(step_idx, [])
            sid = step_id_map.get(step_idx)
            if not sid:
                continue
            if deps:
                for dep_idx in deps:
                    dep_sid = step_id_map.get(dep_idx)
                    if dep_sid:
                        edges.append({"from": dep_sid, "to": sid, "color": "#58a6ff"})
            else:
                edges.append({"from": obj_id, "to": sid, "color": color})

    # ── Post-execution stages: Result Collation → Manuscript ──
    artifacts = ctx.get("artifacts", {})
    center_x = svg_width // 2
    stage_w = 300
    stage_h = 55
    stage_pending = "#6c757d"      # gray for not yet done

    # Collect last step IDs from each objective (to draw converging edges)
    last_step_ids = []
    for obj_idx, obj in enumerate(objectives):
        obj_num = obj.get("number", obj_idx + 1)
        obj_key = f"objective_{obj_num}"
        executed = step_results.get(obj_key, [])
        planned = obj.get("steps", [])
        num_steps = max(len(planned), len(executed))
        if num_steps > 0:
            last_step_ids.append(f"obj{obj_num}_s{num_steps}")
        else:
            last_step_ids.append(f"obj_{obj_num}")

    # ── Result Collation node (yellow) ──
    collation_done = bool(artifacts.get("figure_catalog") or artifacts.get("result_presentation_content"))
    collation_color = "#D4A017" if collation_done else stage_pending  # gold/yellow when done
    collation_y = post_stage_start_y

    nodes.append({
        "id": "result_collation", "x": center_x, "y": collation_y,
        "w": stage_w, "h": stage_h,
        "label": "Result Collation" + (" [OK]" if collation_done else " [Pending]"),
        "color": collation_color, "text_color": "white", "font_size": 12,
        "border_color": "#28a745" if collation_done else "#6c757d"
    })
    for sid in last_step_ids:
        edges.append({"from": sid, "to": "result_collation", "color": "#555"})

    collation_popup = '<h3 style="border-bottom-color:#D4A017;">Result Collation</h3>'
    if collation_done:
        collation_popup += f'<div class="meta-item"><span class="meta-label">Status</span><span class="meta-value" style="color:#28a745;font-weight:bold;">Completed</span></div>'
        catalog = artifacts.get("figure_catalog", [])
        if catalog:
            collation_popup += f'<div class="meta-item" style="margin-top:8px"><span class="meta-label">Figures</span><span class="meta-value">{len(catalog)} figure(s) cataloged</span></div>'
    else:
        collation_popup += '<p style="color:#999;font-style:italic;">Not yet generated.</p>'
    details["result_collation"] = collation_popup

    # ── Manuscript node ──
    ms_content = artifacts.get("manuscript_content", "")
    ms_path = artifacts.get("manuscript_md", "")
    ms_done = bool(ms_content or ms_path)
    ms_color = "#7B2D8E" if ms_done else stage_pending  # purple when done
    ms_y = collation_y + post_stage_spacing

    nodes.append({
        "id": "manuscript", "x": center_x, "y": ms_y,
        "w": stage_w, "h": stage_h,
        "label": "Manuscript" + (" [OK]" if ms_done else " [Pending]"),
        "color": ms_color, "text_color": "white", "font_size": 12,
        "border_color": "#28a745" if ms_done else "#6c757d"
    })
    edges.append({"from": "result_collation", "to": "manuscript", "color": "#555"})

    ms_popup = '<h3 style="border-bottom-color:#7B2D8E;">Manuscript</h3>'
    if ms_done:
        ms_popup += f'<div class="meta-item"><span class="meta-label">Status</span><span class="meta-value" style="color:#28a745;font-weight:bold;">Completed</span></div>'
        if ms_path:
            ms_popup += f'<div class="meta-item" style="margin-top:8px"><span class="meta-label">File</span><span class="meta-value">{html_lib.escape(str(ms_path))}</span></div>'
        ms_docx = artifacts.get("result_manuscript_docx", "")
        ms_pdf = artifacts.get("result_manuscript_pdf", "")
        formats_html = ""
        if ms_docx:
            formats_html += f'<span class="file-tag">{html_lib.escape(str(ms_docx))}</span>'
        if ms_pdf:
            formats_html += f'<span class="file-tag">{html_lib.escape(str(ms_pdf))}</span>'
        if formats_html:
            ms_popup += f'<div class="code-section"><h4>Exported Formats</h4><div class="files-list">{formats_html}</div></div>'
        preview = (ms_content or "")[:1500]
        if preview:
            ms_popup += f'<div class="code-section"><h4>Content Preview</h4><pre style="color:#e2e8f0;max-height:400px;">{html_lib.escape(preview)}</pre></div>'
    else:
        ms_popup += '<p style="color:#999;font-style:italic;">Not yet generated.</p>'
    details["manuscript"] = ms_popup

    # ── Build node lookup ──
    node_lookup = {n["id"]: n for n in nodes}

    # ── Generate HTML ──
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{html_lib.escape(title)} - Execution Flowchart</title>
<style>
  body {{ margin: 0; font-family: 'Segoe UI', Arial, sans-serif; background: #f8f9fa; display: flex; flex-direction: column; height: 100vh; }}
  h1 {{ text-align: center; padding: 12px 0 4px 0; margin: 0; font-size: 18px; color: #2C3E50; }}
  .legend {{
    display: flex; gap: 16px; padding: 8px 24px; background: #fff; font-size: 0.8em;
    border-bottom: 1px solid #dee2e6; align-items: center; flex-wrap: wrap;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 3px; }}
  .graph-panel {{ flex: 1; overflow: hidden; position: relative; cursor: grab; }}
  svg {{ display: block; width: 100%; height: 100%; }}
  .node-box {{ cursor: pointer; transition: filter 0.15s; }}
  .node-box:hover {{ filter: brightness(1.12); }}
  marker {{ overflow: visible; }}

  /* Zoom controls toolbar */
  .zoom-toolbar {{
    position: absolute; top: 10px; right: 10px;
    display: flex; flex-direction: column; gap: 4px; z-index: 50;
    background: white; border-radius: 8px; padding: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15); border: 1px solid #ddd;
  }}
  .zoom-btn {{
    width: 32px; height: 32px; border: 1px solid #ddd; border-radius: 6px;
    background: #fff; cursor: pointer; font-size: 16px;
    display: flex; align-items: center; justify-content: center;
    color: #444; transition: all 0.15s; padding: 0;
  }}
  .zoom-btn:hover {{ background: #e9ecef; color: #111; }}
  .zoom-btn:active {{ background: #dee2e6; }}
  .zoom-level {{ font-size: 10px; text-align: center; color: #888; padding: 2px 0; user-select: none; }}
  .zoom-divider {{ height: 1px; background: #e0e0e0; margin: 2px 0; }}

  /* Popup overlay */
  .popup-overlay {{
    display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.35); z-index: 100; justify-content: center; align-items: center;
  }}
  .popup-overlay.visible {{ display: flex; }}

  /* Popup box */
  .popup-box {{
    background: white; border-radius: 10px; padding: 24px 28px;
    max-width: 800px; width: 92%; max-height: 80vh; overflow-y: auto;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25); position: relative;
    animation: popIn 0.18s ease-out;
  }}
  @keyframes popIn {{
    from {{ transform: scale(0.92); opacity: 0; }}
    to {{ transform: scale(1); opacity: 1; }}
  }}
  .popup-box h3 {{ margin-top: 0; color: #2C3E50; border-bottom: 2px solid #4C72B0; padding-bottom: 8px; }}
  .popup-box p {{ line-height: 1.6; color: #333; font-size: 13px; }}
  .popup-close {{
    position: absolute; top: 10px; right: 14px;
    background: none; border: none; font-size: 20px; color: #999;
    cursor: pointer; line-height: 1; padding: 4px 8px; border-radius: 4px;
  }}
  .popup-close:hover {{ background: #f0f0f0; color: #333; }}

  /* Execution popup specifics */
  .meta-grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 16px;
  }}
  .meta-item {{
    background: #f1f3f5; padding: 8px 12px; border-radius: 6px; font-size: 0.85em;
  }}
  .meta-label {{ display: block; color: #6c757d; font-size: 0.8em; }}
  .meta-value {{ display: block; color: #212529; margin-top: 2px; }}
  .code-section {{ margin-top: 12px; }}
  .code-section h4 {{ font-size: 0.95em; color: #495057; margin-bottom: 8px; margin-top: 0; }}
  pre {{
    background: #1e293b; padding: 14px; border-radius: 8px; overflow-x: auto;
    font-size: 0.82em; line-height: 1.5; color: #e2e8f0;
    font-family: 'Fira Code', 'Consolas', monospace; white-space: pre-wrap; word-wrap: break-word;
    max-height: 350px; overflow-y: auto;
  }}
  .output-section pre {{ color: #a3e635; background: #0c1222; }}
  .error-section pre {{ color: #f87171; background: #1c0c0c; }}
  .files-list {{ display: flex; flex-wrap: wrap; gap: 4px; }}
  .file-tag {{
    display: inline-block; background: #e7f5ff; color: #1971c2;
    padding: 2px 8px; border-radius: 4px; font-size: 0.8em;
  }}
</style>
</head>
<body>
<h1>{html_lib.escape(title)} — Execution Flowchart</h1>
<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#2C3E50;"></div> Research Question</div>
  <div class="legend-item"><div class="legend-dot" style="background:#4C72B0;"></div> Objective</div>
  <div class="legend-item"><div class="legend-dot" style="background:#28a745;"></div> Step (OK)</div>
  <div class="legend-item"><div class="legend-dot" style="background:#dc3545;"></div> Step (Error)</div>
  <div class="legend-item"><div class="legend-dot" style="background:#6c757d;"></div> Step (Pending)</div>
  <div class="legend-item"><div class="legend-dot" style="background:#58a6ff;"></div> Dependency</div>
  <div class="legend-item"><div class="legend-dot" style="background:#40916C;"></div> Analysis Stage</div>
  <div class="legend-item"><div class="legend-dot" style="background:#7B2D8E;"></div> Manuscript</div>
  <span style="margin-left:auto; color:#94a3b8;">Click any node to view details</span>
</div>
<div class="graph-panel" id="graphPanel">
<!-- Zoom controls -->
<div class="zoom-toolbar">
  <button class="zoom-btn" onclick="zoomIn()" title="Zoom in">+</button>
  <button class="zoom-btn" onclick="zoomOut()" title="Zoom out">&minus;</button>
  <div class="zoom-divider"></div>
  <span class="zoom-level" id="zoomLevel">100%</span>
  <div class="zoom-divider"></div>
  <button class="zoom-btn" onclick="zoomReset()" title="Reset view">&#8634;</button>
  <button class="zoom-btn" onclick="zoomFit()" title="Fit to view">&#11036;</button>
  <div class="zoom-divider"></div>
  <button class="zoom-btn" id="fullscreenBtn" onclick="toggleFullscreen()" title="Toggle fullscreen">&#x26F6;</button>
</div>
<svg id="flowchartSvg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#666"/>
    </marker>
    <marker id="arrowblue" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#58a6ff"/>
    </marker>
  </defs>
  <g id="zoomGroup">
"""

    # Draw edges
    for edge in edges:
        fn = node_lookup.get(edge["from"])
        tn = node_lookup.get(edge["to"])
        if not fn or not tn:
            continue
        x1 = fn["x"]
        y1 = fn["y"] + fn["h"] // 2
        x2 = tn["x"]
        y2 = tn["y"] - tn["h"] // 2
        ec = edge["color"]
        marker = "url(#arrowblue)" if ec == "#58a6ff" else "url(#arrowhead)"
        html_content += (
            f'    <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{ec}" stroke-width="2" marker-end="{marker}"/>\n'
        )

    # Draw nodes
    slant = 15
    for n in nodes:
        nx_ = n["x"] - n["w"] // 2
        ny = n["y"] - n["h"] // 2
        nid = n["id"]
        border_color = n.get("border_color", "#333")

        html_content += f'    <g class="node-box" onclick="showPopup(\'{nid}\')">\n'

        if n.get("shape") == "parallelogram":
            pts = (f"{nx_ + slant},{ny} "
                   f"{nx_ + n['w']},{ny} "
                   f"{nx_ + n['w'] - slant},{ny + n['h']} "
                   f"{nx_},{ny + n['h']}")
            html_content += (
                f'      <polygon points="{pts}" fill="{n["color"]}" '
                f'stroke="{border_color}" stroke-width="2"/>\n'
            )
        else:
            html_content += (
                f'      <rect x="{nx_}" y="{ny}" width="{n["w"]}" height="{n["h"]}" '
                f'rx="10" ry="10" fill="{n["color"]}" stroke="{border_color}" stroke-width="1.5"/>\n'
            )

        # Text rendering with word wrap
        label = n["label"]
        fs = n["font_size"]
        max_chars = n["w"] // max(int(fs * 0.55), 1)
        lines = _wrap_text(label, max_chars)

        line_height = fs + 3
        total_text_h = len(lines) * line_height
        text_start_y = n["y"] - total_text_h / 2 + fs * 0.4

        for li, line_text in enumerate(lines):
            ty = text_start_y + li * line_height
            safe_text = line_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_content += (
                f'      <text x="{n["x"]}" y="{ty}" text-anchor="middle" '
                f'fill="{n["text_color"]}" font-size="{fs}" font-weight="bold" '
                f'font-family="Segoe UI, Arial, sans-serif">{safe_text}</text>\n'
            )

        html_content += '    </g>\n'

    details_json = json.dumps(details, default=str)

    html_content += f"""  </g>
</svg>
</div>

<!-- Popup overlay -->
<div class="popup-overlay" id="popupOverlay" onclick="closePopup(event)">
  <div class="popup-box" id="popupBox" onclick="event.stopPropagation()">
    <button class="popup-close" onclick="closePopup()">&times;</button>
    <div id="popupContent"></div>
  </div>
</div>

<script>
const details = {details_json};

// Zoom & Pan state
let scale = 1;
let panX = 0, panY = 0;
let isPanning = false;
let _wasDragged = false;
let panStartX = 0, panStartY = 0;
let _mdX = 0, _mdY = 0;
const PAN_THRESHOLD = 4;
const MIN_SCALE = 0.2;
const MAX_SCALE = 5;
const ZOOM_STEP = 0.15;
const svg = document.getElementById('flowchartSvg');
const zoomGroup = document.getElementById('zoomGroup');
const graphPanel = document.getElementById('graphPanel');

function applyTransform() {{
  zoomGroup.setAttribute('transform', `translate(${{panX}}, ${{panY}}) scale(${{scale}})`);
  document.getElementById('zoomLevel').textContent = Math.round(scale * 100) + '%';
}}

function zoomIn() {{
  scale = Math.min(MAX_SCALE, scale + ZOOM_STEP);
  applyTransform();
}}

function zoomOut() {{
  scale = Math.max(MIN_SCALE, scale - ZOOM_STEP);
  applyTransform();
}}

function zoomReset() {{
  scale = 1; panX = 0; panY = 0;
  applyTransform();
}}

function zoomFit() {{
  const bbox = zoomGroup.getBBox();
  const vb = svg.viewBox.baseVal;
  const pad = 30;
  const scaleX = (vb.width - pad * 2) / bbox.width;
  const scaleY = (vb.height - pad * 2) / bbox.height;
  scale = Math.min(scaleX, scaleY, MAX_SCALE);
  panX = (vb.width / scale - bbox.width) / 2 - bbox.x + pad / scale;
  panY = (vb.height / scale - bbox.height) / 2 - bbox.y + pad / scale;
  applyTransform();
}}

function toggleFullscreen() {{
  const elem = document.documentElement;
  if (!document.fullscreenElement && !document.webkitFullscreenElement) {{
    if (elem.requestFullscreen) {{
      elem.requestFullscreen();
    }} else if (elem.webkitRequestFullscreen) {{
      elem.webkitRequestFullscreen();
    }}
  }} else {{
    if (document.exitFullscreen) {{
      document.exitFullscreen();
    }} else if (document.webkitExitFullscreen) {{
      document.webkitExitFullscreen();
    }}
  }}
}}

document.addEventListener('fullscreenchange', function() {{
  const btn = document.getElementById('fullscreenBtn');
  if (document.fullscreenElement) {{
    btn.innerHTML = '&#x2716;';
    btn.title = 'Exit fullscreen';
  }} else {{
    btn.innerHTML = '&#x26F6;';
    btn.title = 'Toggle fullscreen';
  }}
  setTimeout(zoomFit, 200);
}});

// Mouse wheel zoom (centered on cursor)
graphPanel.addEventListener('wheel', function(e) {{
  e.preventDefault();
  const rect = svg.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;
  const vb = svg.viewBox.baseVal;
  const svgX = (mouseX / rect.width) * vb.width;
  const svgY = (mouseY / rect.height) * vb.height;
  const oldScale = scale;
  if (e.deltaY < 0) {{
    scale = Math.min(MAX_SCALE, scale + ZOOM_STEP);
  }} else {{
    scale = Math.max(MIN_SCALE, scale - ZOOM_STEP);
  }}
  panX = svgX / scale - svgX / oldScale + panX;
  panY = svgY / scale - svgY / oldScale + panY;
  applyTransform();
}}, {{ passive: false }});

// Pan with left-click drag (or middle mouse)
graphPanel.addEventListener('mousedown', function(e) {{
  if (e.button === 0 || e.button === 1) {{
    e.preventDefault();
    isPanning = true;
    _wasDragged = false;
    panStartX = e.clientX;
    panStartY = e.clientY;
    _mdX = e.clientX;
    _mdY = e.clientY;
    graphPanel.style.cursor = 'grabbing';
  }}
}});

window.addEventListener('mousemove', function(e) {{
  if (!isPanning) return;
  const dist = Math.abs(e.clientX - _mdX) + Math.abs(e.clientY - _mdY);
  if (dist > PAN_THRESHOLD) _wasDragged = true;
  const vb = svg.viewBox.baseVal;
  const rect = svg.getBoundingClientRect();
  const dx = (e.clientX - panStartX) * (vb.width / rect.width) / scale;
  const dy = (e.clientY - panStartY) * (vb.height / rect.height) / scale;
  panX += dx;
  panY += dy;
  panStartX = e.clientX;
  panStartY = e.clientY;
  applyTransform();
}});

window.addEventListener('mouseup', function() {{
  if (isPanning) {{
    isPanning = false;
    graphPanel.style.cursor = '';
  }}
}});

// Suppress node clicks after a drag
graphPanel.addEventListener('click', function(e) {{
  if (_wasDragged) {{
    e.stopPropagation();
    _wasDragged = false;
  }}
}}, true);

// Node popup
function showPopup(nodeId) {{
  const content = document.getElementById('popupContent');
  const overlay = document.getElementById('popupOverlay');
  if (details[nodeId]) {{
    content.innerHTML = details[nodeId];
  }} else {{
    content.innerHTML = '<p style="color:#999; font-style:italic;">No details available.</p>';
  }}
  overlay.classList.add('visible');

  // Highlight selected node
  document.querySelectorAll('.node-box rect, .node-box polygon').forEach(el => {{
    el.setAttribute('stroke-width', el.tagName === 'polygon' ? '2' : '1.5');
    el.setAttribute('stroke', el.dataset.origStroke || '#333');
  }});
  const clicked = event.currentTarget.querySelector('rect, polygon');
  if (clicked) {{
    clicked.setAttribute('stroke-width', '3.5');
    clicked.setAttribute('stroke', '#FFD700');
  }}
}}

function closePopup(e) {{
  if (e && e.target !== document.getElementById('popupOverlay')) return;
  document.getElementById('popupOverlay').classList.remove('visible');
  document.querySelectorAll('.node-box rect, .node-box polygon').forEach(el => {{
    el.setAttribute('stroke-width', el.tagName === 'polygon' ? '2' : '1.5');
    el.setAttribute('stroke', el.dataset.origStroke || '#333');
  }});
}}

// Navigate to the step card in the parent WebUI
function goToStepCard(phase) {{
  closePopup();
  // If inside an iframe, post message to parent; otherwise try directly
  if (window.parent && window.parent !== window) {{
    window.parent.postMessage({{ type: 'scrollToStepCard', phase: phase }}, '*');
  }} else {{
    // Standalone — try finding the card in this document
    const card = document.getElementById('wf-card-' + phase);
    if (card) card.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
  }}
}}

document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape') closePopup();
}});

// Store original stroke colors
document.querySelectorAll('.node-box rect, .node-box polygon').forEach(el => {{
  el.dataset.origStroke = el.getAttribute('stroke') || '#333';
}});

// Fit to view on initial load
window.addEventListener('DOMContentLoaded', function() {{
  setTimeout(zoomFit, 100);
}});
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Persist to DB for cross-machine sharing
    task_id = ctx.get("task_id")
    if task_id:
        try:
            from WebUI.conversation_db import save_artifact
            save_artifact(task_id, "execution_flowchart", html_content)
        except Exception:
            pass

    print(f"Execution flowchart saved to: {output_path}")
    return output_path