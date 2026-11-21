"""
Research Plan Visualization
Generates interactive HTML visualization of research plans with expandable details
"""

import json
import os
from typing import Dict, Any


def _persist_artifact(task_id, artifact_type, content):
    """Best-effort save to DB; silently ignored if DB layer unavailable."""
    try:
        from WebUI.conversation_db import save_artifact
        save_artifact(task_id, artifact_type, content)
    except Exception:
        pass


def generate_research_plan_html(research_plan: Dict[str, Any], task_id: str = None) -> str:
    """
    Generate an interactive HTML visualization of the research plan.
    Supports the new structure: objectives as objects with number, title, hypothesis, steps[].
    """
    title = research_plan.get("title", "Research Plan")
    rq = research_plan.get("research_question", "")
    intent = research_plan.get("research_intent", "")
    objectives = research_plan.get("objectives", [])
    data_context = research_plan.get("data_context", {})
    considerations = research_plan.get("key_considerations", [])

    # Build objective cards
    objective_cards = ""
    for obj in objectives:
        idx = obj.get("number", 1)
        obj_title = obj.get("title", "")
        hypothesis = obj.get("hypothesis", "")
        steps = obj.get("steps", [])
        total_steps = len(steps)

        steps_html = ""
        for i, step in enumerate(steps, 1):
            step_desc = step.get('description', str(step)) if isinstance(step, dict) else step
            depends_on_json = json.dumps(step.get('depends_on', []) if isinstance(step, dict) else [])
            input_nodes_json = json.dumps(step.get('input_nodes', []) if isinstance(step, dict) else [])
            output_node = step.get('output_node', '') if isinstance(step, dict) else ''
            steps_html += f"""
                <div class="analysis-step" data-obj="{idx}" data-step="{i}" data-depends-on='{depends_on_json}' data-input-nodes='{input_nodes_json}' data-output-node="{output_node}">
                    <div class="step-header">
                        <span class="step-number">{i}</span>
                        <span class="step-desc" contenteditable="true">{step_desc}</span>
                        <button class="rerun-step-btn" onclick="rerunStep(event, {idx}, {i})" title="Edit &amp; Re-run this step">&#9654;</button>
                        <button class="delete-btn delete-step-btn" onclick="deleteStep(event, {idx}, {i})" title="Delete step">x</button>
                    </div>
                </div>"""

        objective_cards += f"""
        <div class="objective-card" id="objective-{idx}" data-obj="{idx}">
            <div class="objective-header" onclick="toggleObjective({idx})">
                <div class="objective-title">
                    <span class="objective-icon">🎯</span>
                    <span class="objective-number">Objective {idx}: <span class="obj-title-text" contenteditable="true" onclick="event.stopPropagation()">{obj_title}</span></span>
                    <span class="step-count" id="step-count-{idx}">({total_steps} steps)</span>
                </div>
                <div class="objective-actions">
                    <button class="delete-btn delete-obj-btn" onclick="deleteObjective(event, {idx})" title="Delete objective">x</button>
                    <span class="toggle-icon" id="toggle-{idx}">&#9660;</span>
                </div>
            </div>
            <div class="objective-summary">
                <div class="hypothesis-text">
                    <strong>Hypothesis:</strong> <span class="hyp-text" contenteditable="true">{hypothesis}</span>
                </div>
            </div>
            <div class="objective-details" id="details-{idx}">
                <div class="steps-container" id="steps-container-{idx}">{steps_html}</div>
                <button class="add-step-btn" onclick="addStep({idx})">+ Add Step</button>
            </div>
        </div>"""

    # Data context section
    data_context_html = ""
    if data_context:
        items = "".join(f"<li><strong>{k.replace('_', ' ').title()}:</strong> {v}</li>" for k, v in data_context.items())
        data_context_html = f"""
        <div class="info-section" style="border-left-color: #ffc107;">
            <h3 style="color: #ffc107;">📊 Data Context</h3>
            <ul>{items}</ul>
        </div>"""

    # Key considerations
    considerations_html = ""
    if considerations:
        items = "".join(f"<li>{c}</li>" for c in considerations)
        considerations_html = f"""
        <div class="info-section" style="border-left-color: #dc3545;">
            <h3 style="color: #dc3545;">⚠️ Key Considerations</h3>
            <ul>{items}</ul>
        </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Plan</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8f9fa; padding: 10px; color: #333; font-size: 14px; }}
        .container {{ max-width: 100%; margin: 0 auto; }}
        .overview-section {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .overview-section h2 {{ margin-bottom: 10px; font-size: 1.3em; display: flex; align-items: center; gap: 8px; }}
        .overview-section p {{ line-height: 1.5; font-size: 0.95em; opacity: 0.95; margin-bottom: 6px; }}
        .objectives-container {{ display: flex; flex-direction: column; gap: 12px; }}
        .objective-card {{ background: white; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); overflow: hidden; transition: all 0.2s ease; }}
        .objective-card:hover {{ box-shadow: 0 2px 6px rgba(0,0,0,0.12); }}
        .objective-header {{ padding: 12px 15px; cursor: pointer; background: linear-gradient(to right, #f8f9fa, #fff); border-left: 4px solid #28a745; display: flex; justify-content: space-between; align-items: center; gap: 10px; }}
        .objective-header:hover {{ background: #f1f3f5; }}
        .objective-title {{ display: flex; align-items: center; gap: 8px; }}
        .objective-icon {{ font-size: 1.2em; }}
        .objective-number {{ font-weight: 600; color: #28a745; font-size: 1em; }}
        .step-count {{ font-size: 0.8em; color: #6c757d; margin-left: 5px; }}
        .objective-summary {{ padding: 12px 15px; background: #f8f9fa; border-bottom: 1px solid #e9ecef; }}
        .hypothesis-text {{ font-size: 0.93em; line-height: 1.4; color: #495057; font-style: italic; }}
        .toggle-icon {{ font-size: 1.2em; color: #6c757d; transition: transform 0.3s ease; min-width: 20px; text-align: center; }}
        .toggle-icon.open {{ transform: rotate(180deg); }}
        .objective-details {{ max-height: 0; overflow: hidden; transition: max-height 0.4s ease; }}
        .objective-details.open {{ max-height: 3000px; padding: 12px 15px; border-top: 1px solid #e9ecef; }}
        .steps-container {{ display: flex; flex-direction: column; gap: 8px; }}
        .analysis-step {{ background: white; padding: 10px 12px; border-radius: 6px; border-left: 3px solid #17a2b8; }}
        .step-header {{ display: flex; gap: 10px; align-items: flex-start; }}
        .step-number {{ background: #17a2b8; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.8em; flex-shrink: 0; }}
        .step-desc {{ flex: 1; line-height: 1.4; font-size: 0.9em; color: #495057; }}
        .info-section {{ background: white; padding: 12px 15px; border-radius: 8px; margin-top: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border-left: 3px solid #667eea; }}
        .info-section h3 {{ margin-bottom: 10px; font-size: 1.1em; display: flex; align-items: center; gap: 8px; }}
        .info-section ul {{ padding-left: 20px; }}
        .info-section li {{ line-height: 1.5; font-size: 0.9em; color: #6c757d; margin-bottom: 4px; }}

        /* Editable elements */
        [contenteditable="true"] {{ border-bottom: 1px dashed #ccc; outline: none; min-width: 50px; }}
        [contenteditable="true"]:focus {{ border-bottom: 2px solid #667eea; background: #f0f4ff; border-radius: 2px; padding: 0 2px; }}

        /* Delete buttons */
        .delete-btn {{ background: none; border: none; color: #ccc; cursor: pointer; font-size: 1em; padding: 2px 6px; border-radius: 50%; transition: all 0.15s; line-height: 1; font-weight: bold; }}
        .delete-btn:hover {{ background: #dc3545; color: white; }}
        .delete-obj-btn {{ font-size: 1.1em; margin-right: 4px; }}
        .objective-actions {{ display: flex; align-items: center; gap: 4px; }}

        /* Re-run step button */
        .rerun-step-btn {{ background: none; border: 1px solid #28a745; color: #28a745; cursor: pointer; font-size: 0.75em; padding: 2px 8px; border-radius: 4px; transition: all 0.15s; flex-shrink: 0; line-height: 1.4; }}
        .rerun-step-btn:hover {{ background: #28a745; color: white; }}

        /* Add step button */
        .add-step-btn {{ margin-top: 8px; padding: 5px 14px; border: 1px dashed #17a2b8; border-radius: 4px; background: none; color: #17a2b8; cursor: pointer; font-size: 0.85em; transition: all 0.15s; }}
        .add-step-btn:hover {{ background: #17a2b8; color: white; }}

        /* Save bar */
        .save-bar {{ position: sticky; bottom: 0; background: #fff; border-top: 2px solid #667eea; padding: 8px 15px; display: none; align-items: center; justify-content: space-between; gap: 10px; z-index: 10; box-shadow: 0 -2px 8px rgba(0,0,0,0.1); }}
        .save-bar.visible {{ display: flex; }}
        .save-bar .save-btn {{ padding: 6px 20px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: 600; font-size: 0.9em; }}
        .save-bar .save-btn:hover {{ background: #218838; }}
        .save-bar .save-btn:disabled {{ background: #ccc; cursor: default; }}
        .save-bar .discard-btn {{ padding: 6px 14px; background: #f8f9fa; color: #666; border: 1px solid #ddd; border-radius: 4px; cursor: pointer; font-size: 0.85em; }}
        .save-bar .discard-btn:hover {{ background: #e9ecef; }}
        .save-bar .save-status {{ font-size: 0.85em; color: #666; }}

        /* Delete confirmation dialog overlay */
        .delete-dialog-overlay {{ position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.45); z-index: 100; display: flex; align-items: center; justify-content: center; }}
        .delete-dialog {{ background: white; border-radius: 10px; padding: 20px; max-width: 460px; width: 90%; box-shadow: 0 8px 30px rgba(0,0,0,0.25); }}
        .delete-dialog h3 {{ margin: 0 0 10px; font-size: 1.05em; color: #dc3545; display: flex; align-items: center; gap: 8px; }}
        .delete-dialog p {{ font-size: 0.9em; color: #495057; line-height: 1.5; margin: 0 0 8px; }}
        .delete-dialog .dep-list {{ background: #fff3cd; border-radius: 6px; padding: 8px 12px; margin: 8px 0 14px; font-size: 0.85em; }}
        .delete-dialog .dep-list li {{ margin-bottom: 3px; }}
        .delete-dialog .btn-row {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; margin-top: 14px; }}
        .delete-dialog button {{ padding: 7px 16px; border-radius: 5px; border: 1px solid #ddd; cursor: pointer; font-size: 0.85em; font-weight: 500; }}
        .delete-dialog .btn-cancel {{ background: #f8f9fa; color: #495057; }}
        .delete-dialog .btn-cancel:hover {{ background: #e9ecef; }}
        .delete-dialog .btn-proceed {{ background: #dc3545; color: white; border-color: #dc3545; }}
        .delete-dialog .btn-proceed:hover {{ background: #c82333; }}
        .delete-dialog .btn-regen {{ background: #667eea; color: white; border-color: #667eea; }}
        .delete-dialog .btn-regen:hover {{ background: #5a6fd6; }}
        .delete-dialog .regen-status {{ margin-top: 10px; font-size: 0.85em; color: #667eea; display: none; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="overview-section">
            <h2>📋 {title}</h2>
            <p><strong>Research Question:</strong> {rq}</p>
            {'<p>' + intent + '</p>' if intent else ''}
        </div>
        <div class="objectives-container">{objective_cards}</div>
        {data_context_html}
        {considerations_html}
    </div>
    <div class="save-bar" id="save-bar">
        <span class="save-status" id="save-status">Unsaved changes</span>
        <div style="display:flex; gap:8px;">
            <button class="discard-btn" onclick="discardChanges()">Discard</button>
            <button class="save-btn" id="save-btn" onclick="saveChanges()">Save Changes</button>
        </div>
    </div>
    <script>
        const TASK_ID = '{task_id or "default"}';
        let dirty = false;

        function markDirty() {{
            if (!dirty) {{
                dirty = true;
                document.getElementById('save-bar').classList.add('visible');
                document.getElementById('save-status').textContent = 'Unsaved changes';
            }}
        }}

        function toggleObjective(idx) {{
            const details = document.getElementById(`details-${{idx}}`);
            const toggle = document.getElementById(`toggle-${{idx}}`);
            details.classList.toggle('open');
            toggle.classList.toggle('open');
        }}

        function deleteObjective(e, idx) {{
            e.stopPropagation();
            const card = document.getElementById(`objective-${{idx}}`);
            if (card) {{
                card.remove();
                markDirty();
            }}
        }}

        // Find steps that depend on a given step (via depends_on or input_nodes)
        function findDependents(objIdx, stepNum) {{
            const container = document.getElementById(`steps-container-${{objIdx}}`);
            if (!container) return [];
            const dependents = [];
            const targetStep = container.querySelector(`.analysis-step[data-step="${{stepNum}}"]`);
            const targetOutput = targetStep ? (targetStep.dataset.outputNode || '') : '';

            container.querySelectorAll('.analysis-step').forEach(stepEl => {{
                const sNum = parseInt(stepEl.dataset.step);
                if (sNum === stepNum) return;

                // Check depends_on
                let deps = [];
                try {{ deps = JSON.parse(stepEl.dataset.dependsOn || '[]'); }} catch(e) {{}}
                if (deps.includes(stepNum)) {{
                    dependents.push({{ num: sNum, desc: stepEl.querySelector('.step-desc').textContent.trim(), reason: 'depends_on' }});
                    return;
                }}

                // Check if this step's input_nodes include the target's output_node
                if (targetOutput) {{
                    let inputs = [];
                    try {{ inputs = JSON.parse(stepEl.dataset.inputNodes || '[]'); }} catch(e) {{}}
                    if (inputs.includes(targetOutput)) {{
                        dependents.push({{ num: sNum, desc: stepEl.querySelector('.step-desc').textContent.trim(), reason: 'uses output node: ' + targetOutput }});
                    }}
                }}
            }});
            return dependents;
        }}

        function deleteStep(e, objIdx, stepIdx) {{
            e.stopPropagation();
            const stepEl = e.target.closest('.analysis-step');
            if (!stepEl) return;
            const stepNum = parseInt(stepEl.dataset.step);

            // Check for dependent steps
            const dependents = findDependents(objIdx, stepNum);

            if (dependents.length === 0) {{
                // No dependents — delete directly
                stepEl.remove();
                renumberSteps(objIdx);
                markDirty();
                return;
            }}

            // Show confirmation dialog
            showDeleteDialog(objIdx, stepNum, stepEl, dependents);
        }}

        function showDeleteDialog(objIdx, stepNum, stepEl, dependents) {{
            // Remove any existing dialog
            const old = document.querySelector('.delete-dialog-overlay');
            if (old) old.remove();

            const stepDesc = stepEl.querySelector('.step-desc').textContent.trim();
            const depItems = dependents.map(d => `<li><strong>Step ${{d.num}}:</strong> ${{d.desc}} <em style="color:#856404;">(${{d.reason}})</em></li>`).join('');

            const overlay = document.createElement('div');
            overlay.className = 'delete-dialog-overlay';
            overlay.innerHTML = `
                <div class="delete-dialog">
                    <h3>&#9888;&#65039; Step Has Dependents</h3>
                    <p>Deleting <strong>Step ${{stepNum}}: ${{stepDesc}}</strong> will affect these steps that depend on its output:</p>
                    <div class="dep-list"><ul>${{depItems}}</ul></div>
                    <p style="font-size:0.85em; color:#6c757d;">Choose how to proceed:</p>
                    <div class="btn-row">
                        <button class="btn-cancel" onclick="this.closest('.delete-dialog-overlay').remove()">Cancel</button>
                        <button class="btn-proceed" id="dlg-proceed">Delete Anyway</button>
                        <button class="btn-regen" id="dlg-regen">Delete &amp; Regenerate Plan</button>
                    </div>
                    <div class="regen-status" id="dlg-regen-status">Regenerating plan...</div>
                </div>
            `;
            document.body.appendChild(overlay);

            // Close on overlay background click
            overlay.addEventListener('click', function(ev) {{
                if (ev.target === overlay) overlay.remove();
            }});

            // Proceed without regeneration
            document.getElementById('dlg-proceed').onclick = function() {{
                stepEl.remove();
                renumberSteps(objIdx);
                markDirty();
                overlay.remove();
            }};

            // Delete & regenerate plan via LLM
            document.getElementById('dlg-regen').onclick = function() {{
                const regenBtn = this;
                regenBtn.disabled = true;
                regenBtn.textContent = 'Regenerating...';
                document.getElementById('dlg-regen-status').style.display = '';

                // Remove the step from DOM first
                stepEl.remove();
                renumberSteps(objIdx);

                // Collect the plan with the step removed
                const plan = collectPlan();

                // Build context about what was deleted
                const deletionContext = {{
                    deleted_step_number: stepNum,
                    deleted_step_description: stepDesc,
                    deleted_from_objective: objIdx,
                    affected_steps: dependents.map(d => ({{ step_number: d.num, description: d.desc, reason: d.reason }}))
                }};

                // Send to parent for LLM regeneration
                window.parent.postMessage({{
                    type: 'research_plan_regen_after_delete',
                    plan: plan,
                    deletion_context: deletionContext
                }}, '*');

                // Listen for regeneration result
                function onRegenResult(ev) {{
                    if (!ev.data || ev.data.type !== 'plan_regen_result') return;
                    window.removeEventListener('message', onRegenResult);
                    overlay.remove();
                    if (ev.data.success) {{
                        // Reload iframe to show regenerated plan
                        window.location.reload();
                    }} else {{
                        alert('Regeneration failed: ' + (ev.data.error || 'Unknown error'));
                        markDirty();
                    }}
                }}
                window.addEventListener('message', onRegenResult);
            }};
        }}

        function renumberSteps(objIdx) {{
            const container = document.getElementById(`steps-container-${{objIdx}}`);
            if (!container) return;
            const steps = container.querySelectorAll('.analysis-step');
            const oldToNew = {{}};

            // Build old→new mapping and renumber
            steps.forEach((step, i) => {{
                const oldNum = parseInt(step.dataset.step);
                const newNum = i + 1;
                oldToNew[oldNum] = newNum;
                step.dataset.step = newNum;
                const numEl = step.querySelector('.step-number');
                if (numEl) numEl.textContent = newNum;
            }});

            // Update depends_on arrays: remap to new numbers, remove deleted refs
            steps.forEach(step => {{
                let deps = [];
                try {{ deps = JSON.parse(step.dataset.dependsOn || '[]'); }} catch(e) {{}}
                const newDeps = deps.map(d => oldToNew[d]).filter(d => d !== undefined);
                step.dataset.dependsOn = JSON.stringify(newDeps);
            }});

            const countEl = document.getElementById(`step-count-${{objIdx}}`);
            if (countEl) countEl.textContent = `(${{steps.length}} steps)`;
        }}

        function addStep(objIdx) {{
            const container = document.getElementById(`steps-container-${{objIdx}}`);
            if (!container) return;
            const existingSteps = container.querySelectorAll('.analysis-step');
            const newIdx = existingSteps.length + 1;
            const stepDiv = document.createElement('div');
            stepDiv.className = 'analysis-step';
            stepDiv.dataset.obj = objIdx;
            stepDiv.dataset.step = newIdx;
            stepDiv.dataset.dependsOn = '[]';
            stepDiv.innerHTML = `
                <div class="step-header">
                    <span class="step-number">${{newIdx}}</span>
                    <span class="step-desc" contenteditable="true">New step description</span>
                    <button class="delete-btn delete-step-btn" onclick="deleteStep(event, ${{objIdx}}, ${{newIdx}})" title="Delete step">x</button>
                </div>`;
            container.appendChild(stepDiv);
            // Focus the new step text
            const descEl = stepDiv.querySelector('.step-desc');
            if (descEl) {{ descEl.focus(); selectAll(descEl); }}
            renumberSteps(objIdx);
            markDirty();
        }}

        function selectAll(el) {{
            const range = document.createRange();
            range.selectNodeContents(el);
            const sel = window.getSelection();
            sel.removeAllRanges();
            sel.addRange(range);
        }}

        // Collect the current plan state from the DOM
        function collectPlan() {{
            const objectives = [];
            document.querySelectorAll('.objective-card').forEach(card => {{
                const objNum = parseInt(card.dataset.obj);
                const titleEl = card.querySelector('.obj-title-text');
                const hypEl = card.querySelector('.hyp-text');
                const steps = [];
                card.querySelectorAll('.analysis-step').forEach((stepEl, i) => {{
                    let dependsOn = [];
                    try {{ dependsOn = JSON.parse(stepEl.dataset.dependsOn || '[]'); }} catch(e) {{}}
                    let inputNodes = [];
                    try {{ inputNodes = JSON.parse(stepEl.dataset.inputNodes || '[]'); }} catch(e) {{}}
                    const outputNode = stepEl.dataset.outputNode || '';
                    steps.push({{
                        step_number: i + 1,
                        description: stepEl.querySelector('.step-desc').textContent.trim(),
                        depends_on: dependsOn,
                        input_nodes: inputNodes,
                        output_node: outputNode
                    }});
                }});
                objectives.push({{
                    number: objNum,
                    title: titleEl ? titleEl.textContent.trim() : '',
                    hypothesis: hypEl ? hypEl.textContent.trim() : '',
                    steps: steps
                }});
            }});
            return {{ objectives: objectives, task_id: TASK_ID }};
        }}

        function saveChanges() {{
            const btn = document.getElementById('save-btn');
            const status = document.getElementById('save-status');
            btn.disabled = true;
            btn.textContent = 'Saving...';
            status.textContent = 'Saving...';

            const plan = collectPlan();
            // Post to parent window
            window.parent.postMessage({{ type: 'research_plan_edit', plan: plan }}, '*');
        }}

        // Listen for save confirmation from parent
        window.addEventListener('message', function(e) {{
            if (e.data && e.data.type === 'plan_save_result') {{
                const btn = document.getElementById('save-btn');
                const status = document.getElementById('save-status');
                if (e.data.success) {{
                    dirty = false;
                    document.getElementById('save-bar').classList.remove('visible');
                    btn.disabled = false;
                    btn.textContent = 'Save Changes';
                }} else {{
                    status.textContent = 'Save failed: ' + (e.data.error || 'Unknown error');
                    btn.disabled = false;
                    btn.textContent = 'Retry Save';
                }}
            }}
        }});

        function discardChanges() {{
            // Reload the iframe to reset all changes
            window.location.reload();
        }}

        function rerunStep(e, objIdx, stepIdx) {{
            e.stopPropagation();
            // Save any edits first, then notify parent to re-run this step
            const plan = collectPlan();
            const btn = e.target;
            btn.disabled = true;
            btn.textContent = '...';

            // Post edit-and-rerun message to parent window
            window.parent.postMessage({{
                type: 'research_plan_step_rerun',
                plan: plan,
                objective_index: objIdx,
                step_index: stepIdx
            }}, '*');
        }}

        // Listen for rerun confirmation from parent
        window.addEventListener('message', function(e) {{
            if (e.data && e.data.type === 'step_rerun_started') {{
                // Reset all rerun buttons
                document.querySelectorAll('.rerun-step-btn').forEach(btn => {{
                    btn.disabled = false;
                    btn.innerHTML = '&#9654;';
                }});
            }}
        }});

        // Track edits on contenteditable elements
        document.addEventListener('input', function(e) {{
            if (e.target.getAttribute('contenteditable') === 'true') {{
                markDirty();
            }}
        }});

        // Auto-expand first objective
        window.addEventListener('DOMContentLoaded', function() {{
            const d = document.getElementById('details-1');
            const t = document.getElementById('toggle-1');
            if (d) {{ d.classList.add('open'); }}
            if (t) {{ t.classList.add('open'); }}
        }});
    </script>
</body>
</html>"""
    return html


def save_research_plan_visualization(research_plan: Dict[str, Any], output_dir: str, task_id: str = None) -> dict:
    """
    Generate and save both plan HTML and flowchart HTML.

    Returns:
        Dict with 'plan_path' and 'flowchart_path'
    """
    viz_dir = os.path.join(output_dir, "research_plan_viz")
    os.makedirs(viz_dir, exist_ok=True)

    suffix = f"_{task_id}" if task_id else ""

    # Save raw plan HTML
    plan_html = generate_research_plan_html(research_plan, task_id)
    plan_path = os.path.join(viz_dir, f"research_plan{suffix}.html")
    with open(plan_path, 'w', encoding='utf-8') as f:
        f.write(plan_html)
    if task_id:
        _persist_artifact(task_id, "research_plan", plan_html)

    # Save interactive overview HTML (SVG-based with popup details)
    from utils.agm_helper import show_plan_overview_popup
    flowchart_path = os.path.join(viz_dir, f"research_plan_flowchart{suffix}.html")
    show_plan_overview_popup(research_plan, output_path=flowchart_path)
    if task_id:
        try:
            with open(flowchart_path, 'r', encoding='utf-8') as f:
                _persist_artifact(task_id, "research_plan_flowchart", f.read())
        except Exception:
            pass

    # Save ArcGIS Model Builder-style geoprocessing workflow HTML
    geoprocessing_path = None
    try:
        from utils.agm_helper import show_geoprocessing_workflow_plan_overview_popup_new
        geoprocessing_path = os.path.join(viz_dir, f"research_plan_geoprocessing{suffix}.html")
        show_geoprocessing_workflow_plan_overview_popup_new(research_plan, output_path=geoprocessing_path)
        if task_id:
            with open(geoprocessing_path, 'r', encoding='utf-8') as f:
                _persist_artifact(task_id, "research_plan_geoprocessing", f.read())
    except Exception as e:
        print(f"Warning: Could not generate geoprocessing workflow: {e}")

    return {'plan_path': plan_path, 'flowchart_path': flowchart_path, 'geoprocessing_path': geoprocessing_path}



def create_research_plan_flowchart_interactive(structured_plan, output_path="research_plan_flowchart.html"):
    """
    Create FULLY INTERACTIVE flowchart with draggable nodes using vis.js.
    Features: Drag nodes in any direction, zoom, pan

    Args:
        structured_plan: Dictionary with research plan structure
        output_path: Path to save HTML file

    Returns:
        Path to the saved flowchart
    """
    # Build nodes and edges with manual positioning
    nodes = []
    edges = []
    node_id = 0

    # Title node (centered at top)
    title = structured_plan.get('title', 'Research Plan')
    nodes.append({
        'id': node_id,
        'label': title,
        'x': 0,
        'y': 0,
        'color': {'background': '#ADD8E6', 'border': '#000'},
        'font': {'size': 16, 'bold': True},
        'shape': 'box',
        'fixed': False  # Allow movement
    })
    title_id = node_id
    node_id += 1

    # Objectives and steps
    objectives = structured_plan.get('objectives', [])
    y_offset = 150

    for obj_idx, obj in enumerate(objectives):
        obj_title = f"Obj {obj['number']}\n{obj['title'][:40]}..."
        obj_y = y_offset * (obj_idx + 1)

        obj_node_id = node_id
        nodes.append({
            'id': obj_node_id,
            'label': obj_title,
            'x': 0,
            'y': obj_y,
            'color': {'background': '#90EE90', 'border': '#000'},
            'shape': 'box',
            'font': {'size': 12},
            'fixed': False
        })
        edges.append({'from': title_id, 'to': obj_node_id, 'arrows': 'to'})
        node_id += 1

        # Steps (spread horizontally)
        steps = obj.get('steps', [])
        num_steps = len(steps)
        step_spacing = 200
        start_x = -(num_steps - 1) * step_spacing / 2

        for i, step in enumerate(steps):
            step_text = step.get("description", str(step)) if isinstance(step, dict) else str(step)
            step_label = f"{chr(97+i)}. {step_text[:40]}..."
            step_x = start_x + (i * step_spacing)
            step_y = obj_y + 120

            step_node_id = node_id
            nodes.append({
                'id': step_node_id,
                'label': step_label,
                'x': step_x,
                'y': step_y,
                'color': {'background': '#FFFFE0', 'border': '#000'},
                'shape': 'box',
                'font': {'size': 10},
                'fixed': False
            })
            edges.append({'from': obj_node_id, 'to': step_node_id, 'arrows': 'to'})
            node_id += 1

    # Create HTML with vis.js
    import json as json_module

    nodes_json = json_module.dumps(nodes)
    edges_json = json_module.dumps(edges)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Research Plan - Interactive Flowchart</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        #mynetwork {{
            width: 100%;
            height: 800px;
            border: 1px solid lightgray;
        }}
        h1 {{ text-align: center; font-family: Arial; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div id="mynetwork"></div>
    <script>
        var nodes = new vis.DataSet({nodes_json});
        var edges = new vis.DataSet({edges_json});

        var container = document.getElementById('mynetwork');
        var data = {{ nodes: nodes, edges: edges }};

        var options = {{
            physics: {{ enabled: false }},  // No physics - manual positioning
            interaction: {{
                dragNodes: true,     // Drag nodes freely
                dragView: true,      // Pan canvas
                zoomView: true       // Zoom with scroll
            }},
            edges: {{ smooth: {{ type: 'cubicBezier' }} }}
        }};

        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✓ Interactive flowchart saved to: {output_path}")
    print(f"  Features: Drag nodes, zoom, pan, physics simulation")

    return output_path

