"""
Skill document lookup tool for code-generation agents.

Skills are plain Markdown files under ``skills/`` with a YAML frontmatter
block that declares their metadata (id, agent, keywords, summary). The
loader auto-discovers every ``*_skill.md`` in that directory, so adding a
new skill is a one-file change — no Python edits required.

Usage
-----
    from utils.toolbase.skill_lookup_tool import build_skill_lookup_tool
    tool = build_skill_lookup_tool()

    # Or, for non-tool code-gen paths:
    from utils.toolbase.skill_lookup_tool import get_relevant_skills
    text = get_relevant_skills(step_description)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from langchain_core.tools import tool


SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "skills"
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


def _parse_frontmatter(text: str) -> Tuple[dict, str]:
    """Split a markdown file into (metadata_dict, body).

    Returns ({}, text) if no frontmatter block is present.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), m.group(2)


def _load_registry() -> Dict[str, dict]:
    """Discover all skill docs and index them by filename.

    Each entry carries: filename, path, keywords, summary, agent, body.
    Files missing frontmatter are still loaded (with empty metadata) so
    nothing silently disappears — but they won't be keyword-matchable.
    """
    registry: Dict[str, dict] = {}
    if not SKILLS_DIR.exists():
        return registry
    for path in sorted(SKILLS_DIR.glob("*_skill.md")):
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        meta, body = _parse_frontmatter(raw)
        kw = meta.get("keywords") or []
        if isinstance(kw, str):
            kw = [kw]
        registry[path.name] = {
            "filename": path.name,
            "path": path,
            "id": meta.get("id") or path.stem.replace("_skill", ""),
            "agent": meta.get("agent", ""),
            "keywords": [str(k).lower() for k in kw],
            "summary": meta.get("summary", "").strip(),
            "body": body,
            "full": raw,
        }
    return registry


def _score(query: str, entry: dict) -> int:
    """Keyword hit count for a query against one skill entry."""
    q = query.lower()
    return sum(1 for kw in entry["keywords"] if kw and kw in q)


def _match(query: str, agent: str | None = None) -> List[dict]:
    """Return matched skill entries, best-first, ties broken by filename.

    A "match" requires at least one keyword hit. If ``agent`` is given,
    restrict to entries whose frontmatter ``agent`` field equals it.
    """
    registry = _load_registry()
    entries = registry.values()
    if agent:
        entries = [e for e in entries if e.get("agent") == agent]
    scored = [(_score(query, e), e) for e in entries]
    scored = [(s, e) for s, e in scored if s > 0]
    scored.sort(key=lambda t: (-t[0], t[1]["filename"]))
    return [e for _, e in scored]


def _section(body: str, section: str) -> str:
    """Return the requested ``## <section>`` block from the markdown body,
    or an empty string if no heading matches.

    Heading match is case-insensitive and tolerates partial matches
    (e.g. ``section="pitfalls"`` matches ``## Common Pitfalls``).
    """
    if not section:
        return body
    needle = section.lower().strip()
    lines = body.splitlines()
    out: List[str] = []
    capturing = False
    for line in lines:
        if line.startswith("## "):
            heading = line[3:].strip().lower()
            if needle in heading:
                capturing = True
                out.append(line)
                continue
            if capturing:
                break  # next same-level heading ends the section
        elif capturing:
            out.append(line)
    return "\n".join(out).strip()


def _available_list(agent: str | None = None) -> str:
    """Human-readable list of available skills (for no-match fallbacks)."""
    entries = list(_load_registry().values())
    if agent:
        entries = [e for e in entries if e.get("agent") == agent]
    if not entries:
        return "  (no skill documents found)"
    lines = []
    for e in sorted(entries, key=lambda x: x["filename"]):
        name = e["id"].upper()
        summ = e["summary"] or "(no summary)"
        lines.append(f"  - {name}: {summ}")
    return "\n".join(lines)


def build_skill_lookup_tool(agent: str | None = None):
    """Return a LangChain tool that looks up skill documents.

    If ``agent`` is given (e.g. ``"modeling"`` or ``"data"``), only skills
    whose frontmatter ``agent`` matches are visible to the tool.
    """

    @tool
    def lookup_skill_doc(query: str, section: str = "") -> str:
        """Look up a skill document with descriptions and code examples for a
        specific analysis method. Use this BEFORE writing code for complex
        analyses to get correct imports, parameters, data prep, and pitfalls.

        Args:
            query: A short phrase naming the method. Examples: "gwr",
                "pearson correlation", "correlation matrix", "moran", "lisa".
            section: Optional. Return only one ``## <section>`` of the doc
                (case-insensitive, partial match). Examples: "example",
                "pitfalls", "when to use". Leave empty to return the full doc.

        Returns:
            The matched skill document(s), or a list of available skills if
            nothing matched.
        """
        matches = _match(query, agent=agent)
        if not matches:
            return (
                f"No skill document matched '{query}'.\n"
                f"Available skill documents:\n{_available_list(agent=agent)}"
            )

        if len(matches) > 1:
            # Disambiguate rather than dumping everything — keeps token use sane
            lines = [f"Multiple skills matched '{query}'. Call lookup_skill_doc again with a more specific query:"]
            for e in matches:
                lines.append(f"  - id='{e['id']}' — {e['summary'] or '(no summary)'}")
            return "\n".join(lines)

        entry = matches[0]
        body = entry["body"]
        if section:
            chunk = _section(body, section)
            if chunk:
                return chunk
            return (
                f"No section matching '{section}' in '{entry['id']}'. "
                f"Returning full document.\n\n{body}"
            )
        return body

    return lookup_skill_doc


def get_relevant_skills(step_description: str, agent: str | None = None) -> str:
    """Keyword-match skill docs against a step description. For non-tool
    code-gen paths where the LLM can't call tools. Returns concatenated
    matched bodies, or empty string if nothing matches.

    Pass ``agent="modeling"`` or ``agent="data"`` to restrict the search.
    """
    matches = _match(step_description, agent=agent)
    return "\n\n---\n\n".join(e["body"] for e in matches)
