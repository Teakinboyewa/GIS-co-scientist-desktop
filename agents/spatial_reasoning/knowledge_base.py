"""
The shared spatial knowledge base and its three text-level deliveries.

The study compares *how* spatial knowledge enters the agent, so every
condition must receive the same content. Content lives only in
spatial_kb/spatial_kb.yaml; the functions here change its form, never its
substance:

  B1  render_prompt_checklist()   principles + checklist in the system prompt
  B2  retrieve(query)             entries returned on demand by a search tool
  B3  render_skill_index() /      if/then rule files the agent loads by name
      render_skill(concept)

Every renderer emits every field of every entry it covers, and
tests/test_spatial_conditions.py checks that, so no delivery is quietly
richer than another.
"""

import hashlib
import os
import re
from functools import lru_cache

import yaml

KB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "spatial_kb", "spatial_kb.yaml")

CONCEPTS = ("scale", "dependence", "heterogeneity")
ASSUMPTION_FIELDS = ("scale_and_zoning", "independence", "global_or_local")
_REQUIRED = ("id", "concept", "assumption_field", "title", "condition",
             "implication", "diagnostic", "response")

CONCEPT_TITLES = {
    "scale": "Scale (modifiable areal unit problem)",
    "dependence": "Spatial dependence",
    "heterogeneity": "Spatial heterogeneity",
}


def _clean(text):
    return " ".join(str(text or "").split())


def _validate(kb):
    entries = kb.get("entries") or []
    if not entries:
        raise ValueError(f"{KB_PATH}: no entries")
    seen = set()
    for entry in entries:
        missing = [k for k in _REQUIRED if not entry.get(k)]
        if missing:
            raise ValueError(f"{KB_PATH}: entry {entry.get('id')!r} missing {missing}")
        if entry["id"] in seen:
            raise ValueError(f"{KB_PATH}: duplicate id {entry['id']!r}")
        seen.add(entry["id"])
        if entry["concept"] not in CONCEPTS:
            raise ValueError(f"{KB_PATH}: {entry['id']}: unknown concept {entry['concept']!r}")
        if entry["assumption_field"] not in ASSUMPTION_FIELDS:
            raise ValueError(f"{KB_PATH}: {entry['id']}: unknown assumption_field "
                             f"{entry['assumption_field']!r}")
        for key in ("name", "procedure"):
            if not (entry["diagnostic"] or {}).get(key):
                raise ValueError(f"{KB_PATH}: {entry['id']}: diagnostic.{key} missing")
        for key in ("if_detected", "if_not_detected"):
            if not (entry["response"] or {}).get(key):
                raise ValueError(f"{KB_PATH}: {entry['id']}: response.{key} missing")
    return kb


@lru_cache(maxsize=1)
def load_kb():
    with open(KB_PATH, encoding="utf-8") as f:
        return _validate(yaml.safe_load(f))


def entries(concept=None):
    items = load_kb()["entries"]
    return [e for e in items if concept is None or e["concept"] == concept]


def fingerprint():
    """sha256 of the knowledge-base file, recorded with every run."""
    with open(KB_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def entry_text(entry):
    """All of an entry's content as plain sentences. The unit of retrieval
    (B2) and the reference against which the other renderers are checked."""
    d, r = entry["diagnostic"], entry["response"]
    return "\n".join([
        f"[{entry['id']}] {entry['title']} ({CONCEPT_TITLES[entry['concept']]})",
        f"Condition: {_clean(entry['condition'])}",
        f"Implication: {_clean(entry['implication'])}",
        f"Diagnostic ({_clean(d['name'])}): {_clean(d['procedure'])}",
        f"If detected: {_clean(r['if_detected'])}",
        f"If not detected: {_clean(r['if_not_detected'])}",
    ])


# ── B1: principles and checklist in the system prompt ────────────────

def render_prompt_checklist():
    lines = [
        "**SPATIAL PRINCIPLES**",
        "Apply these principles when forming hypotheses and the analytical plan.",
        "",
    ]
    for concept in CONCEPTS:
        lines.append(f"*{CONCEPT_TITLES[concept]}*")
        for e in entries(concept):
            lines.append(f"- {e['title']}: when {_clean(e['condition'])[0].lower()}"
                         f"{_clean(e['condition'])[1:]} {_clean(e['implication'])}")
        lines.append("")
    lines.append("**SPATIAL CHECKLIST**")
    for e in entries():
        d, r = e["diagnostic"], e["response"]
        lines.append(
            f"- [ ] {e['title']} — check by {_clean(d['name'])}: {_clean(d['procedure'])} "
            f"If detected: {_clean(r['if_detected'])} "
            f"If not detected: {_clean(r['if_not_detected'])}")
    return "\n".join(lines)


# ── B2: retrieval on demand ──────────────────────────────────────────

_TOKEN = re.compile(r"[a-z0-9']+")
_STOP = frozenset("""a an and are as at be by can do does for from has have how if in
into is it its not of on or other so such than that the their them then these this
to use used using was what when where whether which with""".split())


def _tokens(text):
    # Single letters are dropped: the "I" of "Moran's I" would otherwise
    # match the pronoun in a query such as "should I use GWR".
    return [t for t in _TOKEN.findall(str(text).lower()) if len(t) > 1 and t not in _STOP]


def retrieve(query, k=3):
    """Return the k entries that best match *query*, as full entry text.

    Lexical overlap, not embeddings: deterministic, free, and identical across
    repeated runs, so run-to-run variance reflects the agent, not the index.
    Ties keep knowledge-base order.
    """
    q = set(_tokens(query))
    scored = []
    for i, e in enumerate(entries()):
        title = set(_tokens(e["title"] + " " + e["diagnostic"]["name"]))
        body = set(_tokens(entry_text(e)))
        score = 2 * len(q & title) + len(q & body)
        scored.append((-score, i, e))
    scored.sort(key=lambda t: (t[0], t[1]))
    hits = [e for s, _, e in scored if s < 0][:k]
    return hits


def build_retrieval_tool(access_log):
    """LangChain tool for B2. Each call is appended to *access_log* so the
    analysis can tell 'guidance never retrieved' from 'retrieved and ignored'."""
    from langchain_core.tools import tool

    @tool
    def retrieve_spatial_knowledge(query: str) -> str:
        """Search the spatial-analysis knowledge base (scale and the modifiable
        areal unit problem, spatial dependence, spatial heterogeneity). Returns
        the most relevant guidance: when it applies, why it matters, the
        diagnostic that settles it and how to respond."""
        hits = retrieve(query)
        access_log.append({"tool": "retrieve_spatial_knowledge", "query": query,
                           "returned": [e["id"] for e in hits]})
        if not hits:
            return "No matching guidance. Try different terms."
        return "\n\n".join(entry_text(e) for e in hits)

    return retrieve_spatial_knowledge


# ── B3: if/then skill files (NORA-style) ─────────────────────────────

SKILL_NAMES = {c: f"spatial-{c}" for c in CONCEPTS}


def render_skill(concept):
    lines = [f"# Skill: {SKILL_NAMES[concept]} — {CONCEPT_TITLES[concept]}", ""]
    for e in entries(concept):
        d, r = e["diagnostic"], e["response"]
        lines += [
            f"## {e['title']}",
            f"IF {_clean(e['condition'])}",
            f"BECAUSE {_clean(e['implication'])}",
            f"THEN run {_clean(d['name'])}: {_clean(d['procedure'])}",
            f"  IF the diagnostic detects it THEN {_clean(r['if_detected'])}",
            f"  ELSE {_clean(r['if_not_detected'])}",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def render_skill_index():
    lines = ["**AVAILABLE SPATIAL SKILLS**",
             "Load a skill with `load_spatial_skill` when it is relevant to the "
             "hypotheses or plan you are writing."]
    for concept in CONCEPTS:
        titles = "; ".join(e["title"] for e in entries(concept))
        lines.append(f"- {SKILL_NAMES[concept]}: decision rules for "
                     f"{CONCEPT_TITLES[concept].lower()} ({titles})")
    return "\n".join(lines)


def build_skill_tool(access_log):
    from langchain_core.tools import tool

    by_name = {name: concept for concept, name in SKILL_NAMES.items()}

    @tool
    def load_spatial_skill(name: str) -> str:
        """Load a spatial skill file by name (see AVAILABLE SPATIAL SKILLS).
        Returns its if/then decision rules."""
        concept = by_name.get(name.strip())
        access_log.append({"tool": "load_spatial_skill", "name": name,
                           "returned": [e["id"] for e in entries(concept)] if concept else []})
        if not concept:
            return f"Unknown skill {name!r}. Available: {', '.join(by_name)}"
        return render_skill(concept)

    return load_spatial_skill


# ── What the planner receives under each delivery ─────────────────────

def planner_delivery(kb_delivery, access_log):
    """(system-prompt addition, extra tools) for the hypothesis and planning
    agent. 'none' adds nothing, which keeps B0 byte-identical to the
    published system."""
    if kb_delivery == "none":
        return "", []
    if kb_delivery == "prompt":
        return render_prompt_checklist(), []
    if kb_delivery == "retrieval":
        return ("**SPATIAL KNOWLEDGE BASE**\nUse `retrieve_spatial_knowledge` to look up "
                "spatial-analysis guidance while forming hypotheses and the plan.",
                [build_retrieval_tool(access_log)])
    if kb_delivery == "skill":
        return render_skill_index(), [build_skill_tool(access_log)]
    raise ValueError(f"Unknown kb_delivery {kb_delivery!r}")


def compose_planner_prompt(base_prompt, kb_delivery, access_log):
    """(full system prompt, extra tools) for the planner. The planner and
    the Spatial Conditions page both call this, so what the page shows is
    what the planner is sent."""
    addition, tools = planner_delivery(kb_delivery, access_log)
    if not addition:
        return base_prompt, tools
    return f"{base_prompt}\n\n{addition}\n", tools
