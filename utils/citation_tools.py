"""
Multi-source citation lookup with cross-provider merging.

Sources (all free, no API key required):
  1. Semantic Scholar — semantic search, abstracts, citation counts
  2. OpenAlex        — broad coverage (250M+ works), concept tagging
  3. Crossref        — authoritative DOI / venue / year metadata

Public API (unchanged):
  - search_papers(query, limit, year_range=None, sources=None)
  - format_authors_inline(authors)
  - format_apa_reference(paper, ref_num)

Unified paper dict schema:
  {
    paper_id, title, authors (list[str]), year, venue, doi,
    citation_count, abstract, sources (list[str]), concepts (list[str])
  }

Merging: papers are deduplicated by DOI (case-insensitive) first, then by a
normalized title hash for entries without a DOI. When the same paper is
returned by multiple sources, fields are merged with per-field preference
(e.g. abstract: S2 > OpenAlex; venue: Crossref > S2 > OpenAlex).
"""

import os
import json
import time
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
import requests


# ── Endpoints ───────────────────────────────────────────────────────────────
_S2_SEARCH       = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_FIELDS       = "title,authors,year,venue,externalIds,abstract,citationCount,publicationVenue"
_OPENALEX_SEARCH = "https://api.openalex.org/works"
_CROSSREF_SEARCH = "https://api.crossref.org/works"

# Identify politely to both OpenAlex and Crossref (they request a mailto).
_MAILTO = os.environ.get("AGM_CITATION_MAILTO", "agm@local")


# ── Per-source throttles ────────────────────────────────────────────────────
_last_request_time = {"s2": 0.0, "openalex": 0.0, "crossref": 0.0}
_MIN_INTERVAL = {"s2": 1.2, "openalex": 0.15, "crossref": 0.15}


def _throttle(source):
    now = time.time()
    gap = _MIN_INTERVAL.get(source, 0.2)
    elapsed = now - _last_request_time.get(source, 0.0)
    if elapsed < gap:
        time.sleep(gap - elapsed)
    _last_request_time[source] = time.time()


# ── Per-source disk cache ───────────────────────────────────────────────────
_CACHE_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".cache",
)
_CACHE_TTL_SECONDS = 7 * 24 * 3600  # 7 days


def _cache_dir(source):
    d = os.path.join(_CACHE_BASE, f"citations_{source}")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_key(query, limit, year_range):
    raw = f"{query}|{limit}|{year_range or ''}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _cache_get(source, query, limit, year_range):
    try:
        path = os.path.join(_cache_dir(source), _cache_key(query, limit, year_range) + ".json")
        if not os.path.isfile(path):
            return None
        if (time.time() - os.path.getmtime(path)) > _CACHE_TTL_SECONDS:
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _cache_put(source, query, limit, year_range, papers):
    try:
        path = os.path.join(_cache_dir(source), _cache_key(query, limit, year_range) + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(papers, f)
    except Exception:
        pass


# ── Per-source circuit breakers ─────────────────────────────────────────────
_circuit_open_until = {"s2": 0.0, "openalex": 0.0, "crossref": 0.0}
_CIRCUIT_COOLDOWN = 60.0


def _circuit_is_open(source):
    return time.time() < _circuit_open_until.get(source, 0.0)


def _trip_circuit(source):
    _circuit_open_until[source] = time.time() + _CIRCUIT_COOLDOWN


def reset_circuit(source=None):
    """Reset the connection-failure circuit breaker for one source or all."""
    if source is None:
        for k in _circuit_open_until:
            _circuit_open_until[k] = 0.0
    else:
        _circuit_open_until[source] = 0.0


# ── Year-range helpers ──────────────────────────────────────────────────────

def _parse_year_range(year_range):
    """'2015-2023' → (2015, 2023) ; '2020-' → (2020, None) ; None → (None, None)."""
    if not year_range:
        return None, None
    m = re.match(r"^(\d{4})?\s*-\s*(\d{4})?$", year_range.strip())
    if not m:
        return None, None
    lo = int(m.group(1)) if m.group(1) else None
    hi = int(m.group(2)) if m.group(2) else None
    return lo, hi


# ── Source: Semantic Scholar ────────────────────────────────────────────────

def _search_semantic_scholar(query, limit, year_range):
    if _circuit_is_open("s2"):
        return {"error": "Semantic Scholar unreachable (circuit open)."}
    params = {
        "query": query,
        "fields": _S2_FIELDS,
        "limit": min(limit, 10),
    }
    if year_range:
        params["year"] = year_range

    data = None
    for attempt in range(3):
        _throttle("s2")
        try:
            resp = requests.get(_S2_SEARCH, params=params, timeout=15)
            if resp.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.ConnectionError as e:
            _trip_circuit("s2")
            return {"error": f"S2 connection error: {e}"}
        except requests.RequestException as e:
            if attempt < 2:
                time.sleep(3)
                continue
            return {"error": f"S2: {e}"}

    if data is None:
        return {"error": "S2: no response."}

    papers = []
    for p in data.get("data", []):
        authors = [a.get("name", "") for a in (p.get("authors") or [])]
        ext = p.get("externalIds") or {}
        venue = p.get("venue", "") or (p.get("publicationVenue") or {}).get("name", "")
        papers.append({
            "paper_id": p.get("paperId", ""),
            "title": p.get("title", "") or "",
            "authors": authors,
            "year": p.get("year"),
            "venue": venue,
            "doi": (ext.get("DOI") or "").lower(),
            "citation_count": p.get("citationCount", 0) or 0,
            "abstract": (p.get("abstract") or "")[:400],
            "sources": ["semantic_scholar"],
            "concepts": [],
        })
    return papers


# ── Source: OpenAlex ────────────────────────────────────────────────────────

def _reconstruct_inverted_abstract(inv_index):
    """OpenAlex returns abstracts as {word: [positions]} — reconstruct linearly."""
    if not inv_index or not isinstance(inv_index, dict):
        return ""
    positions = []
    for word, locs in inv_index.items():
        for loc in (locs or []):
            positions.append((loc, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def _search_openalex(query, limit, year_range):
    if _circuit_is_open("openalex"):
        return {"error": "OpenAlex unreachable (circuit open)."}
    lo, hi = _parse_year_range(year_range)
    filters = []
    if lo is not None:
        filters.append(f"from_publication_date:{lo}-01-01")
    if hi is not None:
        filters.append(f"to_publication_date:{hi}-12-31")

    params = {
        "search": query,
        "per-page": min(limit, 25),
        "mailto": _MAILTO,
    }
    if filters:
        params["filter"] = ",".join(filters)

    try:
        _throttle("openalex")
        resp = requests.get(_OPENALEX_SEARCH, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.ConnectionError as e:
        _trip_circuit("openalex")
        return {"error": f"OpenAlex connection error: {e}"}
    except requests.RequestException as e:
        return {"error": f"OpenAlex: {e}"}

    papers = []
    for w in data.get("results", [])[:limit]:
        authors = []
        for auth in (w.get("authorships") or []):
            name = (auth.get("author") or {}).get("display_name")
            if name:
                authors.append(name)
        doi_url = (w.get("doi") or "")
        doi = doi_url.replace("https://doi.org/", "").lower() if doi_url else ""
        host = (w.get("host_venue") or {}) or (w.get("primary_location") or {}).get("source") or {}
        venue = host.get("display_name", "") or ""
        abstract = _reconstruct_inverted_abstract(w.get("abstract_inverted_index"))
        concepts = [c.get("display_name") for c in (w.get("concepts") or [])[:6] if c.get("display_name")]
        papers.append({
            "paper_id": (w.get("id") or "").replace("https://openalex.org/", ""),
            "title": w.get("title", "") or "",
            "authors": authors,
            "year": w.get("publication_year"),
            "venue": venue,
            "doi": doi,
            "citation_count": w.get("cited_by_count", 0) or 0,
            "abstract": abstract[:400],
            "sources": ["openalex"],
            "concepts": concepts,
        })
    return papers


# ── Source: Crossref ────────────────────────────────────────────────────────

def _search_crossref(query, limit, year_range):
    if _circuit_is_open("crossref"):
        return {"error": "Crossref unreachable (circuit open)."}
    params = {
        "query": query,
        "rows": min(limit, 20),
        "mailto": _MAILTO,
    }
    lo, hi = _parse_year_range(year_range)
    # Crossref filters
    filts = []
    if lo is not None:
        filts.append(f"from-pub-date:{lo}")
    if hi is not None:
        filts.append(f"until-pub-date:{hi}")
    if filts:
        params["filter"] = ",".join(filts)

    try:
        _throttle("crossref")
        resp = requests.get(_CROSSREF_SEARCH, params=params, timeout=15,
                            headers={"User-Agent": f"AGM/1.0 (mailto:{_MAILTO})"})
        resp.raise_for_status()
        data = resp.json()
    except requests.ConnectionError as e:
        _trip_circuit("crossref")
        return {"error": f"Crossref connection error: {e}"}
    except requests.RequestException as e:
        return {"error": f"Crossref: {e}"}

    items = (data.get("message") or {}).get("items") or []
    papers = []
    for it in items[:limit]:
        authors = []
        for a in (it.get("author") or []):
            name_parts = [a.get("given", ""), a.get("family", "")]
            name = " ".join(p for p in name_parts if p).strip()
            if name:
                authors.append(name)
        # Year: pick first available from issued/created
        year = None
        for k in ("issued", "published-print", "published-online", "created"):
            dp = ((it.get(k) or {}).get("date-parts") or [[None]])[0]
            if dp and dp[0]:
                year = dp[0]
                break
        title = (it.get("title") or [""])[0]
        venue = (it.get("container-title") or [""])[0]
        doi = (it.get("DOI") or "").lower()
        papers.append({
            "paper_id": doi,   # Crossref is DOI-native
            "title": title or "",
            "authors": authors,
            "year": year,
            "venue": venue,
            "doi": doi,
            "citation_count": it.get("is-referenced-by-count", 0) or 0,
            "abstract": "",   # Crossref rarely provides abstracts
            "sources": ["crossref"],
            "concepts": [],
        })
    return papers


# ── Merge logic ─────────────────────────────────────────────────────────────

_SOURCE_FNS = {
    "semantic_scholar": _search_semantic_scholar,
    "openalex":         _search_openalex,
    "crossref":         _search_crossref,
}

# Per-field preference order when merging same-paper hits across sources.
# First match wins; falls back to whichever source had a non-empty value.
_FIELD_PREFERENCE = {
    "abstract":       ["semantic_scholar", "openalex", "crossref"],
    "venue":          ["crossref", "semantic_scholar", "openalex"],
    "year":           ["crossref", "semantic_scholar", "openalex"],
    "title":          ["crossref", "semantic_scholar", "openalex"],
    "citation_count": ["semantic_scholar", "openalex", "crossref"],
    "authors":        ["crossref", "semantic_scholar", "openalex"],
}


def _norm_title(t):
    t = (t or "").lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _merge_key(p):
    doi = (p.get("doi") or "").strip().lower()
    if doi:
        return ("doi", doi)
    nt = _norm_title(p.get("title"))
    if nt:
        return ("title", nt[:120])
    return ("id", p.get("paper_id") or str(id(p)))


def _pick(a, b, field, source_of):
    """Pick the preferred value of `field` across two merge-partners."""
    pref = _FIELD_PREFERENCE.get(field, [])
    candidates = [(source_of.get(id(a)), a.get(field)), (source_of.get(id(b)), b.get(field))]
    for src in pref:
        for s, v in candidates:
            if s == src and _non_empty(v):
                return v
    for _, v in candidates:
        if _non_empty(v):
            return v
    return a.get(field)


def _non_empty(v):
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, (list, tuple, dict)):
        return len(v) > 0
    if isinstance(v, (int, float)):
        return True
    return bool(v)


def _merge_pair(a, b, source_of):
    merged = dict(a)
    for field in ("title", "authors", "year", "venue", "doi", "abstract", "citation_count"):
        merged[field] = _pick(a, b, field, source_of)
    merged["sources"] = sorted(set((a.get("sources") or []) + (b.get("sources") or [])))
    merged["concepts"] = sorted(set((a.get("concepts") or []) + (b.get("concepts") or [])))
    # paper_id: keep the S2 id if present (stable + used elsewhere); else DOI.
    if "semantic_scholar" in merged["sources"]:
        merged["paper_id"] = a.get("paper_id") if "semantic_scholar" in (a.get("sources") or []) else b.get("paper_id")
    else:
        merged["paper_id"] = merged.get("doi") or a.get("paper_id") or b.get("paper_id") or ""
    return merged


def _merge_papers(results_by_source):
    """
    results_by_source: dict[source_name -> list[paper]].
    Returns a merged, deduped list sorted by (coverage, citation_count).
    """
    source_of = {}  # id(paper) -> source_name
    all_papers = []
    for src, papers in results_by_source.items():
        if not isinstance(papers, list):
            continue
        for p in papers:
            source_of[id(p)] = src
            all_papers.append(p)

    bucket = {}
    for p in all_papers:
        k = _merge_key(p)
        if k in bucket:
            merged = _merge_pair(bucket[k], p, source_of)
            # Transfer source_of for the merged (so subsequent merges keep per-source tracking).
            source_of[id(merged)] = source_of.get(id(bucket[k]), None)
            bucket[k] = merged
        else:
            bucket[k] = p

    merged_list = list(bucket.values())
    # Rank: multi-source hits first, then citation_count.
    merged_list.sort(
        key=lambda p: (-(len(p.get("sources") or [])), -(p.get("citation_count") or 0))
    )
    return merged_list


# ── Public API ──────────────────────────────────────────────────────────────

DEFAULT_SOURCES = ("semantic_scholar", "openalex", "crossref")


def search_papers(query, limit=5, year_range=None, sources=None):
    """
    Multi-source paper search with cross-provider merging.

    Parameters
    ----------
    query : str
        Free-text search terms.
    limit : int
        Max results to return AFTER merging. Each source is asked for up to
        ``limit`` hits (slightly more for OpenAlex/Crossref which are cheap)
        so dedup still yields enough unique papers.
    year_range : str | None
        E.g. "2015-2023" or "2020-" for open-ended.
    sources : iterable[str] | None
        Subset of ``{"semantic_scholar", "openalex", "crossref"}``. Defaults
        to all three. Pass a single-element tuple (e.g. ``("semantic_scholar",)``)
        to reproduce the old behavior.

    Returns
    -------
    list[dict]  on success — merged, ranked papers with schema documented at
                the top of this module.
    dict        on error   — ``{"error": "..."}`` (only when ALL sources failed).
    """
    srcs = tuple(sources) if sources else DEFAULT_SOURCES
    srcs = tuple(s for s in srcs if s in _SOURCE_FNS)
    if not srcs:
        return {"error": f"No valid sources. Pick from {list(_SOURCE_FNS)}."}

    # Per-source cache lookup first — avoids hitting the wire at all for
    # repeat queries within the 7-day TTL.
    results_by_source = {}
    to_fetch = []
    for src in srcs:
        cached = _cache_get(src, query, limit, year_range)
        if cached is not None:
            results_by_source[src] = cached
        else:
            to_fetch.append(src)

    # Fetch remaining sources in parallel (each is rate-limited internally).
    if to_fetch:
        def _run(src):
            try:
                res = _SOURCE_FNS[src](query, limit, year_range)
            except Exception as e:
                return src, {"error": f"{src}: {e}"}
            if isinstance(res, list):
                _cache_put(src, query, limit, year_range, res)
            return src, res

        with ThreadPoolExecutor(max_workers=len(to_fetch)) as pool:
            for src, res in pool.map(_run, to_fetch):
                results_by_source[src] = res

    # Separate lists from errors. If every source errored, return the first error.
    good = {s: r for s, r in results_by_source.items() if isinstance(r, list)}
    errors = {s: r for s, r in results_by_source.items() if isinstance(r, dict)}
    if not good:
        first_err = next(iter(errors.values()), {"error": "No results."})
        return {"error": "; ".join(f"{s}: {e.get('error','?')}" for s, e in errors.items())
                         or first_err.get("error", "No results.")}

    merged = _merge_papers(good)
    return merged[:limit]


def format_authors_inline(authors):
    """
    Format an author list for in-text citation style.

    1 author  → "Smith"
    2 authors → "Smith and Jones"
    3+ authors → "Smith et al."
    """
    if not authors:
        return "Unknown"
    last_names = []
    for name in authors:
        parts = (name or "").strip().split()
        if parts:
            last_names.append(parts[-1])
    if len(last_names) == 1:
        return last_names[0]
    if len(last_names) == 2:
        return f"{last_names[0]} and {last_names[1]}"
    return f"{last_names[0]} et al."


def format_apa_reference(paper, ref_num):
    """
    Format a paper dict into a numbered APA-style reference string.

    Example output:
        [1] Smith, J.A., Jones, B.C. (2021). Title of the paper.
            *Journal Name*. https://doi.org/10.1234/example
    """
    authors = paper.get("authors", [])
    year = paper.get("year") or "n.d."
    title = paper.get("title", "Untitled")
    venue = paper.get("venue", "")
    doi = paper.get("doi", "")

    formatted = []
    for name in authors[:6]:
        parts = (name or "").strip().split()
        if len(parts) >= 2:
            last = parts[-1]
            initials = ".".join(p[0].upper() for p in parts[:-1]) + "."
            formatted.append(f"{last}, {initials}")
        elif parts:
            formatted.append(parts[0])

    if len(authors) > 6:
        author_str = ", ".join(formatted[:6]) + ", et al."
    elif len(formatted) == 0:
        author_str = "Unknown"
    elif len(formatted) == 1:
        author_str = formatted[0]
    else:
        author_str = ", ".join(formatted[:-1]) + ", & " + formatted[-1]

    ref = f"[{ref_num}] {author_str} ({year}). {title}."
    if venue:
        ref += f" *{venue}*."
    if doi:
        ref += f" https://doi.org/{doi}"
    return ref
