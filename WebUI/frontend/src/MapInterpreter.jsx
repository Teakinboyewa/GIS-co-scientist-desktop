import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, getApiKey, saveApiKey } from "./api.js";
import { navigate } from "./nav.js";

const mapIdFromPath = () => window.location.pathname.match(/^\/map-interpreter\/([^/]+)/)?.[1] || null;

// How interpretations and answers are produced: the proposed framework, or the baseline it is compared with
export const METHODS = {
  framework: { label: "Framework", desc: "Measure the map, then answer only from the measured table" },
  model_only: { label: "Model only", desc: "Baseline: the model looks at the image, nothing is measured" },
};
const METHOD_STORAGE = "mi_method";

export default function MapInterpreter() {
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("");
  const [method, setMethod] = useState(() => {
    try {
      return METHODS[localStorage.getItem(METHOD_STORAGE)] ? localStorage.getItem(METHOD_STORAGE) : "framework";
    } catch {
      return "framework";
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(METHOD_STORAGE, method);
    } catch {
      /* storage blocked */
    }
  }, [method]);
  const [maps, setMaps] = useState([]);
  const [current, setCurrent] = useState(null);
  const [reading, setReading] = useState(null); // file name while Stages 1-3 run
  const [error, setError] = useState(null);
  const [needsKey, setNeedsKey] = useState(false);
  const [interpretations, setInterpretations] = useState({}); // "map id|method" -> {loading, data, error}
  const [threads, setThreads] = useState({}); // map id -> [{question, loading, data, error}]
  const [measuring, setMeasuring] = useState({}); // map id -> true while Stages 1-3 run

  const [errorLog, setErrorLog] = useState([]);
  const handleError = useCallback((e) => {
    if (e.status === 401) setNeedsKey(true);
    setError(e.message);
    setErrorLog(e.log || []);
  }, []);

  // A saved session comes back with the map: its interpretations (one per method) and its question thread
  const restore = useCallback((entry) => {
    setCurrent(entry);
    setInterpretations((s) => {
      const next = { ...s };
      for (const [m, data] of Object.entries(entry.interpretations || {})) {
        if (!next[`${entry.id}|${m}`]?.loading) next[`${entry.id}|${m}`] = { data };
      }
      return next;
    });
    setThreads((t) =>
      (t[entry.id] || []).some((item) => item.loading)
        ? t // a question is being answered right now; keep the live thread
        : { ...t, [entry.id]: (entry.thread || []).map((r) => ({ question: r.question, method: r.requested_method || r.method, data: r })) }
    );
  }, []);
  const load = useCallback((id) => api.getMap(id).then(restore).catch(handleError), [restore, handleError]);

  useEffect(() => {
    api.models().then((r) => {
      setModels(r.models);
      setModel(r.default);
    });
    api.listMaps().then((r) => setMaps(r.maps)).catch(() => {});
    const id = mapIdFromPath();
    if (id) load(id);
  }, [load]);

  useEffect(() => {
    const onPop = () => {
      const id = mapIdFromPath();
      if (!id) setCurrent(null);
      else if (id !== current?.id) load(id);
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, [current, load]);

  // Keep the open map and the sidebar in step with the server's copy (measuring changes its status)
  const updateEntry = useCallback((entry) => {
    if (!entry) return;
    setCurrent((c) => (c?.id === entry.id ? entry : c));
    setMaps((list) => list.map((m) => (m.id === entry.id ? entry : m)));
  }, []);

  const measure = (id) => {
    setError(null);
    setMeasuring((s) => ({ ...s, [id]: true }));
    api
      .measure(id, model)
      .then(updateEntry)
      .catch(handleError)
      .finally(() => setMeasuring((s) => ({ ...s, [id]: false })));
  };

  // one interpretation per map and method, so both can be generated and compared
  const interpret = (id) => {
    const key = `${id}|${method}`;
    setInterpretations((s) => ({ ...s, [key]: { loading: true } }));
    api
      .interpret(id, model, method)
      .then((data) => {
        updateEntry(data.map);
        setInterpretations((s) => ({ ...s, [key]: { data } }));
      })
      .catch((e) => {
        if (e.status === 401) setNeedsKey(true);
        setInterpretations((s) => ({ ...s, [key]: { error: e.message } }));
      });
  };

  // Uploading only stores and shows the map; nothing is generated until a button asks for it
  const addMap = useCallback(
    async (file) => {
      setError(null);
      setReading(file.name);
      try {
        const entry = await api.addMap(file);
        setMaps((m) => [...m, entry]);
        setCurrent(entry);
        window.history.pushState({}, "", `/map-interpreter/${entry.id}`);
      } catch (e) {
        handleError(e);
      } finally {
        setReading(null);
      }
    },
    [handleError]
  );

  const open = (entry) => {
    setError(null);
    window.history.pushState({}, "", `/map-interpreter/${entry.id}`);
    load(entry.id);
  };

  // A new session starts on the blank upload page; it is created when a map is added
  const newSession = () => {
    setError(null);
    setCurrent(null);
    window.history.pushState({}, "", "/map-interpreter");
  };

  const rename = (entry, title) =>
    api
      .renameMap(entry.id, title)
      .then((s) => {
        setMaps((list) => list.map((m) => (m.id === s.id ? { ...m, title: s.title } : m)));
        setCurrent((c) => (c?.id === s.id ? { ...c, title: s.title } : c));
      })
      .catch(handleError);

  const remove = (entry) => {
    if (!window.confirm(`Remove the session "${entry.title || entry.name}" and everything generated for it?`)) return;
    api
      .deleteMap(entry.id)
      .then(() => {
        setMaps((list) => list.filter((m) => m.id !== entry.id));
        if (current?.id === entry.id) {
          setCurrent(null);
          window.history.pushState({}, "", "/map-interpreter");
        }
      })
      .catch(handleError);
  };

  const ask = (question) => {
    const id = current.id;
    const index = (threads[id] || []).length;
    const update = (patch) =>
      setThreads((t) => {
        const list = [...(t[id] || [])];
        list[index] = { ...list[index], ...patch };
        return { ...t, [id]: list };
      });
    update({ question, method, loading: true });
    api
      .ask(id, question, model, method)
      .then((data) => {
        updateEntry(data.map);
        update({ loading: false, data });
      })
      .catch((e) => {
        if (e.status === 401) setNeedsKey(true);
        update({ loading: false, error: e.message });
      });
  };

  return (
    <div className="mi">
      <header className="mi-bar">
        <button className="link-btn" onClick={() => navigate("/home")}>← Agents</button>
        <h1>Map Interpreter</h1>
        <div className="mi-bar-right">
          <div className="method" role="radiogroup" aria-label="Method">
            <span className="method-label">Method</span>
            {Object.entries(METHODS).map(([key, m]) => (
              <button
                key={key}
                role="radio"
                aria-checked={method === key}
                className={`method-opt ${method === key ? "on" : ""} m-${key}`}
                title={m.desc}
                onClick={() => setMethod(key)}
              >
                {m.label}
              </button>
            ))}
          </div>
          <label className="field-inline">
            <span>Model</span>
            <select value={model} onChange={(e) => setModel(e.target.value)} id="mi-model">
              {models.map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </label>
          <KeyStatus forceOpen={needsKey} onSaved={() => { setNeedsKey(false); setError(null); }} />
        </div>
      </header>

      <div className="mi-body">
        <aside className="mi-side">
          <button className={`btn btn-primary btn-block ${!current && !reading ? "is-current" : ""}`} onClick={newSession}>
            + New session
          </button>
          <div className="side-label">Sessions</div>
          {maps.length === 0 && (
            <p className="muted small">Each map you add becomes a session, saved with everything generated for it.</p>
          )}
          <ul className="map-list">
            {[...maps].reverse().map((m) => (
              <SessionItem
                key={m.id}
                session={m}
                active={current?.id === m.id}
                onOpen={() => open(m)}
                onRemove={() => remove(m)}
                onRename={(title) => rename(m, title)}
              />
            ))}
          </ul>
        </aside>

        <main className="mi-main">
          {error && (
            <div className="banner banner-error" role="alert">
              <div className="banner-body">
                <span>{error}</span>
                {errorLog.length > 0 && (
                  <details>
                    <summary>Steps before it failed</summary>
                    <LogView lines={errorLog} />
                  </details>
                )}
              </div>
              <button className="link-btn" onClick={() => setError(null)}>Dismiss</button>
            </div>
          )}
          {reading ? (
            <Reading name={reading} />
          ) : current ? (
            <Workspace
              key={current.id}
              entry={current}
              measuring={!!measuring[current.id]}
              onMeasure={() => measure(current.id)}
              method={method}
              interpretation={interpretations[`${current.id}|${method}`]}
              onInterpret={() => interpret(current.id)}
              thread={threads[current.id] || []}
              onAsk={ask}
            />
          ) : (
            <Empty onFile={addMap} />
          )}
        </main>
      </div>
      <LogDock
        entry={current}
        active={
          !!current &&
          (!!measuring[current.id] ||
            Object.keys(METHODS).some((m) => interpretations[`${current.id}|${m}`]?.loading) ||
            (threads[current.id] || []).some((t) => t.loading))
        }
      />
    </div>
  );
}

// ---------------------------------------------------------------- upload

function useFilePicker(onFile) {
  const input = useRef(null);
  const pick = () => input.current?.click();
  const element = (
    <input
      ref={input}
      id="mi-file"
      type="file"
      accept=".png,.jpg,.jpeg,.webp,image/png,image/jpeg,image/webp"
      hidden
      onChange={(e) => {
        const file = e.target.files?.[0];
        e.target.value = "";
        if (file) onFile(file);
      }}
    />
  );
  return [pick, element];
}

// One session in the sidebar: its title (renamable), date and status
function SessionItem({ session, active, onOpen, onRemove, onRename }) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(session.title || session.name);
  const title = session.title || session.name;
  const date = session.created
    ? new Date(session.created * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" })
    : "";
  const finish = (save) => {
    setEditing(false);
    if (save && value.trim() !== title) onRename(value.trim());
    else setValue(title);
  };
  return (
    <li className="map-row">
      {editing ? (
        <div className={`map-item active`}>
          <img src={session.image_url} alt="" />
          <input
            id={`rename-${session.id}`}
            className="rename-input"
            value={value}
            autoFocus
            maxLength={120}
            onChange={(e) => setValue(e.target.value)}
            onBlur={() => finish(true)}
            onKeyDown={(e) => {
              if (e.key === "Enter") finish(true);
              if (e.key === "Escape") finish(false);
            }}
            aria-label="Session title"
          />
        </div>
      ) : (
        <button className={`map-item ${active ? "active" : ""}`} onClick={onOpen} onDoubleClick={() => setEditing(true)} title={title}>
          <img src={session.image_url} alt="" />
          <span className="map-item-text">
            <span className="map-item-name">{title}</span>
            <span className="map-item-meta">
              {date && <span>{date}</span>}
              <StatusChip measured={session.measured} />
            </span>
          </span>
        </button>
      )}
      {!editing && (
        <span className="map-actions">
          <button className="map-action" onClick={() => setEditing(true)} title="Rename" aria-label={`Rename ${title}`}>✎</button>
          <button className="map-action map-action-danger" onClick={onRemove} title="Remove" aria-label={`Remove ${title}`}>×</button>
        </span>
      )}
    </li>
  );
}

function Empty({ onFile }) {
  const [pick, input] = useFilePicker(onFile);
  const [over, setOver] = useState(false);
  return (
    <div className="empty">
      <div
        className={`drop ${over ? "drop-over" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          const file = e.dataTransfer.files?.[0];
          if (file) onFile(file);
        }}
      >
        <svg viewBox="0 0 24 24" className="drop-icon" aria-hidden="true">
          <path d="M3 6l6-2.5 6 2.5 6-2.5v14.5l-6 2.5-6-2.5-6 2.5z" />
          <path d="M9 3.5v14.5M15 6v14.5" />
        </svg>
        <h2>Add a thematic map</h2>
        <p>Drop a PNG, JPG or WebP here, or</p>
        <button className="btn btn-primary" onClick={pick}>Choose a file</button>
        {input}
      </div>
      <div className="how">
        <h3>What you can run</h3>
        <p className="muted small">The map opens right away. Each step below runs only when you click its button.</p>
        <ol>
          <li><b>Anatomy.</b> Code finds the legend (color bar or class patches) and any lettered markers.</li>
          <li><b>Specification.</b> The model reads the variable, units, region, legend labels and printed statistics.</li>
          <li><b>Measured reading.</b> Code aligns the map to reference boundaries and reads every unit's color against the legend.</li>
          <li><b>Interpretation.</b> The model writes findings from the measured table; each finding cites the rows it rests on.</li>
        </ol>
        <p className="muted small">
          Measured reading works for US maps of <b>counties</b> (any state, or the whole country),{" "}
          <b>census tracts</b> and <b>states</b>, with a color bar (vertical or horizontal) or a classed legend. Other maps are
          interpreted by the model alone and marked <b>Model only</b>.
        </p>
      </div>
    </div>
  );
}

function Reading({ name }) {
  return (
    <div className="reading" aria-live="polite">
      <div className="spinner" />
      <h2>Uploading {name}</h2>
    </div>
  );
}

// ---------------------------------------------------------------- workspace

function StatusChip({ measured }) {
  if (measured === true) return <span className="chip chip-ok">Measured</span>;
  if (measured === false) return <span className="chip chip-warn">Model only</span>;
  return <span className="chip chip-idle">Not measured</span>;
}

function Workspace({ entry, measuring, onMeasure, method, interpretation, onInterpret, thread, onAsk }) {
  // With the framework, interpreting or asking an unmeasured map measures it on the server first
  const measuringFirst =
    entry.measured === null &&
    ((method === "framework" && interpretation?.loading) || thread.some((t) => t.loading && t.method === "framework"));
  return (
    <div className="ws">
      {entry.measured === false && (
        <div className="banner banner-warn">
          <b>Model only.</b> {entry.reason || "This map could not be measured."} Interpretations and answers come
          from the model looking at the image and are not checked against measurements.
        </div>
      )}
      <div className="ws-grid">
        <section className="panel ws-left">
          <MapView entry={entry} />
          <EvidencePanel entry={entry} measuring={measuring || measuringFirst} onMeasure={onMeasure} method={method} />
        </section>
        <section className="ws-right">
          <InterpretationCard state={interpretation} onRun={onInterpret} measured={entry.measured} method={method} />
          <QA thread={thread} onAsk={onAsk} measured={entry.measured} method={method} />
        </section>
      </div>
    </div>
  );
}

// The map, with the reference boundaries as fitted during measuring drawn over it on request
function MapView({ entry }) {
  const [show, setShow] = useState(false);
  const [overlay, setOverlay] = useState(null); // {loading} | {data} | {error}
  const [hover, setHover] = useState(null); // {name, x, y}
  const rows = useMemo(
    () => Object.fromEntries((entry.evidence?.table || []).map((r) => [String(r.name), r])),
    [entry.evidence]
  );
  // a new measurement places the boundaries again, so drop the old outline
  const fitKey = entry.measured ? JSON.stringify(entry.evidence?.diagnostics) : "";
  useEffect(() => {
    setOverlay(null);
    setShow(false);
  }, [fitKey]);

  const toggle = () => {
    const next = !show;
    setShow(next);
    if (next && !overlay?.data) {
      setOverlay({ loading: true });
      api.overlay(entry.id).then((data) => setOverlay({ data })).catch((e) => setOverlay({ error: e.message }));
    }
  };

  const describe = (name) => {
    const r = rows[name];
    if (!r) return name;
    if (r.value !== undefined) return `${name}: ${r.value === null ? "not read" : Number(r.value).toLocaleString()}`;
    if (r.class !== undefined) return `${name}: ${r.class}`;
    if (r.mean_class !== undefined) return `${name}: mean class ${r.mean_class}`;
    return name;
  };

  const data = overlay?.data;
  return (
    <div className="map-block">
      <div className="map-tools">
        <label className="toggle" htmlFor="mi-overlay">
          <input id="mi-overlay" type="checkbox" checked={show} onChange={toggle} disabled={!entry.measured} />
          Show reference boundaries
        </label>
        {!entry.measured && <span className="muted small">Available after measuring</span>}
        {show && overlay?.loading && <span className="muted small">Loading boundaries…</span>}
        {show && overlay?.error && <span className="error-text small">{overlay.error}</span>}
        {show && data && <span className="muted small">{data.units.length.toLocaleString()} outlines · hover for values</span>}
      </div>
      <div className="map-frame">
        <div className="map-stack">
          <img src={entry.image_url} alt={`Uploaded map ${entry.name}`} />
          {show && data && (
            <svg
              className="overlay"
              viewBox={`0 0 ${data.width} ${data.height}`}
              preserveAspectRatio="none"
              onMouseLeave={() => setHover(null)}
            >
              {data.units.map((u) => (
                <path
                  key={u.name}
                  d={u.d}
                  className={hover?.name === u.name ? "on" : ""}
                  onMouseMove={(e) => setHover({ name: u.name, x: e.clientX, y: e.clientY })}
                />
              ))}
            </svg>
          )}
        </div>
      </div>
      {hover && (
        <div className="map-tip" style={{ left: hover.x + 14, top: hover.y + 14 }}>
          {describe(hover.name)}
        </div>
      )}
    </div>
  );
}

function EvidencePanel({ entry, measuring, onMeasure, method }) {
  const [tab, setTab] = useState("spec");
  const ev = entry.evidence;
  return (
    <div className="evidence">
      <div className="panel-head">
        <h2>Measured evidence</h2>
        {entry.measured !== null && !measuring && <button className="link-btn" onClick={onMeasure}>Measure again</button>}
      </div>
      {method === "model_only" && (
        <p className="note-baseline">
          <b>Not used by Model only.</b> The baseline answers from the image alone. Measuring here runs the
          framework's Stages 1–3 for reference or comparison; it does not change Model only answers.
        </p>
      )}
      {measuring ? (
        <div className="pending">
          <div className="spinner spinner-sm" />
          <p className="muted">Finding the legend, aligning reference boundaries and reading every unit. This takes 15–60 seconds.</p>
        </div>
      ) : entry.measured === null ? (
        <div className="blank">
          <p className="muted">
            Not measured yet. Measuring finds the legend, aligns the map to reference boundaries and reads every unit's
            color into a table.
          </p>
          <button className="btn btn-primary" onClick={onMeasure}>Measure map</button>
        </div>
      ) : entry.measured === false ? (
        <>
          <p className="muted">This map could not be measured: {entry.reason}</p>
          {entry.log?.length > 0 && (
            <details open>
              <summary className="small">What ran</summary>
              <LogView lines={entry.log} />
            </details>
          )}
        </>
      ) : (
        <>
          <p className="muted small">Interpretations and answers for this map come only from this table.</p>
          <div className="tabs" role="tablist">
            {[["spec", "Specification"], ["checks", "Measurement checks"], ["table", `Table (${ev.table.length})`], ["log", "Log"]].map(([k, label]) => (
              <button key={k} role="tab" aria-selected={tab === k} className={`tab ${tab === k ? "tab-on" : ""}`} onClick={() => setTab(k)}>
                {label}
              </button>
            ))}
          </div>
          {tab === "spec" && <SpecView spec={ev.spec} />}
          {tab === "checks" && <ChecksView diagnostics={ev.diagnostics} markers={ev.markers} notes={ev.table_notes} />}
          {tab === "table" && <TableView columns={ev.columns} rows={ev.table} />}
          {tab === "log" && <LogView lines={ev.log || []} />}
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- live activity log (bottom-right dock)

const DOCK_STORAGE = "mi_log_open";

function LogDock({ entry, active }) {
  const [open, setOpen] = useState(() => {
    if (window.location.hash === "#log") return true; // a link ending in #log opens the log
    try {
      return localStorage.getItem(DOCK_STORAGE) === "1";
    } catch {
      return false;
    }
  });
  const [lines, setLines] = useState([]);
  const total = useRef(0);
  const body = useRef(null);
  const stick = useRef(true); // follow new lines unless the reader scrolled up
  const id = entry?.id;

  useEffect(() => {
    try {
      localStorage.setItem(DOCK_STORAGE, open ? "1" : "0");
    } catch {
      /* storage blocked */
    }
  }, [open]);

  // a different map has its own log
  useEffect(() => {
    setLines([]);
    total.current = 0;
  }, [id]);

  // poll quickly while something runs, slowly while the panel is open, and once more when a run ends
  useEffect(() => {
    if (!id) return undefined;
    let stopped = false;
    let busy = false; // one request at a time, so the same lines are never added twice
    const pull = () => {
      if (busy) return;
      busy = true;
      api
        .activity(id, total.current)
        .then((r) => {
          if (stopped || !r.lines.length) return;
          total.current = r.total;
          setLines((old) => [...old, ...r.lines]);
        })
        .catch(() => {})
        .finally(() => {
          busy = false;
        });
    };
    pull();
    const timer = active || open ? setInterval(pull, active ? 700 : 4000) : null;
    return () => {
      stopped = true;
      if (timer) clearInterval(timer);
    };
  }, [id, active, open]);

  useEffect(() => {
    if (open && stick.current && body.current) body.current.scrollTop = body.current.scrollHeight;
  }, [lines, open]);

  if (!entry) return null;
  const last = lines[lines.length - 1]?.replace(/^\[[^\]]*\]\s*/, "");
  return (
    <div className={`dock ${open ? "dock-open" : ""}`} aria-live="polite">
      {open ? (
        <section className="dock-panel" aria-label="Activity log">
          <header className="dock-head">
            <span className={`dock-dot ${active ? "live" : ""}`} aria-hidden="true" />
            <span className="dock-title">Activity log</span>
            <span className="dock-sub">{entry.name}</span>
            <button className="dock-btn" onClick={() => setOpen(false)} aria-label="Collapse the activity log">▾</button>
          </header>
          <div
            className="dock-body"
            ref={body}
            onScroll={(e) => {
              const el = e.currentTarget;
              stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
            }}
          >
            <LogView lines={lines} />
            {active && <p className="dock-running">Running…</p>}
          </div>
        </section>
      ) : (
        <button className="dock-tab" onClick={() => setOpen(true)} aria-label="Open the activity log">
          <span className={`dock-dot ${active ? "live" : ""}`} aria-hidden="true" />
          <span className="dock-title">Activity log</span>
          {lines.length > 0 && <span className="dock-count">{lines.length}</span>}
          {active && last && <span className="dock-last">{last}</span>}
          <span aria-hidden="true">▴</span>
        </button>
      )}
    </div>
  );
}

// Steps the framework recorded, each stamped "[ 2.3s] ..." (elapsed) or "[22:47:20] ..." (clock time)
function LogView({ lines }) {
  if (!lines.length) return <p className="muted small">No steps recorded.</p>;
  return (
    <ol className="log">
      {lines.map((line, i) => {
        const [, time, text] = line.match(/^\[\s*([^\]]+)\]\s*(.*)$/) || [null, "", line];
        return (
          <li key={i} className={/^(Stopped|Failed)/.test(text) ? "log-stop" : ""}>
            <span className="log-time">{time}</span>
            <span className="log-text">{text}</span>
          </li>
        );
      })}
    </ol>
  );
}

const SPEC_FIELDS = [
  ["title", "Title"], ["map_type", "Map type"], ["variable", "Variable"], ["units", "Units"],
  ["region", "Region"], ["unit", "Unit"], ["legend_type", "Legend"], ["legend_ticks", "Color bar ticks"],
  ["legend_classes", "Classes"], ["annotations", "Printed statistics"], ["value_is_count", "Raw count"],
];

const show = (v) => (Array.isArray(v) ? (v.length ? v.join(" · ") : "—") : v === true ? "Yes" : v === false ? "No" : v ?? "—");

function SpecView({ spec }) {
  return (
    <dl className="kv">
      {SPEC_FIELDS.filter(([k]) => spec[k] !== undefined && !(Array.isArray(spec[k]) && !spec[k].length)).map(([k, label]) => (
        <div key={k}>
          <dt>{label}</dt>
          <dd>{show(spec[k])}</dd>
        </div>
      ))}
    </dl>
  );
}

function fitChip(iou) {
  if (iou >= 0.9) return <span className="chip chip-ok">good</span>;
  if (iou >= 0.8) return <span className="chip chip-warn">check</span>;
  return <span className="chip chip-bad">poor</span>;
}

function ChecksView({ diagnostics: d, markers, notes }) {
  const rows = [];
  if (d.georef_iou !== undefined) rows.push(["Fit to reference boundaries (IoU)", <>{d.georef_iou} {fitChip(d.georef_iou)}</>]);
  if (d.georef_crs) rows.push(["Projection that fit best", d.georef_crs]);
  if (d.calibration) rows.push(["Color bar calibrated from", d.calibration]);
  if (d.low_confidence_rows !== undefined) rows.push(["Low-confidence units", d.low_confidence_rows]);
  if (d.units_read !== undefined) {
    const total = d.units_read + d.units_too_small + d.units_no_color;
    const pct = (n) => `${n.toLocaleString()} (${Math.round((100 * n) / total)}%)`;
    rows.push(["Units read", pct(d.units_read)]);
    rows.push(["Too small to read (under 1 pixel)", pct(d.units_too_small)]);
    rows.push(["No legend color", pct(d.units_no_color)]);
  }
  const markerList = Object.entries(markers || {});
  if (markerList.length) rows.push(["Markers", markerList.map(([k, v]) => `${k} → ${v ?? "?"}`).join(", ")]);
  return (
    <div className="checks">
      <dl className="kv">
        {rows.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      <details>
        <summary>How the table was measured</summary>
        <p className="muted small pre">{notes}</p>
      </details>
    </div>
  );
}

function TableView({ columns, rows }) {
  const [sort, setSort] = useState(null);
  const [filter, setFilter] = useState("");
  const shown = useMemo(() => {
    let out = rows.filter((r) => String(r.name).toLowerCase().includes(filter.toLowerCase()));
    if (sort) {
      out = [...out].sort((a, b) => {
        const x = a[sort.col], y = b[sort.col];
        const cmp = typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y));
        return sort.dir * cmp;
      });
    }
    return out;
  }, [rows, sort, filter]);
  const fmt = (v) => (v === null || v === undefined ? "—" : typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2)) : v);
  return (
    <div className="table-view">
      <input
        id="mi-table-filter"
        className="input"
        placeholder="Filter by name"
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
      />
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              {columns.map((c) => (
                <th key={c}>
                  <button
                    className="th-btn"
                    onClick={() => setSort((s) => ({ col: c, dir: s?.col === c ? -s.dir : typeof rows[0]?.[c] === "number" ? -1 : 1 }))}
                  >
                    {c}
                    {sort?.col === c ? (sort.dir > 0 ? " ▲" : " ▼") : ""}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <tr key={r.name} className={r.name === "ALL" ? "row-all" : ""}>
                {columns.map((c) => (
                  <td key={c} className={typeof r[c] === "number" ? "num" : ""}>{fmt(r[c])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- interpretation and Q&A

function EvidenceChips({ evidence, check }) {
  if (!evidence?.length) return null;
  const missing = new Set(check?.missing || []);
  return (
    <span className="ev">
      {evidence.map((name) => (
        <span key={name} className={`ev-chip ${missing.has(name) ? "ev-missing" : ""}`} title={missing.has(name) ? "Not a row of the measured table" : "Row of the measured table"}>
          {missing.has(name) ? "⚠ " : ""}
          {name}
        </span>
      ))}
    </span>
  );
}

// Which method produced a result; "fell back" when the framework was asked for but the map could not be measured
function MethodChip({ data, requested }) {
  const used = data?.method || requested;
  const fellBack = data && data.requested_method === "framework" && data.method === "model_only";
  return (
    <span className={`method-chip m-${used}`} title={METHODS[used]?.desc}>
      {METHODS[used]?.label || used}
      {fellBack && " · map not measurable"}
    </span>
  );
}

function InterpretationCard({ state, onRun, measured, method }) {
  const measuresFirst = method === "framework" && measured === null;
  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Interpretation</h2>
        <span className="panel-tools">
          <MethodChip data={state?.data} requested={method} />
          {state?.data && <button className="link-btn" onClick={onRun}>Regenerate</button>}
        </span>
      </div>
      {!state && (
        <div className="blank">
          <p className="muted">
            Not generated yet for {METHODS[method].label}.
            {measuresFirst && " Generating measures the map first."}
            {method === "model_only" && " The model will interpret the image directly; nothing is measured or checked."}
          </p>
          <button className="btn btn-primary" onClick={onRun}>Generate interpretation</button>
        </div>
      )}
      {state?.loading && (
        <div className="pending">
          <div className="spinner spinner-sm" />
          <p className="muted">{measuresFirst ? "Measuring the map, then writing the interpretation…" : "Writing the interpretation…"}</p>
        </div>
      )}
      {state?.error && <p className="error-text">{state.error} <button className="link-btn" onClick={onRun}>Try again</button></p>}
      {state?.data && (
        <div className="interp">
          <p className="summary">{state.data.summary}</p>
          {state.data.findings?.length > 0 && (
            <ul className="findings">
              {state.data.findings.map((f, i) => (
                <li key={i}>
                  <span>{f.claim}</span>
                  {state.data.grounded && (
                    <span className="finding-ev">
                      {f.evidence_check && !f.evidence_check.verified && <span className="chip chip-warn">unverified</span>}
                      <EvidenceChips evidence={f.evidence} check={f.evidence_check} />
                    </span>
                  )}
                </li>
              ))}
            </ul>
          )}
          {state.data.caveats?.length > 0 && (
            <div className="caveats">
              <h3>Caveats</h3>
              <ul>
                {state.data.caveats.map((c, i) => <li key={i}>{c}</li>)}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function QA({ thread, onAsk, measured, method }) {
  const [text, setText] = useState("");
  const submit = (e) => {
    e.preventDefault();
    const q = text.trim();
    if (!q) return;
    onAsk(q);
    setText("");
  };
  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Ask about this map</h2>
        <span className="muted small">Answers with: <MethodChip requested={method} /></span>
      </div>
      {thread.length === 0 && (
        <p className="muted small">
          {measured === false || method === "model_only"
            ? "Ask anything you can see on the map."
            : "For example: Which unit has the highest value? How many units are above 500?"}
          {method === "framework" && measured === null && " The first question measures the map first, which takes 15–60 seconds."}
          {" "}Switch the method at the top to ask the same question both ways and compare.
        </p>
      )}
      <ol className="thread">
        {thread.map((item, i) => (
          <li key={i}>
            <p className="q">
              {item.question} <MethodChip data={item.data} requested={item.method} />
            </p>
            {item.loading && <p className="muted">Answering…</p>}
            {item.error && <p className="error-text">{item.error}</p>}
            {item.data && (
              <div className="a">
                <p className="a-text">{String(item.data.answer ?? "—")}</p>
                {item.data.explanation && <p className="muted small">{item.data.explanation}</p>}
                {item.data.grounded && <EvidenceChips evidence={item.data.evidence} check={item.data.evidence_check} />}
              </div>
            )}
          </li>
        ))}
      </ol>
      <form className="ask" onSubmit={submit}>
        <input id="mi-question" className="input" value={text} onChange={(e) => setText(e.target.value)} placeholder="Type a question" />
        <button className="btn btn-primary" type="submit" disabled={!text.trim()}>Ask</button>
      </form>
    </div>
  );
}

// ---------------------------------------------------------------- API key

function KeyStatus({ forceOpen, onSaved }) {
  const [open, setOpen] = useState(false);
  const [value, setValue] = useState("");
  const hasKey = !!getApiKey();
  const shown = open || forceOpen;
  return (
    <div className="key">
      <button className="link-btn" onClick={() => setOpen((o) => !o)}>
        {hasKey ? "API key set" : "Using server key"}
      </button>
      {shown && (
        <form
          className="key-pop"
          onSubmit={(e) => {
            e.preventDefault();
            if (!value.trim()) return;
            saveApiKey(value);
            setValue("");
            setOpen(false);
            onSaved();
          }}
        >
          <label htmlFor="mi-key">OpenAI API key (shared with the GIS Co-Scientist)</label>
          <input id="mi-key" className="input" type="password" value={value} onChange={(e) => setValue(e.target.value)} placeholder="sk-..." />
          <button className="btn btn-primary" type="submit">Save key</button>
        </form>
      )}
    </div>
  );
}
