import { useMemo } from "react";
import { navigate } from "./nav.js";

// Spatial Analysis and Data Retrieval open the existing GIS Co-Scientist in that mode: the app restores
// its mode from localStorage on load ('agm_intent_mode'), so setting it here needs no change to the app.
function openCoScientist(mode) {
  if (mode) {
    try {
      localStorage.setItem("agm_intent_mode", mode);
    } catch {
      /* storage blocked: the app opens in its last mode */
    }
  }
  window.location.href = "/";
}

const Icon = {
  coscientist: (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="9" />
      <path d="M3 12h18M12 3c2.6 2.6 3.9 5.6 3.9 9s-1.3 6.4-3.9 9c-2.6-2.6-3.9-5.6-3.9-9s1.3-6.4 3.9-9z" />
    </svg>
  ),
  analysis: (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 20V10M10 20V4M16 20v-7M22 20H2" />
    </svg>
  ),
  retrieval: (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <ellipse cx="12" cy="5.5" rx="7" ry="2.5" />
      <path d="M5 5.5v6c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5v-6M12 14v7M8.5 17.5L12 21l3.5-3.5" />
    </svg>
  ),
  handbook: (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M5 4h11a3 3 0 013 3v13H8a3 3 0 01-3-3V4z" />
      <path d="M5 17a3 3 0 013-3h11M9 8h6" />
    </svg>
  ),
  map: (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M3 6l6-2.5 6 2.5 6-2.5v14.5l-6 2.5-6-2.5-6 2.5z" />
      <path d="M9 3.5v14.5M15 6v14.5" />
    </svg>
  ),
};

const AGENTS = [
  {
    name: "GIS Co-Scientist",
    kind: "Workspace",
    desc: "The full research workspace: plan a study, retrieve data, run spatial analyses and write up results, in conversation.",
    icon: Icon.coscientist,
    tone: "blue",
    featured: true,
    open: () => openCoScientist(null),
  },
  {
    name: "Map Interpreter",
    kind: "New",
    desc: "Add a thematic map. It measures the legend and every unit, then interprets the map with cited evidence.",
    icon: Icon.map,
    tone: "green",
    open: () => navigate("/map-interpreter"),
  },
  {
    name: "Spatial Analysis Agent",
    kind: "Mode",
    desc: "Describe an analysis, upload data or let it fetch data, and run it directly.",
    icon: Icon.analysis,
    tone: "navy",
    open: () => openCoScientist("task"),
  },
  {
    name: "Data Retrieval Agent",
    kind: "Mode",
    desc: "Find and download datasets for a place and topic, without running an analysis.",
    icon: Icon.retrieval,
    tone: "teal",
    open: () => openCoScientist("data_retriever"),
  },
  {
    name: "Handbook Studio",
    kind: "Studio",
    desc: "Generate, test and refine knowledge handbooks that teach agents a data source.",
    icon: Icon.handbook,
    tone: "cyan",
    open: () => (window.location.href = "/handbook-studio"),
  },
];

// ---------------------------------------------------------------- hero backdrop: contour lines and a graticule

function smoothClosedPath(points) {
  // Catmull-Rom through the points, as cubic Beziers, closed
  const n = points.length;
  const p = (i) => points[(i + n) % n];
  let d = `M${p(0)[0].toFixed(1)} ${p(0)[1].toFixed(1)}`;
  for (let i = 0; i < n; i++) {
    const [p0, p1, p2, p3] = [p(i - 1), p(i), p(i + 1), p(i + 2)];
    const c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6];
    const c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
    d += `C${c1[0].toFixed(1)} ${c1[1].toFixed(1)} ${c2[0].toFixed(1)} ${c2[1].toFixed(1)} ${p2[0].toFixed(1)} ${p2[1].toFixed(1)}`;
  }
  return d + "Z";
}

function contours(cx, cy, rings, base, step, seed, stretch = 1.5) {
  return Array.from({ length: rings }, (_, i) => {
    const r = base + i * step;
    const pts = Array.from({ length: 48 }, (_, k) => {
      const a = (k / 48) * Math.PI * 2;
      const wobble = 1 + 0.13 * Math.sin(3 * a + seed + i * 0.35) + 0.06 * Math.sin(5 * a + seed * 1.7 - i * 0.2);
      return [cx + Math.cos(a) * r * wobble * stretch, cy + Math.sin(a) * r * wobble];
    });
    return smoothClosedPath(pts);
  });
}

function HeroBackdrop() {
  const lines = useMemo(
    () => [...contours(230, 330, 9, 26, 34, 0.6), ...contours(1010, 40, 8, 30, 38, 2.4, 1.3), ...contours(640, 470, 5, 40, 36, 4.1, 1.8)],
    []
  );
  // a graticule globe at the right, echoing the logo
  const gx = 1180, gy = 210, gr = 230;
  const meridians = [0.2, 0.45, 0.7, 0.92].map((f) => gr * f);
  const parallels = [-0.66, -0.33, 0, 0.33, 0.66].map((f) => gy + gr * f);
  return (
    <svg className="hero-art" viewBox="0 0 1400 460" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
      <defs>
        <clipPath id="globe">
          <circle cx={gx} cy={gy} r={gr} />
        </clipPath>
      </defs>
      <g className="contour">
        {lines.map((d, i) => <path key={i} d={d} />)}
      </g>
      <g className="graticule">
        <circle cx={gx} cy={gy} r={gr} />
        <g clipPath="url(#globe)">
          <line x1={gx} y1={gy - gr} x2={gx} y2={gy + gr} />
          {meridians.map((rx) => <ellipse key={rx} cx={gx} cy={gy} rx={rx} ry={gr} />)}
          {parallels.map((y) => <line key={y} x1={gx - gr} y1={y} x2={gx + gr} y2={y} />)}
        </g>
      </g>
    </svg>
  );
}

// ---------------------------------------------------------------- page

export default function Home() {
  return (
    <div className="home-page">
      <header className="hero">
        <HeroBackdrop />
        <div className="hero-inner">
          <div className="hero-mark">
            <img src="/icon/GIS-Coscientist_logo.png" alt="GIS Co-Scientist logo" />
          </div>
          <div className="hero-text">
            <p className="hero-eyebrow">GIS Co-Scientist</p>
            <h1>Agents for geospatial operations</h1>
            <p className="hero-lede">
              Plan and run spatial analyses, retrieve data, build data-source handbooks and interpret thematic maps.
              Choose where to start.
            </p>
          </div>
        </div>
      </header>

      <main className="agents" aria-label="Agents">
        {AGENTS.map((agent) => (
          <button key={agent.name} className={`agent-card tone-${agent.tone} ${agent.featured ? "featured" : ""}`} onClick={agent.open}>
            <span className="agent-top">
              <span className="agent-icon">{agent.icon}</span>
              <span className={`agent-kind ${agent.kind === "New" ? "is-new" : ""}`}>{agent.kind}</span>
            </span>
            <span className="agent-name">{agent.name}</span>
            <span className="agent-desc">{agent.desc}</span>
            <span className="agent-open">
              Open <span aria-hidden="true">→</span>
            </span>
          </button>
        ))}
      </main>

      <footer className="home-foot">
        <a href="https://giscience.psu.edu" target="_blank" rel="noopener noreferrer">
          <img src="/icon/gibd_logo.png" alt="GIBD" />
          <span>giscience.psu.edu</span>
        </a>
      </footer>
    </div>
  );
}
