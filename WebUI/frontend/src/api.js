// Same origin as the GIS Co-Scientist, so the API key it saved in Settings is reused here.
const KEY_STORAGE = "agm_openai_api_key";

export function getApiKey() {
  try {
    return (localStorage.getItem(KEY_STORAGE) || "").trim();
  } catch {
    return "";
  }
}

export function saveApiKey(key) {
  try {
    localStorage.setItem(KEY_STORAGE, key.trim());
  } catch {
    /* storage blocked: the server's own key is used instead */
  }
}

async function request(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  const key = getApiKey();
  if (key) headers["X-API-Key"] = key;
  const response = await fetch(path, { ...options, headers });
  let body = null;
  try {
    body = await response.json();
  } catch {
    /* non-JSON error page */
  }
  if (!response.ok) {
    const error = new Error(body?.error || `Request failed (${response.status})`);
    error.status = response.status;
    error.log = body?.log || []; // steps that ran before a measurement failed
    throw error;
  }
  return body;
}

export const api = {
  models: () => request("/api/map-interpreter/models"),
  listMaps: () => request("/api/map-interpreter/maps"),
  getMap: (id) => request(`/api/map-interpreter/maps/${id}`),
  deleteMap: (id) => request(`/api/map-interpreter/maps/${id}`, { method: "DELETE" }),
  renameMap: (id, title) =>
    request(`/api/map-interpreter/maps/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    }),
  addMap: (file) => {
    const form = new FormData();
    form.append("file", file);
    return request("/api/map-interpreter/maps", { method: "POST", body: form });
  },
  overlay: (id) => request(`/api/map-interpreter/maps/${id}/overlay`),
  activity: (id, since) => request(`/api/map-interpreter/maps/${id}/activity?since=${since}`),
  measure: (id, model) => post(`/api/map-interpreter/maps/${id}/measure`, { model }),
  // method: "framework" (answer from the measured table) or "model_only" (the baseline)
  interpret: (id, model, method) => post(`/api/map-interpreter/maps/${id}/interpret`, { model, method }),
  ask: (id, question, model, method) => post(`/api/map-interpreter/maps/${id}/ask`, { question, model, method }),
};

function post(path, body) {
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
