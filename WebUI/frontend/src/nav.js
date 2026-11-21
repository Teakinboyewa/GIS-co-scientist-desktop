// Client-side page switch between /home and /map-interpreter (both served by Flask as this app).
export function navigate(path) {
  window.history.pushState({}, "", path);
  window.dispatchEvent(new PopStateEvent("popstate"));
}
