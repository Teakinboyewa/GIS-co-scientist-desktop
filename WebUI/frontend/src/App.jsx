import { useEffect, useState } from "react";
import Home from "./Home.jsx";
import MapInterpreter from "./MapInterpreter.jsx";

// Flask serves this app at /home and /map-interpreter; switching between them stays client-side (nav.js).
export default function App() {
  const [path, setPath] = useState(window.location.pathname);
  useEffect(() => {
    const onPop = () => setPath(window.location.pathname);
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  if (path.startsWith("/map-interpreter")) return <MapInterpreter />;
  return <Home />;
}
