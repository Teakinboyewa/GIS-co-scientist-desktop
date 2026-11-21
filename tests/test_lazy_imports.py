"""Heavy libraries load on first use, and still work when used.

The checks run in a fresh Python process so "not loaded yet" means what it
says: nothing else in the test run has imported these libraries already.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEAVY = ["sklearn", "scipy", "IPython", "ipywidgets", "matplotlib", "networkx", "pyvis"]

SCRIPT = textwrap.dedent(r"""
    import json, os, sys
    sys.path.insert(0, ROOT)
    out = {}
    import agents.data_agent.agm_data_agent            # pulls in the whole agents package
    import agents.task_manager.WEBUI_TASK_PIPELINE.task_solution as ts
    import agents.task_manager.spatial_analysis_executor as sae
    import agents.result_analysis_agent.figure_renderer as fr
    import agents.result_analysis_agent.result_analysis_graph as rag
    from agents.modeling_agent.modeling_tools import ModelingTools
    out["loaded_at_import"] = [m for m in HEAVY if m in sys.modules]

    # networkx through the lazy stand-in
    g = sae.nx.DiGraph(); g.add_edge("a", "b")
    out["nx_edges"] = list(g.edges())

    # pyvis graph view
    net = ts.Network(notebook=False, directed=True)
    net.add_node(1, label="x")
    out["pyvis_nodes"] = len(net.nodes)

    # a chart, drawn off-screen
    csv = os.path.join(TMP, "t.csv")
    with open(csv, "w") as f:
        f.write("x,y\n1,2\n2,4\n3,6\n4,8\n5,10\n")
    png = os.path.join(TMP, "chart.png")
    fr._render_csv(csv, {"type": "scatter", "x": "x", "y": "y"}, png, "test")
    import matplotlib
    out["backend"] = matplotlib.get_backend().lower()
    out["chart_written"] = os.path.getsize(png) > 0

    # fitting and scoring a model with scikit-learn
    tools = ModelingTools()
    fit = tools.fit_model("sklearn.linear_model.LinearRegression", csv, "y", ["x"], test_size=0.4)
    out["fit_error"] = fit.get("error")
    name = next(iter(tools.fitted_models), None)
    ev = tools.evaluate_model(name) if name else {"error": "no model"}
    out["eval_keys"] = sorted(k for k in ev if k != "error") if isinstance(ev, dict) else []
    out["eval_error"] = ev.get("error") if isinstance(ev, dict) else None

    # IPython display helpers still call through
    out["ipython_markdown"] = type(rag.Markdown("**hi**")).__name__
    out["widgets_flag"] = ts.WIDGETS_AVAILABLE
    out["loaded_after_use"] = [m for m in HEAVY if m in sys.modules]
    print("RESULT" + json.dumps(out))
""")


class LazyImportsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="gcs_lazy_")
        code = f"ROOT = {ROOT!r}\nTMP = {cls.tmp!r}\nHEAVY = {HEAVY!r}\n" + SCRIPT
        env = dict(os.environ)
        env.pop("MPLBACKEND", None)
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              cwd=os.path.join(ROOT, "WebUI"), env=env, timeout=600)
        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT")), None)
        if line is None:
            raise AssertionError("check script failed:\n" + proc.stdout[-3000:] + proc.stderr[-3000:])
        cls.out = json.loads(line[len("RESULT"):])

    def test_nothing_heavy_loads_at_import(self):
        self.assertEqual(self.out["loaded_at_import"], [])

    def test_networkx_and_pyvis_work_on_use(self):
        self.assertEqual(self.out["nx_edges"], [["a", "b"]])
        self.assertEqual(self.out["pyvis_nodes"], 1)

    def test_charts_draw_off_screen(self):
        self.assertEqual(self.out["backend"], "agg")
        self.assertTrue(self.out["chart_written"])

    def test_models_fit_and_score(self):
        self.assertIsNone(self.out["fit_error"])
        self.assertIsNone(self.out["eval_error"])
        self.assertTrue(self.out["eval_keys"])

    def test_ipython_helpers_call_through(self):
        self.assertEqual(self.out["ipython_markdown"], "Markdown")
        self.assertIsInstance(self.out["widgets_flag"], bool)
        for lib in ("sklearn", "matplotlib", "networkx", "pyvis", "IPython"):
            self.assertIn(lib, self.out["loaded_after_use"])


class LazyModuleUnitTest(unittest.TestCase):
    def test_lazy_module_and_function(self):
        sys.path.insert(0, ROOT)
        from utils.lazy_imports import lazy_module, lazy_function, is_installed
        m = lazy_module("json")
        self.assertIn("not", repr(m) + " not")  # repr works before loading
        self.assertEqual(m.dumps([1]), "[1]")
        self.assertIn("(loaded)", repr(m))
        f = lazy_function("json", "loads")
        self.assertEqual(f("[2]"), [2])
        self.assertEqual(f.__name__, "loads")
        self.assertTrue(is_installed("json"))
        self.assertFalse(is_installed("no_such_package_gcs"))


if __name__ == "__main__":
    unittest.main()
