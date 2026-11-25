"""Saved workflow state must never hold data-source API keys.

Loads the real save functions without importing the app's GIS/LLM dependencies.
"""
import ast
from datetime import date, datetime
import json
import os
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'_serialize_state', 'save_workflow_state', 'get_workflow_state', '_UNSAVED_STATE_KEYS'}
KEY = 'not-a-real-key-1234567890'


def load(folder):
    tree = ast.parse((ROOT / 'WebUI/agm_workflow.py').read_text(encoding='utf-8'))
    body = [n for n in tree.body
            if getattr(n, 'name', None) in NAMES
            or (isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', None) in NAMES)]
    scope = {
        'json': json, 'os': os, 'np': np, 'date': date, 'datetime': datetime,
        'pd': types.SimpleNamespace(Timestamp=datetime, Timedelta=datetime),
        'workflow_states': {}, 'print': lambda *a: None,
        '_get_state_dir': lambda task_id: folder,
        '_get_state_file': lambda task_id: os.path.join(folder, 'workflow_state.json'),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), 'agm_workflow.py', 'exec'), scope)
    return scope


class SavedStateHasNoKeys(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = tmp.name
        self.m = load(tmp.name)
        self.ctx = {
            'task_id': 't', '_data_source_keys': {'FIRMS_MAP_KEY': KEY},
            'spatial_condition': 'P', 'join_key': 'GEOID',
            'artifacts': {'steps': [{'n': np.int64(3), '_data_source_keys': {'k': KEY}}]},
        }

    def text(self):
        return (Path(self.folder) / 'workflow_state.json').read_text(encoding='utf-8')

    def test_serialized_context_has_no_keys(self):
        self.assertNotIn(KEY, json.dumps(self.m['_serialize_state'](self.ctx)))

    def test_file_has_no_keys_when_context_is_passed_raw(self):
        self.m['save_workflow_state']('t', {'ctx': self.ctx})
        self.assertNotIn(KEY, self.text())
        self.assertNotIn('_data_source_keys', self.text())

    def test_running_workflow_keeps_its_keys(self):
        self.m['save_workflow_state']('t', {'ctx': self.m['_serialize_state'](self.ctx)})
        self.assertEqual(self.ctx['_data_source_keys'], {'FIRMS_MAP_KEY': KEY})

    def test_everything_else_survives_a_reload(self):
        self.m['save_workflow_state']('t', {'ctx': self.ctx, 'structured_plan': {'objectives': [1]}})
        state = self.m['get_workflow_state']('t')
        self.assertEqual(state['structured_plan'], {'objectives': [1]})
        self.assertEqual(state['ctx'], {'task_id': 't', 'spatial_condition': 'P', 'join_key': 'GEOID',
                                        'artifacts': {'steps': [{'n': 3}]}})

    def test_resaving_an_old_file_removes_its_keys(self):
        (Path(self.folder) / 'workflow_state.json').write_text(json.dumps({'ctx': self.ctx}, default=int),
                                                               encoding='utf-8')
        self.m['save_workflow_state']('t', self.m['get_workflow_state']('t'))
        self.assertNotIn(KEY, self.text())


if __name__ == '__main__':
    unittest.main()
