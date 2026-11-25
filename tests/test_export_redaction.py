"""API keys must not leave the computer inside an exported conversation."""
import json
import os
import sys
import unittest
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests import test_conversation_bundle as base  # noqa: E402
from WebUI import conversation_bundle as cb, conversation_db as cdb  # noqa: E402
from WebUI.export_redaction import PLACEHOLDER, Redactor  # noqa: E402

SAVED = "fakeSavedKey0123456789abcdef"
IN_STATE = "fakeStateKey9876543210fedcba"
IN_FILE = "fakeFileKey1122334455"
FORGOTTEN = "fakeforgotten00112233445566778899"
PASSWORD = "p&ss w/rd=1"


class ExportRedactionTest(base.ConversationBundleTest):
    def test_no_key_in_bundle_and_original_untouched(self):
        out = os.path.join(self.sender.root, "Outputs", self.cid)
        state_path = os.path.join(out, "workflow_state.json")
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        state["ctx"] = {"name": "São Paulo", "_data_source_keys": {"A": IN_STATE, "B": "abc"}}
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
        code = f'api_key = "{IN_STATE}"\nurl = "https://x.org/area/csv/{SAVED}/VIIRS?key={FORGOTTEN}"\n'
        code_path = os.path.join(out, "step.py")
        with open(code_path, "w", encoding="utf-8", newline="") as f:
            f.write(code)
        with open(os.path.join(os.path.dirname(self.upload_path), "notes.csv"), "w") as f:
            f.write(f"a,{SAVED}\n")
        keys = os.path.join(self.sender.root, "agents", "data_agent", "DataRetriever_Handbooks", "Keys")
        os.makedirs(keys)
        with open(os.path.join(keys, "X.keys"), "w") as f:
            f.write(f"[API_Key]\nX_key = {IN_FILE}\n[Links]\nwebsite = https://maps.example.org\n")
        self.sender.use()
        cdb.add_message(self.cid, "assistant", f"used {IN_FILE}", message_type="workflow",
                        metadata={"code": code, "pw": "u=p%26ss%20w%2Frd%3D1",
                                  "site": "https://maps.example.org"})
        cdb.save_file_blob(self.cid, "extra.txt", f"k={SAVED}".encode())
        cdb.save_artifact(self.cid, "page", f"<b>{SAVED}</b>")

        cb.export_conversation(self.cid, self.bundle, app_root=self.sender.root,
                               saved_keys={"S": SAVED, "nested": {"P": PASSWORD}, "none": None})
        with zipfile.ZipFile(self.bundle) as zf:
            everything = b"\n".join(zf.read(n) for n in zf.namelist()).decode("latin-1")
            exported = json.loads(zf.read(f"outputs/{self.cid}/workflow_state.json"))
        for value in (SAVED, IN_STATE, IN_FILE, FORGOTTEN, PASSWORD, "p%26ss%20w%2Frd%3D1",
                      "_data_source_keys"):
            self.assertNotIn(value, everything)
        self.assertIn(PLACEHOLDER, everything)
        self.assertIn("https://maps.example.org", everything)
        self.assertEqual(exported["ctx"], {"name": "São Paulo"})
        with open(code_path, encoding="utf-8", newline="") as f:
            self.assertEqual(f.read(), code)
        self.recipient.use()
        self.assertTrue(cb.import_conversation(self.bundle, "r", app_root=self.recipient.root))

    def test_forgotten_keys_by_the_way_they_are_written(self):
        for text in (f'api_key = "{FORGOTTEN}"', f"MAP_KEY = '{FORGOTTEN}'", f'key = "{FORGOTTEN}"',
                     f'{{"X-Api-Key": "{FORGOTTEN}"}}', f'get("FIRMS_MAP_KEY", "{FORGOTTEN}")',
                     f"https://a.gov/x?in=state:45&key={FORGOTTEN}", f"u&api_key={FORGOTTEN}",
                     f"DEFAULT_MAP_KEY = '{FORGOTTEN}'", f"'api_key': '{FORGOTTEN}'",
                     f"&appid={FORGOTTEN}"):
            for shown in (text, json.dumps("x\n    " + text), json.dumps(json.dumps("y\n" + text))):
                self.assertNotIn(FORGOTTEN, Redactor().text(shown), shown)

    def test_short_and_path_keys(self):
        for text, key in (('email = "a@b.org"\n    key = "tealfox12ab34"', "tealfox12ab34"),
                          ("https://aqs.epa.gov/x?email=a@b.org&key=amberfox71", "amberfox71"),
                          ("https://firms.modaps.eosdis.nasa.gov/api/area/csv/0a1b2c3d4e5f60718293a4b5c6d7e8f9/VIIRS",
                           "0a1b2c3d4e5f60718293a4b5c6d7e8f9")):
            for shown in (text, json.dumps(text)):
                self.assertNotIn(key, Redactor().text(shown))

    def test_ordinary_text_is_kept(self):
        for text in ('join_key = "GEOID20_tract_2020_id"', 'primary_key: "abc123abc123abc123"',
                     "api_key = os.environ['CENSUS_API_KEY']", 'api_key = "YOUR_API_KEY"',
                     "task-a4c5a1b9-449e-440c-a2cf-003513b58787", "risk-2020-estimates-by-county-level",
                     "key=GEOID", 'key = "tract_2020_id"', "https://www.arcgis.com/home/item.html?id=0a1b2c3d4e5f60718293a4b5c6d7e8f9", '{"key": "a4"}'):
            self.assertEqual(Redactor(["DEMO_KEY", "abc", ""]).text(text), text)

    def test_binary_files_unchanged(self):
        data = b"\x89PNG" + SAVED.encode()
        self.assertEqual(Redactor([SAVED]).file("a.png", data), data)


if __name__ == "__main__":
    unittest.main()
