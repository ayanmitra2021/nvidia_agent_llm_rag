"""The authored language/mode matrix and saved work are release contracts."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2] / "notebooks"
GRAPH = json.loads((ROOT / "composer/translation_web.json").read_text())
SPEC = importlib.util.spec_from_file_location("course_modes", ROOT / "composer/switch_language.py")
PROJECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROJECTOR)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class CourseModes(unittest.TestCase):
    def test_checked_in_projection_matches_source(self):
        for name, entry in GRAPH["notebooks"].items():
            notebook = json.loads((ROOT / name).read_text())
            self.assertEqual(PROJECTOR._project(entry, "en", notebook, "exercises"), notebook, name)

    def workspace(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        for name in GRAPH["notebooks"]:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes((ROOT / name).read_bytes())
        path = root / "graph.json"
        path.write_text(json.dumps(GRAPH))
        return root, path

    def test_every_discovered_language_and_mode(self):
        code_by_mode = {}
        for language in GRAPH["languages"]:
            for mode in ("exercises", "solutions"):
                root, path = self.workspace()
                self.assertTrue(PROJECTOR.switch_language(language, root, path, mode))
                for name, entry in GRAPH["notebooks"].items():
                    with self.subTest(language=language, mode=mode, notebook=name):
                        notebook = json.loads((root / name).read_text())
                        self.assertEqual(len({c["id"] for c in notebook["cells"]}), len(notebook["cells"]))
                        code = {c["id"]: "".join(c["source"]) for c in notebook["cells"]
                                if c["cell_type"] == "code" and c["id"] != "course-language-switch"}
                        key = name, mode
                        self.assertEqual(code, code_by_mode.setdefault(key, code))
                        for definition in entry["cells"]:
                            if definition["cell_type"] == "code" and definition["id"] != "course-language-switch":
                                expected = definition.get("solution", definition["content"]) if mode == "solutions" else definition["content"]
                                self.assertEqual(code[definition["id"]], expected)
                        if name == GRAPH["entrypoint"]:
                            selector = next(c for c in notebook["cells"] if c["id"] == "course-language-switch")
                            self.assertIn('show_language_selector()', "".join(selector["source"]))
                            self.assertEqual(notebook["metadata"]["course_mode"], mode)
                            self.assertEqual(notebook["metadata"]["course_language"], language)
                        self.assertEqual(PROJECTOR._project(entry, language, notebook, mode), notebook)

    def test_code_edits_and_inserted_cells_survive_round_trip(self):
        for name, entry in GRAPH["notebooks"].items():
            original = json.loads((ROOT / name).read_text())
            exercise = PROJECTOR._project(entry, "en", original, "exercises")
            code = next(c for c in exercise["cells"] if c["cell_type"] == "code" and c["id"] != "course-language-switch")
            code["source"] = ["learner_code = 123\n"]
            exercise["cells"].insert(1, {"id": "learner-note", "cell_type": "markdown", "source": ["Keep my notes"], "metadata": {}})
            exercise["cells"].insert(2, {"id": "learner-scratch", "cell_type": "code", "source": ["scratch = input('try me')"], "metadata": {}, "outputs": [{"output_type": "stream", "name": "stdout", "text": "previous output"}], "execution_count": 1})
            solved = PROJECTOR._project(entry, "en", exercise, "solutions")
            if entry.get("role") != "assessment":
                self.assertNotIn("learner-scratch", {c["id"] for c in solved["cells"]})
            for language in GRAPH["languages"]:
                translated = PROJECTOR._project(entry, language, solved, "solutions")
                restored = PROJECTOR._project(entry, language, translated, "exercises")
                self.assertEqual("".join(next(c for c in restored["cells"] if c["id"] == code["id"])["source"]), "".join(code["source"]))
                self.assertIn("learner-note", {c["id"] for c in restored["cells"]})
                self.assertEqual(next(c for c in restored["cells"] if c["id"] == "learner-scratch")["source"], ["scratch = input('try me')"])
                self.assertEqual(next(c for c in restored["cells"] if c["id"] == "learner-scratch")["outputs"], [])
            if entry.get("role") != "assessment":
                next(c for c in solved["cells"] if c["id"] == code["id"])["source"] = ["my_solution = 456\n"]
                back = PROJECTOR._project(entry, "en", solved, "exercises")
                again = PROJECTOR._project(entry, "en", back, "solutions")
                self.assertEqual("".join(next(c for c in again["cells"] if c["id"] == code["id"])["source"]), "my_solution = 456\n")

    def test_language_only_keeps_mode(self):
        root, path = self.workspace()
        self.assertTrue(PROJECTOR.switch_language("en", root, path, "solutions"))
        self.assertTrue(PROJECTOR.switch_language("ja", root, path))
        for name, entry in GRAPH["notebooks"].items():
            if entry.get("role") != "assessment":
                self.assertEqual(json.loads((root / name).read_text())["metadata"]["course_mode"], "solutions")

    def test_reverted_code_and_deleted_scratch_cells_stay_removed(self):
        for name, entry in GRAPH["notebooks"].items():
            if entry.get("role") == "assessment":
                continue
            for mode, other in (("exercises", "solutions"), ("solutions", "exercises")):
                notebook = PROJECTOR._project(entry, "en", json.loads((ROOT / name).read_text()), mode)
                cell = next(c for c in notebook["cells"] if c["cell_type"] == "code" and c["id"] != "course-language-switch")
                canonical, cell_id = copy.deepcopy(cell["source"]), cell["id"]
                cell["source"] = ["temporary_edit = True\n"]
                notebook["cells"].append({"cell_type": "code", "id": "scratch", "metadata": {}, "source": ["x = 1"], "outputs": [], "execution_count": None})
                notebook = PROJECTOR._project(entry, "en", PROJECTOR._project(entry, "en", notebook, other), mode)
                next(c for c in notebook["cells"] if c["id"] == cell_id)["source"] = canonical
                notebook["cells"] = [c for c in notebook["cells"] if c["id"] != "scratch"]
                notebook = PROJECTOR._project(entry, "ja", PROJECTOR._project(entry, "ja", notebook, other), mode)
                self.assertEqual(next(c for c in notebook["cells"] if c["id"] == cell_id)["source"], canonical)
                self.assertNotIn("scratch", {c["id"] for c in notebook["cells"]})

    def test_map_discovers_every_solution_and_preserves_assessment(self):
        mapping = json.loads((ROOT / "composer/solutions_map.json").read_text())
        self.assertEqual(mapping["languages"], GRAPH["languages"])
        self.assertEqual(mapping["assessment_policy"], "unchanged")
        expected = {(name, c["id"]): c for name, entry in GRAPH["notebooks"].items()
                    for c in entry["cells"] if "solution" in c}
        rows = {(row["notebook"], row["cell_id"]): row for row in mapping["mappings"]}
        self.assertEqual(len(rows), len(mapping["mappings"]))
        self.assertEqual(set(rows), set(expected))
        for key, definition in expected.items():
            self.assertNotEqual(GRAPH["notebooks"][key[0]].get("role"), "assessment")
            self.assertEqual(rows[key]["exercise_sha256"], digest(definition["content"]))
            self.assertEqual(rows[key]["solution_sha256"], digest(definition["solution"]))
            self.assertEqual(rows[key]["sources"], [s["reference"] for s in rows[key]["source_cells"]])
            for source in rows[key]["source_cells"]:
                filename, cell_id = source["reference"].split("#")
                document = json.loads((ROOT / filename).read_text())
                cell = document["cells"][int(cell_id)] if cell_id.isdigit() else next(c for c in document["cells"] if c["id"] == cell_id)
                self.assertEqual(source["sha256"], digest("".join(cell["source"])), source["reference"])
        for name, entry in GRAPH["notebooks"].items():
            if any("ASSESSMENT TODO" in str(c["content"]) for c in entry["cells"]):
                self.assertEqual(entry.get("role"), "assessment", name)
                self.assertFalse(any("solution" in c for c in entry["cells"]))

    def test_original_todo_and_interaction_paths_remain_visible(self):
        for name, entry in GRAPH["notebooks"].items():
            for definition in entry["cells"]:
                if "solution" not in definition or definition["cell_type"] != "code":
                    continue
                comments = {line.strip().lstrip("#").strip() for line in definition["solution"].splitlines() if line.lstrip().startswith("#")}
                for line in definition["content"].splitlines():
                    if re.search(r"\bTODO\b|\b(?:input|getpass)\s*\(|\.launch\s*\(", line):
                        self.assertIn(line.strip().lstrip("#").strip(), comments, (name, definition["id"], line))

    def test_solution_code_compiles_without_interactive_calls_or_ultra(self):
        for name, entry in GRAPH["notebooks"].items():
            if entry.get("role") == "assessment":
                continue
            for definition in entry["cells"]:
                if definition["cell_type"] != "code":
                    continue
                source = definition.get("solution", definition["content"])
                if source.startswith("%%js"):
                    continue
                python = "\n".join(line for line in source.splitlines() if not line.startswith(("%", "!")))
                tree = compile(python, f"{name}:{definition['id']}", "exec", flags=ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        self.assertNotIn(node.func.id, {"input", "getpass"}, (name, definition["id"]))
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "launch":
                        self.fail(f"Unattended mode must not open an interactive UI: {name}:{definition['id']}")
                self.assertNotIn("nemotron-3-ultra", source)

    def test_new_removed_renamed_and_malformed_notebooks_fail_without_writes(self):
        for mutation in ("new", "removed", "renamed", "malformed", "duplicate-id", "missing-code"):
            root, path = self.workspace()
            name = next(iter(GRAPH["notebooks"]))
            target = root / name
            if mutation == "new":
                (root / "new-lesson.ipynb").write_bytes(target.read_bytes())
            elif mutation == "removed":
                target.unlink()
            elif mutation == "renamed":
                target.rename(root / "renamed-lesson.ipynb")
            elif mutation == "malformed":
                target.write_text("{broken")
            else:
                document = json.loads(target.read_text())
                if mutation == "duplicate-id":
                    document["cells"][1]["id"] = document["cells"][0]["id"]
                else:
                    document["cells"] = [c for c in document["cells"] if c["cell_type"] != "code"]
                target.write_text(json.dumps(document))
            before = {p: p.read_bytes() for p in root.glob("*.ipynb")}
            self.assertFalse(PROJECTOR.switch_language("en", root, path, "solutions"), mutation)
            self.assertEqual(before, {p: p.read_bytes() for p in root.glob("*.ipynb")})

    def test_unknown_locale_and_assessment_injection_fail(self):
        root, path = self.workspace()
        self.assertFalse(PROJECTOR.switch_language("unknown", root, path, "solutions"))
        graph = copy.deepcopy(GRAPH)
        assessment = next(e for e in graph["notebooks"].values() if e.get("role") == "assessment")
        assessment["cells"][0]["solution"] = "Injected answer"
        path.write_text(json.dumps(graph))
        self.assertFalse(PROJECTOR.switch_language("en", root, path, "solutions"))
        assessment["cells"][0].pop("solution")
        assessment.pop("role")
        path.write_text(json.dumps(graph))
        self.assertFalse(PROJECTOR.switch_language("en", root, path, "solutions"))

    def test_write_failure_rolls_back_the_whole_switch(self):
        root, path = self.workspace()
        before = {p: p.read_bytes() for p in root.glob("*.ipynb")}
        replace = PROJECTOR.os.replace
        calls = 0

        def fail_second(source, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("Injected second-file write failure")
            return replace(source, target)

        with patch.object(PROJECTOR.os, "replace", side_effect=fail_second):
            self.assertFalse(PROJECTOR.switch_language("ja", root, path, "solutions"))
        self.assertEqual(before, {p: p.read_bytes() for p in root.glob("*.ipynb")})
        self.assertFalse(list(root.glob("*.tmp")))
        self.assertFalse(list(root.glob("*.backup")))


if __name__ == "__main__":
    unittest.main()
