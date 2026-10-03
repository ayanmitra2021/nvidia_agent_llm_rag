#!/usr/bin/env python3
"""Project every learner notebook from one serialized localization graph."""

import argparse
import copy
import fcntl
import fnmatch
import json
import os
import re
import tempfile
from pathlib import Path


def _source(text):
    lines = text.split("\n")
    return [line + "\n" if index < len(lines) - 1 else line for index, line in enumerate(lines)]


def _cell(cell_type, text, definition):
    cell = {
        "cell_type": cell_type,
        "metadata": copy.deepcopy(definition.get("metadata", {})),
        "source": _source(text),
    }
    if definition.get("id"):
        cell["id"] = definition["id"]
    if cell_type == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def _project(entry, language, notebook, mode="exercises"):
    definitions = entry["cells"]
    ids = {definition.get("id"): index for index, definition in enumerate(definitions)}
    existing = {cell.get("id"): cell for cell in notebook.get("cells", []) if cell.get("id")}
    if len(ids) != len(definitions) or None in ids:
        raise ValueError("Canonical cells need unique IDs")
    if len(existing) != len(notebook.get("cells", [])):
        raise ValueError("Notebook cells need unique IDs before switching")
    assessment = entry.get("role") == "assessment"
    previous_mode = notebook.get("metadata", {}).get("course_mode", "exercises")
    saved = copy.deepcopy(notebook.get("metadata", {}).get("course_mode_edits", {}))
    if not assessment and mode != previous_mode:
        # Keep only learner edits. Canonical examples stay in the source graph.
        edits = {}
        for definition in definitions:
            if definition["cell_type"] != "code" or definition["id"] == "course-language-switch":
                continue
            current = existing.get(definition["id"])
            if current is None:
                raise ValueError(f"missing code cell {definition['id']!r}")
            canonical = definition.get("solution", definition["content"]) if previous_mode == "solutions" else definition["content"]
            source = "".join(current.get("source", []))
            if source != canonical:
                edits[definition["id"]] = source
        saved[previous_mode] = edits
    locale_ids = {extra.get("id") for values in entry.get("lang_extras", {}).values() for extra in values}
    learner_cells = {}
    anchor = -1
    for current in notebook.get("cells", []):
        if current["id"] in ids:
            anchor = ids[current["id"]]
        elif current["id"] not in locale_ids:
            extra = copy.deepcopy(current)
            if extra["cell_type"] == "code":
                extra.update(execution_count=None, outputs=[])
            learner_cells.setdefault(anchor, []).append(extra)
    saved_extra = copy.deepcopy(notebook.get("metadata", {}).get("course_mode_added_cells", {}))
    if not assessment and mode != previous_mode:
        saved_extra[previous_mode] = {str(anchor): cells for anchor, cells in learner_cells.items()}
        learner_cells = {int(anchor): cells for anchor, cells in saved_extra.get(mode, {}).items()}
    extras = {}
    for extra in entry.get("lang_extras", {}).get(language, []):
        if "after_id" in extra:
            after = -1 if extra["after_id"] is None else ids.get(extra["after_id"])
            if after is None:
                raise ValueError(f"unknown extra anchor {extra['after_id']!r}")
        else:
            after = extra["after_cell"]
        extras.setdefault(after, []).append(extra)

    cells = learner_cells.get(-1, []) + [_cell(extra["cell_type"], extra["content"], extra) for extra in extras.get(-1, [])]
    for index, definition in enumerate(definitions):
        if language not in definition.get("exclude_languages", []):
            content = definition.get("solution", definition["content"]) if mode == "solutions" and entry.get("role") != "assessment" else definition["content"]
            if definition["cell_type"] == "code" and definition["id"] != "course-language-switch" and (previous_mode == mode or assessment):
                current = existing.get(definition.get("id"))
                if current is None:
                    raise ValueError(f"missing preserved cell {definition.get('id')!r}")
                text = "".join(current.get("source", []))
            else:
                text = content[language] if isinstance(content, dict) else content
                if definition["cell_type"] == "code":
                    text = saved.get(mode, {}).get(definition["id"], text)
            if definition.get("id") == "course-language-switch":
                # Keep Run All in the selected mode after a CLI or notebook switch.
                text = re.sub(r'(?m)^MODE\s*=.*$', f'MODE = "{mode}"  # exercises, solutions', text)
                text = re.sub(r'(?m)^LANGUAGE\s*=.*$', f'LANGUAGE = "{language}"  # en, tw, zh, ja', text)
            cells.append(_cell(definition["cell_type"], text, definition))
        cells.extend(_cell(extra["cell_type"], extra["content"], extra) for extra in extras.get(index, []))
        cells.extend(learner_cells.get(index, []))

    projected = copy.deepcopy(notebook)
    projected["cells"] = cells
    projected["nbformat"] = 4
    projected["nbformat_minor"] = max(5, int(projected.get("nbformat_minor", 5)))
    if entry.get("role") != "assessment":
        projected.setdefault("metadata", {})["course_mode"] = mode
        projected["metadata"]["course_language"] = language
        if any(saved.values()):
            projected["metadata"]["course_mode_edits"] = saved
        else:
            projected["metadata"].pop("course_mode_edits", None)
        if any(saved_extra.values()):
            projected["metadata"]["course_mode_added_cells"] = saved_extra
        else:
            projected["metadata"].pop("course_mode_added_cells", None)
    return projected


def _switch_language(language, notebook_dir=".", graph_path="composer/translation_web.json", mode=None, preserve_extra_notebooks=False):
    notebook_dir = Path(notebook_dir)
    graph_path = Path(graph_path)
    if not graph_path.is_file():
        print(f"Error: {graph_path} not found")
        return False
    graph = json.loads(graph_path.read_text(encoding="utf-8"))

    if mode is not None and mode not in ("exercises", "solutions"):
        print("Error: mode must be exercises or solutions")
        return False

    languages = graph.get("languages")
    if not isinstance(languages, list) or not languages or len(languages) != len(set(languages)):
        print("Error: translation graph has invalid 'languages' metadata")
        return False
    if language not in languages:
        print(f"Error: unknown language '{language}'. Choose from: {', '.join(languages)}")
        return False

    notebooks = graph.get("notebooks")
    if not isinstance(notebooks, dict) or not notebooks:
        print("Error: translation graph has invalid 'notebooks' metadata")
        return False
    patterns = graph.get("notebook_globs", ["*.ipynb"])
    ignored_patterns = graph.get("ignored_notebook_globs", [])
    if (
        not isinstance(ignored_patterns, list)
        or any(not isinstance(pattern, str) or not pattern for pattern in ignored_patterns)
    ):
        print("Error: translation graph has invalid 'ignored_notebook_globs' metadata")
        return False
    discovered = {
        path.relative_to(notebook_dir).as_posix()
        for pattern in patterns
        for path in notebook_dir.glob(pattern)
        if path.is_file()
    }
    expected = set(notebooks)
    ignored = {
        relative
        for relative in discovered
        if any(fnmatch.fnmatch(relative.lower(), pattern.lower()) for pattern in ignored_patterns)
    }
    hidden = expected & ignored
    if hidden:
        print(f"Error: ignored notebook pattern matches canonical notebook(s): {', '.join(sorted(hidden))}")
        return False
    discovered -= ignored
    if expected - discovered or (discovered - expected and not preserve_extra_notebooks):
        print(
            "Error: translation graph and learner notebooks differ: "
            f"missing={', '.join(sorted(expected - discovered)) or 'none'}; "
            f"unexpected={', '.join(sorted(discovered - expected)) or 'none'}"
        )
        return False

    projected = {}
    try:
        current = {relative: json.loads((notebook_dir / relative).read_text(encoding="utf-8")) for relative in notebooks}
        if mode is None:
            modes = {notebook.get("metadata", {}).get("course_mode", "exercises") for relative, notebook in current.items() if notebooks[relative].get("role") != "assessment"}
            if len(modes) != 1 or not modes.issubset({"exercises", "solutions"}):
                raise ValueError("mixed notebook modes; choose --mode exercises or --mode solutions")
            mode = modes.pop()
        for relative, entry in notebooks.items():
            has_assessment = any("ASSESSMENT TODO" in str(c.get("content", "")) for c in entry.get("cells", []))
            if ("assessment" in relative.lower() or has_assessment) and entry.get("role") != "assessment":
                raise ValueError(f"{relative}: assessment notebook requires the assessment role")
            for index, definition in enumerate(entry.get("cells", [])):
                solution = definition.get("solution")
                if solution is not None:
                    if entry.get("role") == "assessment":
                        raise ValueError(f"{relative}: assessment cannot define solution-mode content")
                    if definition.get("cell_type") == "code" and not isinstance(solution, str):
                        raise ValueError(f"{relative} cell {index}: solution code must be shared across languages")
                    if not isinstance(solution, (str, dict)) or isinstance(solution, dict) and set(solution) != set(languages):
                        raise ValueError(f"{relative} cell {index}: incomplete solution languages")
                if definition.get("cell_type") == "code" and isinstance(definition.get("content"), dict):
                    raise ValueError(f"{relative} cell {index} has localized executable code")
                content = definition.get("content")
                if language not in definition.get("exclude_languages", []) and isinstance(content, dict) and language not in content:
                    raise ValueError(f"{relative} cell {index} has no content for '{language}'")
            for extra in entry.get("lang_extras", {}).get(language, []):
                if extra.get("cell_type") == "code":
                    raise ValueError(f"{relative} has locale-only executable code for '{language}'")
            notebook = current[relative]
            projected[relative] = _project(entry, language, notebook, mode)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: {error}")
        return False

    temporary, committed, created = [], [], []
    keep_backups = False
    try:
        for relative, notebook in projected.items():
            target = notebook_dir / relative
            descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
            created.append(Path(name))
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(notebook, stream, ensure_ascii=False, indent=1)
                stream.write("\n")
            descriptor, backup = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".backup", dir=target.parent)
            created.append(Path(backup))
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(target.read_bytes())
            os.chmod(name, target.stat().st_mode & 0o777)
            os.chmod(backup, target.stat().st_mode & 0o777)
            temporary.append((Path(name), Path(backup), target))
        for source, backup, target in temporary:
            os.replace(source, target)
            committed.append((backup, target))
    except OSError as error:
        for backup, target in reversed(committed):
            try:
                os.replace(backup, target)
            except OSError:
                keep_backups = True
                print(f"Restore {target} from {backup} before switching again.")
        print(f"Error: could not complete the notebook switch: {error}")
        return False
    finally:
        for path in created:
            if path.suffix != ".backup" or not keep_backups:
                path.unlink(missing_ok=True)

    print(f"Done! Switched {len(projected)} notebooks to {graph.get('lang_names', {}).get(language, language)} ({mode}). Assessment exercises are unchanged by mode.")
    return True


def switch_language(language, notebook_dir=".", graph_path="composer/translation_web.json", mode=None, preserve_extra_notebooks=False):
    """Serialize switches so two notebook cells cannot overwrite each other."""
    with (Path(notebook_dir) / ".course-mode.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another language or mode switch is still running. Try again when it finishes.")
            return False
        return _switch_language(language, notebook_dir, graph_path, mode, preserve_extra_notebooks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language")
    parser.add_argument("--mode", choices=("exercises", "solutions"), help="Keep the current mode when omitted")
    args = parser.parse_args()
    raise SystemExit(0 if switch_language(args.language.lower().strip(), mode=args.mode) else 1)
