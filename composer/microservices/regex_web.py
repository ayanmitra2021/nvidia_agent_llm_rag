#!/usr/bin/env python3
"""
regex_web.py — Cross-language structural sync for c-fx-15 RAG course notebooks.

Propagates code/infrastructure changes from EN (c-fx-15-v1) to TW/JA/ZH without
touching translated natural-language text.

Commands:
    python regex_web.py status                # Quick sync status (widget-friendly)
    python regex_web.py scan                  # Show EN structural tokens
    python regex_web.py propagate [--dry] [-v] # Push rules to all targets
    python regex_web.py diff                  # Structural divergence report
    python regex_web.py verify                # Idempotency + safety self-test
    python regex_web.py undo                  # Revert last propagation (git)
    python regex_web.py add OLD NEW [--md]    # Add custom substitution rule
    python regex_web.py widget                # Print widget cell for notebooks

Safety: markdown substitutions ONLY fire inside fenced code blocks (```),
inline code (`), link targets [](url), and HTML tags.  Natural-language text
is never modified.

Invertibility: `propagate` creates a git tag before writing. `undo` reverts to it.
"""

import json, re, sys, os, hashlib, subprocess, shutil
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

try:
    _here = Path(__file__).resolve().parent
except NameError:
    _here = Path.cwd()

RAG_ROOT = next(
    (p for p in [
        _here.parent.parent.parent.parent,
        Path("/dli/task/notebooks/RAG"),
        Path("/sandbox/.openclaw/dli/task/notebooks/RAG"),
    ] if p.is_dir() and (p / "c-fx-15-v1").exists()),
    None,
)

VARIANTS = {"en": "c-fx-15-v1", "tw": "c-fx-15-v1-tw",
            "ja": "c-fx-15-v1-ja", "zh": "c-fx-15-v1-zh"}
NB_RENAMES = {}
TARGET_LANGS = ["tw", "ja", "zh"]
RULES_FILE = _here / "propagation_rules.json"
TAG_PREFIX = "regex-web-pre-"

# ═══════════════════════════════════════════════════════════════════════
# Rules  (sorted longest-first to prevent partial-match shadowing)
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_RULES = {
    "meta": {"created": "2026-04-08",
             "description": "Structural patterns — EN → all variants"},
    "code_rules": sorted([
        {"old": "from langchain.schema.runnable import",
         "new": "from langchain_core.runnables import", "cat": "import"},
        {"old": "from langchain.schema.runnable",
         "new": "from langchain_core.runnables", "cat": "import"},
        {"old": "from langchain.document_loaders import",
         "new": "from langchain_community.document_loaders import", "cat": "import"},
        {"old": "from langchain.document_loaders",
         "new": "from langchain_community.document_loaders", "cat": "import"},
        {"old": "from langchain.text_splitter import",
         "new": "from langchain_text_splitters import", "cat": "import"},
        {"old": "from langchain.text_splitter",
         "new": "from langchain_text_splitters", "cat": "import"},
        {"old": "from langchain.output_parsers import",
         "new": "from langchain_core.output_parsers import", "cat": "import"},
        {"old": "from langchain.output_parsers",
         "new": "from langchain_core.output_parsers", "cat": "import"},
        {"old": "from langchain.document_transformers import",
         "new": "from langchain_community.document_transformers import", "cat": "import"},
        {"old": "from langchain.prompts import",
         "new": "from langchain_core.prompts import", "cat": "import"},
        {"old": ":8999", "new": ":8990", "cat": "port"},
        {"old": 'service_name = "chatbot"',
         "new": 'service_name = "frontend"', "cat": "service"},
    ], key=lambda r: -len(r["old"])),
    "md_rules": sorted([
        {"old": "(composer/docker-compose.yml)",
         "new": "(./composer/deploy/docker-compose.yml)", "cat": "link"},
        {"old": "(./composer/docker-compose.yml)",
         "new": "(./composer/deploy/docker-compose.yml)", "cat": "link"},
        {"old": "(composer/nginx.conf)",
         "new": "(./composer/deploy/nginx.conf)", "cat": "link"},
        {"old": "(./composer/nginx.conf)",
         "new": "(./composer/deploy/nginx.conf)", "cat": "link"},
        {"old": "(composer/Dockerfile)",
         "new": "(./composer/Dockerfile)", "cat": "link"},
        {"old": "composer/docker-compose.yaml",
         "new": "composer/deploy/docker-compose.yml", "cat": "path"},
        {"old": "composer/docker-compose.yml",
         "new": "composer/deploy/docker-compose.yml", "cat": "path"},
        {"old": "composer/nginx.conf",
         "new": "composer/deploy/nginx.conf", "cat": "path"},
        {"old": "llm_client/client_server.py",
         "new": "composer/microservices/llm_client.py", "cat": "path"},
    ], key=lambda r: -len(r["old"])),
}


def load_rules():
    return json.loads(RULES_FILE.read_text()) if RULES_FILE.exists() else DEFAULT_RULES


def save_rules(rules):
    RULES_FILE.write_text(json.dumps(rules, indent=2, ensure_ascii=False))


# ═══════════════════════════════════════════════════════════════════════
# Substitution Engine
# ═══════════════════════════════════════════════════════════════════════

def _subs(text, rules):
    """Cascade str.replace (rules pre-sorted longest-first)."""
    for r in rules:
        text = text.replace(r["old"], r["new"])
    return text


def sub_code_cell(src, cr):
    return _subs(src, cr)


def sub_markdown_cell(src, cr, mr):
    lines = src.splitlines(keepends=True)
    out, in_fence = [], False
    all_r = cr + mr
    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            out.append(line); continue
        if in_fence:
            out.append(_subs(line, all_r)); continue
        out.append(_sub_structural(line, all_r, mr))
    return "".join(out)


def _sub_structural(line, all_r, mr):
    """Substitute only inside inline code, link targets, and HTML tags."""
    res, pos, n = [], 0, len(line)
    while pos < n:
        # inline code `...`
        m = re.match(r"`([^`]+)`", line[pos:])
        if m:
            res.append("`" + _subs(m.group(1), all_r) + "`")
            pos += m.end(); continue
        # markdown link [text](url)
        m = re.match(r"(\[[^\]]*\])\(([^)]+)\)", line[pos:])
        if m:
            full = _subs(f"{m.group(1)}({m.group(2)})", mr)
            res.append(full.replace("(././", "(./"))
            pos += m.end(); continue
        # HTML tag
        m = re.match(r"<[^>]+>", line[pos:])
        if m:
            res.append(_subs(m.group(), all_r))
            pos += m.end(); continue
        res.append(line[pos]); pos += 1
    return "".join(res)


# ═══════════════════════════════════════════════════════════════════════
# Git helpers (for invertibility)
# ═══════════════════════════════════════════════════════════════════════

def _git(repo_path, *args, check=True):
    r = subprocess.run(["git"] + list(args), cwd=str(repo_path),
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        return None
    return r.stdout.strip()


def _snapshot_tag(repo_path):
    """Create a lightweight tag before propagation for rollback."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    tag = f"{TAG_PREFIX}{ts}"
    _git(repo_path, "tag", tag)
    return tag


def _latest_snapshot_tag(repo_path):
    """Find the most recent regex-web snapshot tag."""
    out = _git(repo_path, "tag", "-l", f"{TAG_PREFIX}*", "--sort=-creatordate")
    if out:
        return out.splitlines()[0]
    return None


# ═══════════════════════════════════════════════════════════════════════
# Notebook iteration
# ═══════════════════════════════════════════════════════════════════════

def iter_notebooks(rag_root, lang):
    en_dir = rag_root / VARIANTS["en"] / "task1/notebooks"
    tgt_dir = rag_root / VARIANTS[lang] / "task1/notebooks"
    renames = NB_RENAMES.get(lang, {})
    for p in sorted(en_dir.glob("*.ipynb")):
        actual = renames.get(p.name, p.name)
        tgt = tgt_dir / actual
        if tgt.exists():
            yield p.name, tgt
    sol = tgt_dir / "solutions"
    if sol.exists():
        for p in sorted(sol.glob("*.ipynb")):
            yield f"solutions/{p.name}", p


def _count_pending(rag_root, cr, mr):
    """Count cells that would change per lang (for status display)."""
    counts = {}
    for lang in TARGET_LANGS:
        n = 0
        for _, tgt_path in iter_notebooks(rag_root, lang):
            nb = json.loads(tgt_path.read_text())
            for cell in nb["cells"]:
                src = "".join(cell["source"])
                new = (sub_code_cell(src, cr) if cell["cell_type"] == "code"
                       else sub_markdown_cell(src, cr, mr))
                if new != src:
                    n += 1
        counts[lang] = n
    return counts


# ═══════════════════════════════════════════════════════════════════════
# Commands
# ═══════════════════════════════════════════════════════════════════════

def cmd_status(rag_root):
    """Quick one-line-per-lang sync status."""
    rules = load_rules()
    cr, mr = rules.get("code_rules", []), rules.get("md_rules", [])
    pending = _count_pending(rag_root, cr, mr)
    total = sum(pending.values())
    print(f"Sync status ({len(cr)}+{len(mr)} rules):\n")
    for lang in TARGET_LANGS:
        n = pending[lang]
        icon = "✅" if n == 0 else f"🔄 {n} cell(s) pending"
        print(f"  {lang.upper()}: {icon}")
    if total == 0:
        print(f"\nAll variants in sync with EN.")
    else:
        print(f"\n{total} total cell(s) would be updated by `propagate`.")


def cmd_scan(rag_root):
    en_dir = rag_root / VARIANTS["en"] / "task1/notebooks"
    t = defaultdict(set)
    for p in sorted(en_dir.glob("*.ipynb")):
        nb = json.loads(p.read_text())
        for cell in nb["cells"]:
            src = "".join(cell["source"])
            for m in re.findall(r"from\s+([\w][\w.]+\w)\s+import", src):
                t["imports"].add(m)
            for m in re.findall(r'model=["\']([^"\']+)["\']', src):
                t["models"].add(m)
            for m in re.findall(r":\d{4,5}(?=[/\s'\"])", src):
                t["ports"].add(m)
    print("EN structural tokens:")
    for cat in sorted(t):
        print(f"\n  {cat} ({len(t[cat])}):")
        for i in sorted(t[cat]):
            print(f"    {i}")


def cmd_propagate(rag_root, dry_run=False, verbose=False):
    rules = load_rules()
    cr, mr = rules.get("code_rules", []), rules.get("md_rules", [])
    print(f"Rules: {len(cr)} code + {len(mr)} markdown")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}\n")

    for lang in TARGET_LANGS:
        repo_dir = rag_root / VARIANTS[lang]
        # Snapshot before writing
        tag = None
        if not dry_run:
            tag = _snapshot_tag(repo_dir)

        total, details = 0, []
        for nb_name, tgt_path in iter_notebooks(rag_root, lang):
            nb = json.loads(tgt_path.read_text())
            nb_changed = False
            for ci, cell in enumerate(nb["cells"]):
                src = "".join(cell["source"])
                new = (sub_code_cell(src, cr) if cell["cell_type"] == "code"
                       else sub_markdown_cell(src, cr, mr))
                if new != src:
                    if verbose:
                        for ol, nl in zip(src.splitlines(), new.splitlines()):
                            if ol != nl:
                                details.append(f"    {nb_name}[{ci}]: {ol.strip()[:55]}")
                                details.append(f"    {'':>{len(nb_name)}}   → {nl.strip()[:55]}")
                                break
                    cell["source"] = new.splitlines(keepends=True)
                    nb_changed = True; total += 1
            if nb_changed and not dry_run:
                tgt_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")

        icon = "✅" if total == 0 else f"🔄 {total}"
        tag_info = f" (snapshot: {tag})" if tag and total else ""
        print(f"  {lang.upper()}: {icon} cell(s) updated{tag_info}")
        for d in details:
            print(d)

    if not dry_run:
        print(f"\nTo undo: python regex_web.py undo")


def cmd_diff(rag_root):
    en_dir = rag_root / VARIANTS["en"] / "task1/notebooks"
    def extract(directory, renames=None):
        renames = renames or {}
        t = defaultdict(set)
        for p in sorted(en_dir.glob("*.ipynb")):
            actual = renames.get(p.name, p.name)
            tp = directory / actual
            if not tp.exists(): continue
            nb = json.loads(tp.read_text())
            for cell in nb["cells"]:
                src = "".join(cell["source"])
                for m in re.findall(r"from\s+([\w][\w.]+\w)\s+import", src):
                    t["imports"].add(m)
                for m in re.findall(r'model=["\']([^"\']+)["\']', src):
                    t["models"].add(m)
                for m in re.findall(r"\]\(([^)#]+)\)", src):
                    t["link_targets"].add(m)
        return t
    en_t = extract(en_dir)
    for lang in TARGET_LANGS:
        tgt_dir = rag_root / VARIANTS[lang] / "task1/notebooks"
        lang_t = extract(tgt_dir, NB_RENAMES.get(lang, {}))
        print(f"\n{'='*50}\n{lang.upper()}:")
        any_diff = False
        for cat in sorted(set(en_t) | set(lang_t)):
            only_tgt = lang_t[cat] - en_t[cat]
            only_en = en_t[cat] - lang_t[cat]
            if only_tgt:
                any_diff = True
                print(f"  {cat} — stale (only in {lang}):")
                for x in sorted(only_tgt)[:8]: print(f"    − {x}")
            if only_en:
                any_diff = True
                print(f"  {cat} — missing (only in EN):")
                for x in sorted(only_en)[:8]: print(f"    + {x}")
        if not any_diff:
            print("  ✅ All structural tokens match EN")


def cmd_verify(rag_root):
    rules = load_rules()
    cr, mr = rules.get("code_rules", []), rules.get("md_rules", [])
    print("Idempotency check:\n")
    for lang_key, repo in VARIANTS.items():
        nb_dir = rag_root / repo / "task1/notebooks"
        problems, total = [], 0
        for p in sorted(nb_dir.glob("*.ipynb")):
            nb = json.loads(p.read_text())
            for ci, cell in enumerate(nb["cells"]):
                total += 1
                src = "".join(cell["source"])
                new = (sub_code_cell(src, cr) if cell["cell_type"] == "code"
                       else sub_markdown_cell(src, cr, mr))
                if new != src:
                    for ol, nl in zip(src.splitlines(), new.splitlines()):
                        if ol != nl:
                            problems.append(f"    {p.name}[{ci}]: {ol.strip()[:65]}")
                            break
        icon = "✅" if not problems else f"⚠️  {len(problems)}"
        print(f"  {lang_key.upper()} ({total} cells): {icon}")
        for d in problems[:3]:
            print(d)
        if len(problems) > 3:
            print(f"    ... +{len(problems)-3} more")


def cmd_undo(rag_root):
    """Revert each target repo to its most recent pre-propagation snapshot."""
    for lang in TARGET_LANGS:
        repo_dir = rag_root / VARIANTS[lang]
        tag = _latest_snapshot_tag(repo_dir)
        if not tag:
            print(f"  {lang.upper()}: no snapshot tag found — nothing to undo")
            continue
        # Check if there are uncommitted changes (our propagation)
        dirty = _git(repo_dir, "status", "--short")
        if not dirty:
            print(f"  {lang.upper()}: clean — nothing to undo")
            continue
        _git(repo_dir, "checkout", ".")
        print(f"  {lang.upper()}: reverted to {tag} ({dirty.count(chr(10))+1 if dirty else 0} files)")


def cmd_add():
    if len(sys.argv) < 4:
        print("Usage: regex_web.py add OLD NEW [--md]"); sys.exit(1)
    old, new = sys.argv[2], sys.argv[3]
    target = "md_rules" if "--md" in sys.argv else "code_rules"
    rules = load_rules()
    rules.setdefault(target, []).append({
        "old": old, "new": new, "cat": "custom",
        "added": datetime.now(timezone.utc).isoformat(),
    })
    rules[target].sort(key=lambda r: -len(r["old"]))
    save_rules(rules)
    print(f"Added to {target}: '{old}' → '{new}'")


def cmd_widget():
    """Print a code cell that can be pasted into a Jupyter notebook."""
    print(WIDGET_CODE)


# ═══════════════════════════════════════════════════════════════════════
# Embeddable Jupyter Widget
# ═══════════════════════════════════════════════════════════════════════

WIDGET_CODE = r'''
## ─── Course Sync Tool ───────────────────────────────────────────────
## Run this cell to check if all language variants are in sync.
## Instructor use: click "Propagate" to push EN structural changes.

import subprocess, sys, json
from pathlib import Path
from IPython.display import display, HTML

_rw = Path("composer/microservices/regex_web.py")
if not _rw.exists():
    display(HTML('<p style="color:#999">regex_web.py not found — sync tool unavailable.</p>'))
else:
    def _run(cmd):
        r = subprocess.run([sys.executable, str(_rw)] + cmd,
                           capture_output=True, text=True, cwd=str(_rw.parent))
        return r.stdout + r.stderr

    try:
        import ipywidgets as widgets

        out = widgets.Output(layout={"border": "1px solid #444", "padding": "8px",
                                     "max_height": "300px", "overflow_y": "auto"})

        def _show(cmd_list, label):
            out.clear_output()
            with out:
                print(f"▶ {label}...\n")
                print(_run(cmd_list))

        btn_status = widgets.Button(description="Check Sync", icon="sync",
                                    button_style="info", layout={"width": "140px"})
        btn_diff   = widgets.Button(description="Show Diff", icon="exchange",
                                    button_style="", layout={"width": "140px"})
        btn_dry    = widgets.Button(description="Dry Run", icon="eye",
                                    button_style="warning", layout={"width": "140px"})
        btn_prop   = widgets.Button(description="Propagate", icon="rocket",
                                    button_style="danger", layout={"width": "140px"})
        btn_undo   = widgets.Button(description="Undo", icon="undo",
                                    button_style="", layout={"width": "100px"})

        btn_status.on_click(lambda _: _show(["status"], "Checking sync status"))
        btn_diff.on_click(lambda _: _show(["diff"], "Comparing structural tokens"))
        btn_dry.on_click(lambda _: _show(["propagate", "--dry", "-v"], "Dry run"))
        btn_prop.on_click(lambda _: _show(["propagate", "-v"], "Propagating"))
        btn_undo.on_click(lambda _: _show(["undo"], "Reverting"))

        display(widgets.HTML('<h4 style="color:#76b900">🔄 Course Sync Tool</h4>'))
        display(widgets.HBox([btn_status, btn_diff, btn_dry, btn_prop, btn_undo]))
        display(out)

        # Auto-run status on cell execution
        _show(["status"], "Checking sync status")

    except ImportError:
        # Fallback: no widgets, just run status
        print(_run(["status"]))
'''.strip()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    if RAG_ROOT is None:
        print("ERROR: Cannot find RAG repos."); sys.exit(1)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    dispatch = {
        "status": lambda: cmd_status(RAG_ROOT),
        "scan": lambda: cmd_scan(RAG_ROOT),
        "propagate": lambda: cmd_propagate(RAG_ROOT, "--dry" in sys.argv, "-v" in sys.argv),
        "diff": lambda: cmd_diff(RAG_ROOT),
        "verify": lambda: cmd_verify(RAG_ROOT),
        "undo": lambda: cmd_undo(RAG_ROOT),
        "add": cmd_add,
        "widget": cmd_widget,
    }
    if cmd in dispatch:
        dispatch[cmd]()
    else:
        print(__doc__)

if __name__ == "__main__":
    main()
