#!/usr/bin/env python3
"""
NVIDIA DLI Content Processor
============================
Tool-based content exploration with streaming LLM/VLM, thinking visualization,
and Gradio UI. Cleaned and consolidated implementation.
"""

import os
import re
import io
import json
import base64
import logging
import time
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Generator, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from html import escape as html_escape

import gradio as gr
import PIL.Image
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

try:
    import nbformat
    HAS_NBFORMAT = True
except ImportError:
    HAS_NBFORMAT = False

# =============================================================================
# CONFIGURATION
# =============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# Server
PORT = int(os.environ.get("PORT", 7860))
ROOT_PATH = os.environ.get("ROOT_PATH", None)

# Directories
BASE_DIR = os.environ.get("NOTEBOOK_DIR", "/docs/notebooks")
IMG_DIR = os.path.join(BASE_DIR, "imgs")
SLIDES_DIR = os.path.join(BASE_DIR, "slides")
SYNTH_DIR = os.environ.get("SYNTH_DIR", "/docs/synth")
DEV_DIR = os.environ.get("DEV_DIR", "/docs/dev")
for d in [SYNTH_DIR, DEV_DIR]:
    os.makedirs(d, exist_ok=True)

# LLM
LLM_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "http://llm_client:9000/v1")

# Limits
CHARS_PER_TOKEN = 4
MAX_RECOMMENDED_TOKENS = 32000

# Models
LLM_MODELS = {
    "nvidia/nemotron-3.5-lightning-30b-a3b": "nvidia/nemotron-3.5-lightning-30b-a3b",
}
VLM_MODELS = {
    "meta/llama-3.2-11b-vision-instruct": "meta/llama-3.2-11b-vision-instruct",
}
DEFAULT_LLM = "nvidia/nemotron-3.5-lightning-30b-a3b"
DEFAULT_VLM = "meta/llama-3.2-11b-vision-instruct"

# =============================================================================
# PRESETS
# =============================================================================
RECIPE_PRESETS = {
    "Summary": "Provide a comprehensive technical summary including overview, key concepts, prerequisites, tools, code patterns, and applications. Enumerate and explain every notebook so a professor could reconstruct the course narrative.",
    "Quiz": "Create 5 multiple choice questions (4 options, mark correct, explain) and 2 coding challenges (problem, example, solution).",
    "Study Guide": "Create a study guide: learning objectives, concept relationships, key explanations, takeaways, practice suggestions.",
    "Key Points": "Extract key points: main arguments, definitions, code patterns, concept relationships. Be concise.",
    "Explain Code": "Explain each code block: what it does, how it works, why, and pitfalls. Assume intermediate Python.",
    "Agent Mode": "Analyze content and determine which tools to use. Search, extract code, list files as needed. Explain reasoning.",
    "Custom": ""
}

SYSTEM_PRESETS = {
    "Teaching Assistant": "You are an expert teaching assistant for NVIDIA Deep Learning Institute. Provide clear, accurate, educational explanations.",
    "Code Reviewer": "You are a senior code reviewer. Analyze for correctness, efficiency, style, bugs. Suggest improvements.",
    "Q&A Expert": "Answer questions from provided context. If unavailable, say so. Be concise and accurate.",
    "Socratic Tutor": "Guide learning through questions. Don't give direct answers - ask probing questions.",
    "Tool Agent": "You are an agent with tools for file access, search, and extraction. Explain reasoning before acting.",
    "Custom": ""
}

# =============================================================================
# UTILITIES
# =============================================================================
def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN if text else 0

def safe_read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def safe_write(path: str, content: str) -> Optional[str]:
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return None
    except Exception as e:
        return str(e)

def fmt_static(text: str) -> str:
    """Rewrite image paths for static serving."""
    for old, new in [('](imgs/', '](/static/imgs/'), ('](slides/', '](/static/slides/'),
                     ('src="imgs/', 'src="/static/imgs/'), ('src="slides/', 'src="/static/slides/')]:
        text = text.replace(old, new)
    return text

# =============================================================================
# RENDERING HELPERS
# =============================================================================
def status_html(text: str, pulse: bool = False) -> str:
    safe = html_escape(text or "")
    return f"<span class='thinking'>{safe}</span>" if pulse else safe

def render_thinking_box(thinking: str) -> str:
    if not thinking or not thinking.strip():
        return ""
    return f"<details class='think-box' open><summary>💭 Thinking</summary>\n\n{html_escape(thinking)}\n\n</details>"

def render_tool_calls(calls: List[Dict]) -> str:
    lines = ["**🔧 Tool calls:**"]
    for tc in calls:
        fn = tc.get("function", {}) or {}
        lines.append(f"- **{html_escape(fn.get('name', ''))}**: `{fn.get('arguments', '')}`")
    return "\n".join(lines)

def pretty_json(obj_or_str: str, max_len: int = 1200) -> str:
    try:
        parsed = json.loads(obj_or_str)
        text = json.dumps(parsed, indent=2)
    except Exception:
        text = obj_or_str
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return text

def render_tool_result(name: str, result: str, max_len: int = 1200) -> str:
    preview = pretty_json(result)
    return f"<details class='tool-call'><summary>🔧 Tool result: <code>{html_escape(name)}</code></summary>\n\n```json\n{preview}\n```\n\n</details>"

def render_turn(thinking: str, content: str, ui_blocks: str = "", status: str = "") -> str:
    parts = [p for p in [ui_blocks, render_thinking_box(thinking), content] if p and p.strip()]
    if status:
        parts.append(f"<div class='status-line'>{status}</div>")
    return "\n\n".join(parts)

def format_thinking_tags(text: str) -> str:
    """Convert inline thinking tags to collapsible boxes."""
    patterns = [
        (r'<think>(.*?)(</think>|$)', 'Thinking'),
        (r'<thinking>(.*?)(</thinking>|$)', 'Thinking'),
        (r'<reasoning>(.*?)(</reasoning>|$)', 'Reasoning'),
    ]
    for pattern, label in patterns:
        text = re.sub(pattern, rf'<details class="think-box"><summary>💭 {label}...</summary>\n\n\1\n\n</details>', 
                      text, flags=re.DOTALL | re.IGNORECASE)
    return text

# =============================================================================
# EXECUTION PLAN (ReWOO-style progress)
# =============================================================================
@dataclass
class ExecutionStep:
    id: str
    action: str
    status: str = "pending"
    result: str = ""
    duration_ms: int = 0

@dataclass 
class ExecutionPlan:
    steps: List[ExecutionStep] = field(default_factory=list)
    
    def add_step(self, action: str) -> str:
        step_id = f"step_{len(self.steps)+1}"
        self.steps.append(ExecutionStep(id=step_id, action=action))
        return step_id
    
    def start_step(self, step_id: str):
        for s in self.steps:
            if s.id == step_id:
                s.status = "running"
                break
    
    def complete_step(self, step_id: str, result: str = "", duration_ms: int = 0):
        for s in self.steps:
            if s.id == step_id:
                s.status = "done"
                s.result = result[:200] + "..." if len(result) > 200 else result
                s.duration_ms = duration_ms
                break
    
    def fail_step(self, step_id: str, error: str):
        for s in self.steps:
            if s.id == step_id:
                s.status = "error"
                s.result = error
                break
    
    def render(self) -> str:
        icons = {"pending": "⏳", "running": "🔄", "done": "✅", "error": "❌"}
        lines = ["### 📋 Execution Progress\n"]
        for s in self.steps:
            line = f"- {icons.get(s.status, '⏳')} **{s.action}**"
            if s.status == "done" and s.duration_ms:
                line += f" ({s.duration_ms}ms)"
            if s.status == "error":
                line += f" — {s.result}"
            lines.append(line)
        return "\n".join(lines)

# =============================================================================
# RESOURCE SERVER
# =============================================================================
@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]
    
    def to_openai(self) -> Dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


class ContentResourceServer:
    """Resource server for content exploration with tool-based access."""
    
    def __init__(self):
        self.base_dir = Path(BASE_DIR)
        self.synth_dir = Path(SYNTH_DIR)
        self.dev_dir = Path(DEV_DIR)
        self._file_index: Dict[str, Dict] = {}
        self._search_index: Dict[str, List[Dict]] = {}
        self._rebuild_index()
    
    def _rebuild_index(self):
        self._file_index.clear()
        self._search_index.clear()
        
        categories = [
            ("notebooks", self.base_dir, ('.ipynb', '.md', '.txt', '.py', '.json')),
            ("synth", self.synth_dir, ('.md', '.txt', '.json')),
            ("dev", self.dev_dir, ('.ipynb', '.md', '.txt', '.py', '.json')),
        ]
        
        for category, directory, extensions in categories:
            if not directory.exists():
                continue
            for fp in directory.rglob("*"):
                if not fp.is_file():
                    continue
                if fp.suffix not in extensions:
                    continue
                if fp.name.startswith('.'):
                    continue
                if "/." in fp.name:
                    continue
                try:
                    size = fp.stat().st_size
                    self._file_index[str(fp)] = {
                        "path": str(fp), "name": fp.name, "category": category,
                        "extension": fp.suffix, "size_bytes": size,
                        "estimated_tokens": size // CHARS_PER_TOKEN,
                    }
                    self._index_content(str(fp), self._read_file(str(fp)))
                except Exception:
                    pass
    
    def _read_file(self, path: str) -> str:
        if path.endswith('.ipynb'):
            return self._notebook_to_text(path)
        return safe_read(path)
    
    def _notebook_to_text(self, path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                nb = json.load(f) if not HAS_NBFORMAT else nbformat.read(f, as_version=4)
            
            if HAS_NBFORMAT:
                parts = [f"# {os.path.basename(path)}\n---\n"]
                for cell in nb.cells:
                    src = cell.source.strip()
                    if not src:
                        continue
                    if cell.cell_type == 'markdown':
                        parts.append(f"{src}\n\n")
                    elif cell.cell_type == 'code':
                        parts.append(f"```python\n{src}\n```\n\n")
            else:
                parts = [f"# {os.path.basename(path)}\n"]
                for cell in nb.get('cells', []):
                    source = ''.join(cell.get('source', []))
                    if cell.get('cell_type') == 'markdown':
                        parts.append(f"{source}\n\n")
                    elif cell.get('cell_type') == 'code':
                        parts.append(f"```python\n{source}\n```\n\n")
            return ''.join(parts)
        except Exception as e:
            return f"[Error reading notebook: {e}]"
    
    def _index_content(self, path: str, content: str):
        sections = []
        parts = re.split(r'^(#{1,4}\s+.+)$', content, flags=re.MULTILINE)
        current_heading, current_content = "Introduction", ""
        
        for part in parts:
            if re.match(r'^#{1,4}\s+', part):
                if current_content.strip():
                    sections.append({"heading": current_heading, "content": current_content.strip()[:2000]})
                current_heading = part.strip('# \n')
                current_content = ""
            else:
                current_content += part
        
        if current_content.strip():
            sections.append({"heading": current_heading, "content": current_content.strip()[:2000]})
        self._search_index[path] = sections
    
    def refresh_index(self):
        self._rebuild_index()
    
    # === TOOL DEFINITIONS ===
    def get_tools(self) -> List[Dict]:
        tools = [
            ToolDefinition("list_files", "List available course files.", {"type": "object", "properties": {"category": {"type": "string", "enum": ["all", "notebooks", "synth", "dev"], "default": "all"}}, "required": []}),
            ToolDefinition("get_file_info", "Get file info and preview.", {"type": "object", "properties": {"path": {"type": "string"}, "preview_length": {"type": "integer", "default": 500}}, "required": ["path"]}),
            ToolDefinition("load_content", "Load full file contents.", {"type": "object", "properties": {"paths": {"type": "array", "items": {"type": "string"}}, "max_chars": {"type": "integer", "default": 0}}, "required": ["paths"]}),
            ToolDefinition("search_content", "Search across content for keywords.", {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer", "default": 5}}, "required": ["query"]}),
            ToolDefinition("extract_code", "Extract code blocks from files.", {"type": "object", "properties": {"paths": {"type": "array", "items": {"type": "string"}}, "language": {"type": "string", "default": ""}}, "required": ["paths"]}),
            ToolDefinition("extract_urls", "Extract URLs from files.", {"type": "object", "properties": {"paths": {"type": "array", "items": {"type": "string"}}}, "required": ["paths"]}),
            ToolDefinition("extract_images", "Extract image references from files.", {"type": "object", "properties": {"paths": {"type": "array", "items": {"type": "string"}}}, "required": ["paths"]}),
            ToolDefinition("get_learning_objectives", "Extract learning objectives.", {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
            ToolDefinition(
                "list_notebooks",
                "List all Jupyter notebooks (.ipynb) in the dev/notebooks folder.",
                {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "enum": ["dev"], "default": "dev"},
                        "extension": {"type": "string", "enum": [".ipynb"], "default": ".ipynb"},
                    },
                    "required": [],
                },
            ),
        ]
        return [t.to_openai() for t in tools]
    
    def get_tool_names(self) -> List[str]:
        return [t["function"]["name"] for t in self.get_tools()]
    
    # === TOOL EXECUTION ===
    def execute_tool(self, name: str, args: Dict) -> str:
        handlers = {
            "list_files": self._list_files,
            "get_file_info": self._get_file_info,
            "load_content": self._load_content,
            "search_content": self._search_content,
            "extract_code": self._extract_code,
            "extract_urls": self._extract_urls,
            "extract_images": self._extract_images,
            "get_learning_objectives": self._get_learning_objectives,
        }
        handler = handlers.get(name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            return json.dumps(handler(args), ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def _resolve_path(self, path: str) -> Optional[str]:
        """Resolve path by exact match or name match."""
        if path in self._file_index:
            return path
        for p, info in self._file_index.items():
            if info["name"] == path or path in p:
                return p
        return None
    
    def _list_files(self, args: Dict) -> Dict:
        category = args.get("category", "all")
        extension = args.get("extension", "")
        files = [
            info for info in sorted(self._file_index.values(), key=lambda x: x["name"])
            if (category == "all" or info["category"] == category)
            and (extension == "" or info["extension"] == extension)
        ]
        return {"files": files, "total": len(files)}
    
    def _get_file_info(self, args: Dict) -> Dict:
        path = self._resolve_path(args.get("path") or "")
        if not path:
            return {"error": f"File not found: {args.get('path')}"}
        
        info = self._file_index[path].copy()
        content = self._read_file(path)
        preview_len = args.get("preview_length", 500)
        
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        info["title"] = title_match.group(1).strip() if title_match else info["name"]
        info["section_count"] = len(re.findall(r'^#{1,3}\s+', content, re.MULTILINE))
        info["preview"] = content[:preview_len] + ("..." if len(content) > preview_len else "")
        return info
    
    def _load_content(self, args: Dict) -> Dict:
        paths, max_chars = args.get("paths", []), args.get("max_chars", 0)
        results, total_tokens = [], 0
        
        for p in paths:
            resolved = self._resolve_path(p)
            if not resolved:
                results.append({"path": p, "error": "Not found"})
                continue
            
            content = self._read_file(resolved)
            if max_chars > 0 and len(content) > max_chars:
                half = max_chars // 2
                content = f"{content[:half]}\n\n[...truncated...]\n\n{content[-half:]}"
            
            tokens = len(content) // CHARS_PER_TOKEN
            total_tokens += tokens
            results.append({"path": resolved, "name": self._file_index[resolved]["name"], "content": content, "tokens": tokens})
        
        return {"files": results, "total_tokens": total_tokens}
    
    def _search_content(self, args: Dict) -> Dict:
        query = (args.get("query") or "").lower()
        if not query:
            return {"error": "No query provided"}
        
        results = []
        for path, sections in self._search_index.items():
            for section in sections:
                content_lower = section["content"].lower()
                if query in content_lower:
                    count = content_lower.count(query)
                    score = count / (len(content_lower) / 1000 + 1)
                    idx = content_lower.find(query)
                    snippet = f"...{section['content'][max(0, idx-100):idx+len(query)+100]}..."
                    results.append({"path": path, "section": section["heading"], "snippet": snippet, "score": round(score, 3)})
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return {"query": query, "results": results[:args.get("max_results", 5)], "total_matches": len(results)}
    
    def _extract_code(self, args: Dict) -> Dict:
        paths, lang_filter = args.get("paths", []), (args.get("language") or "").lower()
        blocks = []
        
        for p in paths:
            resolved = self._resolve_path(p)
            if not resolved:
                continue
            content = self._read_file(resolved)
            for match in re.finditer(r'```(\w*)\n(.*?)\n```', content, re.DOTALL):
                lang, code = match.group(1) or "text", match.group(2)
                if lang_filter and lang.lower() != lang_filter:
                    continue
                blocks.append({"source": resolved, "language": lang, "code": code, "lines": len(code.split('\n'))})
        
        return {"code_blocks": blocks, "total": len(blocks)}
    
    def _extract_urls(self, args: Dict) -> Dict:
        paths = args.get("paths", [])
        urls, seen = [], set()
        
        for p in paths:
            resolved = self._resolve_path(p)
            if not resolved:
                continue
            content = self._read_file(resolved)
            for match in re.finditer(r'https?://[^\s<>"\')\]]+[^\s<>"\')\].,;:!?]', content):
                url = match.group(0)
                if url not in seen:
                    seen.add(url)
                    urls.append({"url": url, "source": resolved})
        
        return {"urls": urls, "total": len(urls)}
    
    def _extract_images(self, args: Dict) -> Dict:
        paths = args.get("paths", [])
        images, seen = [], set()
        
        for p in paths:
            resolved = self._resolve_path(p)
            if not resolved:
                continue
            content = self._read_file(resolved)
            
            for pattern in [r'!\[([^\]]*)\]\(([^)]+)\)', r'<img[^>]+src=["\']([^"\']+)["\']']:
                for match in re.finditer(pattern, content):
                    if len(match.groups()) == 2:
                        alt, img_path = match.groups()
                    else:
                        alt, img_path = "", match.group(1)
                    
                    img_path = img_path.split('?')[0].strip()
                    if img_path in seen or img_path.startswith(('data:', 'http')):
                        continue
                    seen.add(img_path)
                    
                    resolved_img = None
                    for candidate in [img_path, str(self.base_dir / img_path), 
                                     str(self.base_dir / "imgs" / os.path.basename(img_path)),
                                     str(self.base_dir / "slides" / os.path.basename(img_path))]:
                        if os.path.exists(candidate):
                            resolved_img = os.path.abspath(candidate)
                            break
                    
                    images.append({"original": img_path, "resolved": resolved_img, "alt": alt, 
                                  "source": resolved, "exists": resolved_img is not None})
        
        return {"images": images, "total": len(images)}
    
    def _get_learning_objectives(self, args: Dict) -> Dict:
        path = self._resolve_path(args.get("path") or "")
        if not path:
            return {"error": f"File not found: {args.get('path')}"}
        
        content = self._read_file(path)
        objectives = []
        
        for pattern in [r'Learning\s+Objectives?\s*\n+((?:[-*]\s+.+\n?)+)',
                       r'\*\*Learning\s+Objectives?\*\*\s*\n+((?:[-*]\s+.+\n?)+)',
                       r'By\s+the\s+end.+?:\s*\n+((?:[-*]\s+.+\n?)+)']:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                for line in re.findall(r'[-*]\s*(.+)$', match.group(1), re.MULTILINE):
                    obj = line.strip().strip('*').strip()
                    if obj and obj not in objectives:
                        objectives.append(obj)
                break
        
        return {"path": path, "learning_objectives": objectives or ["No explicit learning objectives found"], "count": len(objectives)}
    
    # === CONVENIENCE ===
    def list_all_sources(self) -> List[Tuple[str, str]]:
        return [
            (os.path.relpath(path, BASE_DIR), path)
            for path, info in sorted(
                self._file_index.items(),
                key=lambda x: (x[1]["category"], x[0])
            )
        ]    
    def list_synth_files(self) -> List[Tuple[str, str]]:
        return [(info["name"], path) for path, info in self._file_index.items() if info["category"] == "synth"]
    
    def get_raw_content(self, paths: List[str], max_chars: int = 0) -> str:
        parts = []
        for p in paths:
            resolved = self._resolve_path(p)
            if resolved:
                content = self._read_file(resolved)
                if max_chars > 0 and len(content) > max_chars:
                    half = max_chars // 2
                    content = f"{content[:half]}\n\n[...truncated...]\n\n{content[-half:]}"
                parts.append(content)
        return "\n\n---\n\n".join(parts)


resource_server = ContentResourceServer()

# =============================================================================
# STREAMING LLM CLIENT
# =============================================================================
class StreamingLLM:
    def __init__(self, model: str, base_url: str = LLM_BASE_URL, temperature: float = 0.2, max_tokens: int = 4000):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def stream(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> Generator[Dict, None, None]:
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature,
                   "max_tokens": self.max_tokens, "stream": True,
                   "chat_template_kwargs": {"enable_thinking": False}}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        try:
            response = requests.post(f"{self.base_url}/chat/completions", json=payload,
                                    headers={"Content-Type": "application/json"}, stream=True, timeout=300)
            response.raise_for_status()
            
            current_tool_calls = {}
            in_thinking = False
            
            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode('utf-8')
                if not line.startswith('data: ') or line[6:] == '[DONE]':
                    continue
                
                try:
                    chunk = json.loads(line[6:])
                    
                    # Nemotron metadata thinking
                    if 'metadata' in chunk and 'reasoning_content' in chunk['metadata']:
                        yield {"type": "thinking", "content": chunk['metadata']['reasoning_content']}
                        in_thinking = True
                    
                    choices = chunk.get('choices', [])
                    if not choices:
                        continue
                    delta = choices[0].get('delta', {})
                    
                    # Reasoning in delta
                    if 'reasoning' in delta:
                        yield {"type": "thinking", "content": delta['reasoning']}
                        in_thinking = True
                    
                    # Tool calls
                    if delta.get('tool_calls'):
                        for tc in delta['tool_calls']:
                            idx = tc.get('index', 0)
                            if idx not in current_tool_calls:
                                current_tool_calls[idx] = {"id": tc.get('id') or f"call_{idx}", "type": "function", "function": {"name": "", "arguments": ""}}
                            if 'id' in tc and tc['id']:
                                current_tool_calls[idx]['id'] = tc['id']
                            if 'function' in tc:
                                if 'name' in tc['function']:
                                    current_tool_calls[idx]['function']['name'] += tc['function']['name']
                                if 'arguments' in tc['function']:
                                    current_tool_calls[idx]['function']['arguments'] += tc['function']['arguments']
                        yield {"type": "tool_calls", "calls": list(current_tool_calls.values())}
                    
                    # Content
                    if delta.get('content'):
                        if in_thinking:
                            in_thinking = False
                            yield {"type": "end_thinking"}
                        yield {"type": "content", "content": delta['content']}
                
                except json.JSONDecodeError:
                    continue
            
            if current_tool_calls:
                yield {"type": "tool_calls_final", "calls": list(current_tool_calls.values())}
                
        except Exception as e:
            yield {"type": "error", "content": str(e)}
    
    def analyze_image(self, path: str, prompt: str) -> Generator[str, None, None]:
        """Stream VLM image analysis."""
        try:
            with PIL.Image.open(path) as img:
                fmt = 'PNG' if path.lower().endswith('.png') else 'JPEG'
                if img.mode == 'RGBA' and fmt == 'JPEG':
                    img = img.convert('RGB')
                if max(img.size) > 1024:
                    ratio = 1024 / max(img.size)
                    img = img.resize((int(img.width * ratio), int(img.height * ratio)), PIL.Image.LANCZOS)
                buffer = io.BytesIO()
                img.save(buffer, format=fmt, quality=85)
                b64 = f"data:image/{fmt.lower()};base64,{base64.b64encode(buffer.getvalue()).decode()}"
        except Exception as e:
            yield f"❌ Image error: {e}"
            return
        
        try:
            response = requests.post(f"{self.base_url}/chat/completions",
                json={"model": self.model, "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": b64}}, {"type": "text", "text": prompt}
                ]}], "temperature": self.temperature, "max_tokens": self.max_tokens, "stream": True},
                headers={"Content-Type": "application/json"}, stream=True, timeout=300)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode('utf-8')
                if line.startswith('data: ') and line[6:] != '[DONE]':
                    try:
                        delta = json.loads(line[6:]).get('choices', [{}])[0].get('delta', {})
                        if delta.get('content'):
                            yield delta['content']
                    except:
                        pass
        except Exception as e:
            yield f"\n\n❌ Error: {e}"

# =============================================================================
# HISTORY HELPERS
# =============================================================================
def normalize_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join((item.get("text") or "") if isinstance(item, dict) else str(item) for item in content)
    if isinstance(content, dict):
        return content.get("text") or ""
    return str(content)

def normalize_history(history: Any) -> List[Dict[str, str]]:
    if not history:
        return []
    if isinstance(history, list) and history and isinstance(history[0], dict):
        return [{"role": str(m.get("role", "assistant")), "content": normalize_content(m.get("content"))} for m in history]
    if isinstance(history, list) and history and isinstance(history[0], (list, tuple)):
        out = []
        for u, a in history:
            if u:
                out.append({"role": "user", "content": normalize_content(u)})
            if a:
                out.append({"role": "assistant", "content": normalize_content(a)})
        return out
    return []

def update_last_assistant(history: List[Dict], content: str) -> List[Dict]:
    if not history:
        return [{"role": "assistant", "content": content}]
    if history[-1].get("role") == "assistant":
        return history[:-1] + [{"role": "assistant", "content": content}]
    return history + [{"role": "assistant", "content": content}]

# =============================================================================
# CHAT HANDLER
# =============================================================================
def chat_stream(message: str, history: List[Dict], context: str, synth_context: str, 
                system: str, temp: float, max_tokens: int, model: str, enabled_tools: List[str]
               ) -> Generator[tuple, None, None]:
    """Stream chat with tool support."""
    
    hist = normalize_history(history)
    if not message.strip():
        yield hist, gr.update(interactive=True), gr.update(interactive=True), gr.update(visible=False)
        return
    
    plan = ExecutionPlan()
    
    initial_status = status_html("🚀 Starting...", pulse=True)
    initial_html = fmt_static(render_turn("", "", plan.render(), initial_status))
    
    base_history = hist + [{"role": "user", "content": message}, {"role": "assistant", "content": initial_html}]
    yield base_history, gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=True)
    
    llm = StreamingLLM(model=LLM_MODELS.get(model, model), base_url=LLM_BASE_URL, temperature=temp, max_tokens=max_tokens)
    
    # Build messages
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if synth_context and synth_context.strip():
        messages.append({"role": "system", "content": f"Previous context:\n{synth_context}"})
    if context.strip():
        messages.append({"role": "system", "content": f"Available context:\n{context}"})
    
    for msg in hist[-12:]:
        content = normalize_content(msg.get("content"))
        if content and not content.startswith("🔧 Tool:"):
            messages.append({"role": msg["role"], "content": content})
    
    messages.append({"role": "user", "content": message})
    
    # Prepare tools
    all_tools = resource_server.get_tools()
    tools = [t for t in all_tools if t["function"]["name"] in enabled_tools] if enabled_tools else None
    
    thinking_buf, content_buf = "", ""
    last_emit = 0.0
    MIN_EMIT = 0.10
    
    def emit(status_text: str, pulse: bool = True):
        nonlocal last_emit
        now = time.time()
        if now - last_emit < MIN_EMIT:
            return None
        last_emit = now
        rendered = fmt_static(render_turn(thinking_buf, content_buf, plan.render(), status_html(status_text, pulse)))
        return update_last_assistant(base_history, rendered)
    
    MAX_ITERATIONS = 10
    for iteration in range(MAX_ITERATIONS):
        tool_calls_pending = []
        step = plan.add_step(f"LLM call {iteration+1}")
        plan.start_step(step)
        
        current_tools = tools if iteration < (MAX_ITERATIONS - 1) else None
        for chunk in llm.stream(messages, tools=tools):
            if chunk["type"] == "error":
                plan.fail_step(step, chunk["content"])
                content_buf += f"\n\n❌ Error: {chunk['content']}"
                rendered = fmt_static(render_turn(thinking_buf, content_buf, plan.render(), status_html("❌ Error", False)))
                yield update_last_assistant(base_history, rendered), gr.update(interactive=True), gr.update(interactive=True), gr.update(visible=False)
                return
            
            elif chunk["type"] == "thinking":
                thinking_buf += chunk.get("content") or ""
                result = emit("💭 Thinking…")
                if result:
                    yield result, gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=True)
            
            elif chunk["type"] == "end_thinking":
                last_emit = 0
                result = emit("🔄 Generating…")
                if result:
                    yield result, gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=True)
            
            elif chunk["type"] == "content":
                content_buf += chunk.get("content") or ""
                result = emit("✍️ Responding…")
                if result:
                    yield result, gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=True)
            
            elif chunk["type"] in ["tool_calls", "tool_calls_final"]:
                if chunk.get("calls"):
                    tool_calls_pending = chunk["calls"]
        
        plan.complete_step(step, "Done")
        
        if not tool_calls_pending:
            break
        
        # Handle tool calls
        content_buf += "\n\n" + render_tool_calls(tool_calls_pending) + "\n"
        last_emit = 0
        rendered = fmt_static(render_turn(thinking_buf, content_buf, plan.render(), status_html("🔧 Calling tools…")))
        yield update_last_assistant(base_history, rendered), gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=True)
        
        messages.append({"role": "assistant", "tool_calls": tool_calls_pending, **({"content": content_buf} if content_buf.strip() else {})})
        
        for tc in tool_calls_pending:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
            except:
                args = {}
            
            tool_step = plan.add_step(f"Tool: {name}")
            plan.start_step(tool_step)
            t0 = time.time()
            result = resource_server.execute_tool(name, args)
            plan.complete_step(tool_step, "", int((time.time() - t0) * 1000))
            
            content_buf += "\n\n" + render_tool_result(name, result) + "\n"
            messages.append({"role": "tool", "tool_call_id": tc.get("id") or "", "name": name, "content": result})
            
            last_emit = 0
            rendered = fmt_static(render_turn(thinking_buf, content_buf, plan.render(), status_html(f"🔧 {name} done", False)))
            yield update_last_assistant(base_history, rendered), gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=True)
        
        content_buf += "\n\n---\n\n"
    
    # Final
    rendered = fmt_static(render_turn(thinking_buf, content_buf, plan.render(), status_html("✅ Complete", False)))
    yield update_last_assistant(base_history, rendered), gr.update(interactive=True), gr.update(interactive=True), gr.update(visible=False)

# =============================================================================
# PROCESS HANDLER
# =============================================================================
def process_stream(sources: List[str], ops: List[str], recipe: str, llm_name: str, vlm_name: str,
                   max_chars: int, history: List[Dict], auto_mode: bool) -> Generator[List[Dict], None, None]:
    """Process sources with operations or agent mode."""
    
    history = list(history or [])
    if not sources:
        history.append({"role": "assistant", "content": "⚠️ Please select at least one source file."})
        yield history
        return
    
    plan = ExecutionPlan()
    step = plan.add_step("Load content")
    plan.start_step(step)
    
    load_result = json.loads(resource_server.execute_tool("load_content", {"paths": sources, "max_chars": int(max_chars) if max_chars else 0}))
    files = load_result.get("files", [])
    total_tokens = load_result.get("total_tokens", 0)
    plan.complete_step(step, f"{len(files)} files")
    
    token_class = "warn" if total_tokens > MAX_RECOMMENDED_TOKENS else ""
    status = f"**Processing:** {', '.join(os.path.basename(s) for s in sources)}\n"
    status += f"**Mode:** {'Auto' if auto_mode else 'Manual'} | **Tokens:** <span class='stats-value {token_class}'>~{total_tokens:,}</span>\n\n"
    
    combined = "\n\n---\n\n".join((f.get("content") or "") for f in files)
    history.append({"role": "assistant", "content": status + plan.render() + "\n\n⏳ Starting..."})
    yield history
    
    if auto_mode:
        yield from _process_auto(sources, recipe, combined, llm_name, history, status, plan)
    else:
        yield from _process_manual(sources, ops, recipe, combined, llm_name, vlm_name, history, status, plan)


def _process_manual(sources, ops, recipe, combined, llm_name, vlm_name, history, status, plan):
    output = ""
    
    if "Output Full Content" in ops:
        step = plan.add_step("Full content")
        plan.start_step(step)
        safe_combined = html.escape(combined)
        output += f"<details><summary>📝 Full Content</summary>\n\n{safe_combined }\n\n</details>\n\n"
        plan.complete_step(step)
        history[-1]["content"] = status + plan.render() + "\n\n---\n\n" + output
        yield history
    
    if "Output Code" in ops:
        step = plan.add_step("Extract code")
        plan.start_step(step)
        result = json.loads(resource_server.execute_tool("extract_code", {"paths": sources}))
        blocks = result.get("code_blocks", [])
        if blocks:
            code_text = "\n\n".join(f"```{b['language']}\n{b['code']}\n```" for b in blocks[:20])
            output += f"<details><summary>📄 Code ({len(blocks)})</summary>\n\n{code_text}\n\n</details>\n\n"
        plan.complete_step(step, f"{len(blocks)} blocks")
        history[-1]["content"] = status + plan.render() + "\n\n---\n\n" + output
        yield history
    
    if "Output URLs" in ops:
        step = plan.add_step("Extract URLs")
        plan.start_step(step)
        result = json.loads(resource_server.execute_tool("extract_urls", {"paths": sources}))
        urls = result.get("urls", [])
        if urls:
            output += f"<details><summary>🔗 URLs ({len(urls)})</summary>\n\n" + "\n".join(f"- [{u['url']}]({u['url']})" for u in urls) + "\n\n</details>\n\n"
        plan.complete_step(step, f"{len(urls)} URLs")
        history[-1]["content"] = status + plan.render() + "\n\n---\n\n" + output
        yield history
    
    if "Output Images" in ops:
        step = plan.add_step("Extract images")
        plan.start_step(step)
        result = json.loads(resource_server.execute_tool("extract_images", {"paths": sources}))
        images = result.get("images", [])
        if images:
            output += f"<details><summary>🖼️ Images ({len(images)})</summary>\n\n" + "\n".join(f"- `{i['original']}` {'✓' if i['exists'] else '✗'}" for i in images) + "\n\n</details>\n\n"
        plan.complete_step(step, f"{len(images)} images")
        history[-1]["content"] = status + plan.render() + "\n\n---\n\n" + output
        yield history
    
    if "Generate" in ops and recipe.strip():
        step = plan.add_step("Generate")
        plan.start_step(step)
        output += "## 🤖 Generated\n\n"
        history[-1]["content"] = status + plan.render() + "\n\n---\n\n" + output + "⏳ Generating..."
        yield history
        
        llm = StreamingLLM(model=LLM_MODELS.get(llm_name, llm_name), base_url=LLM_BASE_URL, max_tokens=4096)
        generated = ""
        for chunk in llm.stream([{"role": "user", "content": f"{recipe}\n\n---\n\nCONTENT:\n{combined}"}]):
            if chunk.get("type") == "content":
                generated += chunk["content"]
                safe_generated = html.escape(generated)
                history[-1]["content"] = status + plan.render() + "\n\n---\n\n" + output + format_thinking_tags(safe_generated)
                yield history
        
        safe_final = html.escape(generated)
        output += format_thinking_tags(safe_final) + "\n\n"
        plan.complete_step(step)
        history[-1]["content"] = status + plan.render() + "\n\n---\n\n" + output
        yield history
    
    if "Analyze Images" in ops:
        step = plan.add_step("VLM analysis")
        plan.start_step(step)
        result = json.loads(resource_server.execute_tool("extract_images", {"paths": sources}))
        resolved = [i for i in result.get("images", []) if i.get("resolved")][:5]
        
        if resolved:
            output += "\n\n## 🖼️ Visual Analysis\n\n"
            vlm = StreamingLLM(model=VLM_MODELS.get(vlm_name, vlm_name), base_url=LLM_BASE_URL, max_tokens=1024)
            
            for img in resolved:
                output += f"### {os.path.basename(img['resolved'])}\n\n"
                history[-1]["content"] = status + plan.render() + "\n\n---\n\n" + output + "⏳ Analyzing..."
                yield history
                
                analysis = ""
                for chunk in vlm.analyze_image(img["resolved"], "Describe this figure in technical detail."):
                    analysis += chunk
                    history[-1]["content"] = status + plan.render() + "\n\n---\n\n" + output + analysis
                    yield history
                output += analysis + "\n\n"
        
        plan.complete_step(step, f"{len(resolved)} analyzed")
    
    history[-1]["content"] = fmt_static(status + plan.render() + "\n\n---\n\n" + output + "\n\n✅ **Complete**")
    yield history


def _process_auto(sources, recipe, combined, llm_name, history, status, plan):
    step = plan.add_step("Agent processing")
    plan.start_step(step)
    
    llm = StreamingLLM(model=LLM_MODELS.get(llm_name, llm_name), base_url=LLM_BASE_URL, max_tokens=4096)
    
    system = """You are an AI assistant with tools for content analysis:
- list_files, get_file_info, load_content, search_content
- extract_code, extract_urls, extract_images, get_learning_objectives

Use tools as needed. Explain reasoning before acting."""
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Task: {recipe}\n\nSources: {sources}\n\nPreview:\n{combined[:2000]}..."}
    ]
    
    output = "## 🤖 Agent Processing\n\n"
    
    for iteration in range(5):
        tool_calls_pending = []
        response_text = ""
        
        iter_step = plan.add_step(f"Iteration {iteration+1}")
        plan.start_step(iter_step)
        history[-1]["content"] = status + plan.render() + "\n\n---\n\n" + output + f"⏳ Iteration {iteration+1}..."
        yield history
        
        for chunk in llm.stream(messages, tools=resource_server.get_tools()):
            if chunk.get("type") == "content":
                response_text += chunk["content"]
                history[-1]["content"] = status + plan.render() + "\n\n---\n\n" + output + format_thinking_tags(response_text)
                yield history
            elif chunk.get("type") == "thinking":
                response_text += chunk["content"]
            elif chunk.get("type") == "tool_calls_final":
                tool_calls_pending = chunk.get("calls", [])
        
        output += response_text + "\n\n"
        
        if not tool_calls_pending:
            plan.complete_step(iter_step, "Done")
            break
        
        plan.complete_step(iter_step, f"{len(tool_calls_pending)} tools")
        messages.append({"role": "assistant", "content": response_text, "tool_calls": tool_calls_pending})
        
        for tc in tool_calls_pending:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
            except:
                args = {}
            
            tool_step = plan.add_step(f"Tool: {name}")
            plan.start_step(tool_step)
            result = resource_server.execute_tool(name, args)
            plan.complete_step(tool_step)
            
            preview = result[:500] + "..." if len(result) > 500 else result
            output += f"<details class='tool-call'><summary>🔧 {name}</summary>\n\n```json\n{preview}\n```\n\n</details>\n\n"
            messages.append({"role": "tool", "tool_call_id": tc.get("id") or "", "name": name, "content": result})
            
            history[-1]["content"] = status + plan.render() + "\n\n---\n\n" + output
            yield history
    
    plan.complete_step(step, "Done")
    history[-1]["content"] = fmt_static(status + plan.render() + "\n\n---\n\n" + output + "\n\n✅ **Complete**")
    yield history

# =============================================================================
# SAVE HANDLER
# =============================================================================
def do_save(history: List[Dict], filename: str) -> str:
    if not history:
        return "Nothing to save"
    
    content = next(((m.get("content") or "") for m in reversed(history) if m.get("role") == "assistant"), "")
    print(content)
    if not content or content.startswith(("Ready", "Select", "⚠️")):
        return "Nothing to save"
    
    content = content.replace("✅ **Complete**", "").replace("⏳ Generating...", "").strip()
    fn = (filename.strip() or f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}") + (".md" if not filename.strip().endswith('.md') else "")
    
    err = safe_write(os.path.join(SYNTH_DIR, fn), content)
    if err:
        return f"❌ {err}"
    
    resource_server.refresh_index()
    return f"✅ Saved: {fn}"

# =============================================================================
# CSS
# =============================================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400&display=swap');
* { font-family: 'Inter', system-ui, sans-serif !important; }
code, pre, .mono { font-family: 'JetBrains Mono', monospace !important; }

.stats { font-family: 'JetBrains Mono', monospace !important; font-size: 12px; padding: 6px 10px; background: var(--background-fill-secondary); border-left: 3px solid #76b900; border-radius: 4px; }
.stats-value { font-weight: bold; }
.warn { color: #ff6b6b !important; }

details { border: 1px solid var(--border-color-primary); padding: 8px 12px; border-radius: 6px; margin: 8px 0; background: var(--background-fill-secondary); }
details summary { cursor: pointer; font-weight: 500; padding: 4px 0; }
details[open] summary { margin-bottom: 8px; border-bottom: 1px solid var(--border-color-primary); padding-bottom: 8px; }

.think-box { border-color: #6366f1; background: var(--background-fill-primary); }
.think-box summary { color: #6366f1; font-style: italic; }

.tool-call { border-color: #76b900; background: rgba(118, 185, 0, 0.1); }
.tool-call summary { color: #5a9400; font-weight: 600; }
.tool-call summary code { background: #76b900; color: white; padding: 2px 6px; border-radius: 3px; }

.thinking { font-weight: 600; animation: pulse 1.2s infinite; }
@keyframes pulse { 0%, 100% { opacity: 0.4; } 50% { opacity: 1; } }

pre { background: var(--background-fill-secondary) !important; border-radius: 6px; padding: 12px; overflow-x: auto; }
.status-line { margin-top: 10px; padding-top: 8px; border-top: 1px solid var(--border-color-primary); font-family: 'JetBrains Mono', monospace !important; font-size: 12px; opacity: 0.8; text-align: right; }
"""

# =============================================================================
# INTERFACE
# =============================================================================
def build_interface() -> gr.Blocks:
    sources = resource_server.list_all_sources()
    synth_files = resource_server.list_synth_files()
    tool_names = resource_server.get_tool_names()
    
    with gr.Blocks(title="DLI Content Processor") as app:
        gr.Markdown("# 🚀 DLI Content Processor")
        
        raw_context = gr.State("")
        synth_context = gr.State("")
        
        with gr.Tabs():
            # === CHAT TAB ===
            with gr.Tab("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Settings")
                        ctx_preset = gr.Dropdown(list(SYSTEM_PRESETS.keys()), value="Teaching Assistant", label="Persona")
                        system_prompt = gr.Textbox(SYSTEM_PRESETS["Teaching Assistant"], label="System", lines=2)
                        chat_model = gr.Dropdown(list(LLM_MODELS.keys()), value=DEFAULT_LLM, label="Model", allow_custom_value=True)
                        with gr.Row():
                            chat_temp = gr.Slider(0, 1, 0.3, step=0.05, label="Temp")
                            max_tokens = gr.Slider(256, 8192, 4000, step=256, label="Max Tokens")
                        
                        gr.Markdown("#### 🔧 Tools")
                        tool_boxes = gr.CheckboxGroup(tool_names, value=tool_names, label="Enabled")
                        
                        gr.Markdown("#### Context")
                        with gr.Row():
                            flip_ctx = gr.Button("🔃 Flip", size="sm")
                            refresh_btn = gr.Button("🔄 Refresh", size="sm")
                        ctx_files = gr.Dropdown(sources, multiselect=True, label="Files")
                        ctx_max = gr.Slider(0, 50000, 15000, step=1000, label="Max Chars")
                        with gr.Row():
                            load_ctx = gr.Button("📁 Load", size="sm")
                            clear_ctx = gr.Button("🗑️ Clear", size="sm")
                        ctx_box = gr.Textbox(label="Loaded", lines=3, interactive=True)
                        ctx_stats = gr.Markdown("", elem_classes=["stats"])
                        
                        gr.Markdown("#### Synth History")
                        synth_select = gr.Dropdown(synth_files, multiselect=True, label="Previous Outputs")
                        with gr.Row():
                            load_synth = gr.Button("📄 Load", size="sm")
                            clear_synth = gr.Button("🗑️ Clear", size="sm")
                        synth_box = gr.Textbox(label="Synth", lines=2, interactive=True)
                    
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot([{"role": "assistant", "content": "Hello! I can use tools to explore content. Try asking me to search, extract code, or analyze files."}], height="70vh", elem_id="chatbot")
                        chat_busy = gr.Markdown("⏳ <span class='thinking'>Processing…</span>", visible=False)
                        with gr.Row():
                            chat_input = gr.Textbox(placeholder="Ask anything...", show_label=False, scale=5)
                            chat_send = gr.Button("Send", variant="primary", scale=1)
                        with gr.Row():
                            stop_btn = gr.Button("Stop", variant="stop", size="sm")
                            clear_chat = gr.Button("Clear", size="sm")
            
            # === PROCESS TAB ===
            with gr.Tab("⚡ Process"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Sources")
                        with gr.Row():
                            flip_src = gr.Button("🔃 Flip", size="sm")
                            refresh_btn2 = gr.Button("🔄 Refresh", size="sm")
                        source_select = gr.Dropdown(sources, multiselect=True, label="Files")
                        max_chars = gr.Slider(0, 50000, 15000, step=1000, label="Max Chars")
                        
                        gr.Markdown("#### Recipe")
                        preset = gr.Dropdown(list(RECIPE_PRESETS.keys()), value="Summary", label="Preset")
                        recipe = gr.Textbox(RECIPE_PRESETS["Summary"], label="Instructions", lines=4)
                        
                        gr.Markdown("#### Mode")
                        auto_mode = gr.Checkbox(label="Auto (AI selects tools)", value=False)
                        ops = gr.CheckboxGroup(["Generate", "Output URLs", "Output Images", "Output Code", "Analyze Images", "Output Full Content"], value=["Generate"], label="Manual Ops")
                        
                        with gr.Row():
                            proc_llm = gr.Dropdown(list(LLM_MODELS.keys()), value=DEFAULT_LLM, label="LLM", allow_custom_value=True)
                            proc_vlm = gr.Dropdown(list(VLM_MODELS.keys()), value=DEFAULT_VLM, label="VLM", allow_custom_value=True)
                        
                        token_est = gr.Markdown("", elem_classes=["stats"])
                        process_btn = gr.Button("🚀 Process", variant="primary", size="lg")
                        busy = gr.Markdown("⏳ <span class='thinking'>Processing…</span>", visible=False)
                        
                        gr.Markdown("#### 💾 Save")
                        with gr.Row():
                            save_name = gr.Textbox(placeholder="output.md", show_label=False, scale=3)
                            save_btn = gr.Button("Save", scale=1)
                        save_status = gr.Markdown("")
                    
                    with gr.Column(scale=2):
                        proc_bot = gr.Chatbot([{"role": "assistant", "content": "Select sources and click Process."}], height="70vh")
                        proc_clear = gr.Button("Clear", size="sm")
            
            # === TOOLS TAB ===
            with gr.Tab("🔧 Tools"):
                gr.Markdown("### Tool Discovery")
                with gr.Row():
                    tools_json = gr.JSON(resource_server.get_tools(), label="Schema")
                    with gr.Column():
                        gr.Markdown("### Test")
                        tool_name = gr.Dropdown(tool_names, label="Tool")
                        tool_args = gr.Textbox(label="Args (JSON)", placeholder='{"query": "GPU"}', lines=2)
                        run_tool = gr.Button("Execute")
                        tool_result = gr.JSON(label="Result")
        
        # === EVENTS ===
        ctx_preset.change(lambda p: SYSTEM_PRESETS.get(p, ""), [ctx_preset], [system_prompt])
        preset.change(lambda p: RECIPE_PRESETS.get(p, ""), [preset], [recipe])
        
        def load_context(paths, mc):
            if not paths:
                return "", "", ""
            result = json.loads(resource_server.execute_tool("load_content", {"paths": paths, "max_chars": int(mc) if mc else 0}))
            files, tokens = result.get("files", []), result.get("total_tokens", 0)
            content = "\n\n---\n\n".join((f.get("content") or "") for f in files)
            display = f"<details open><summary>{len(files)} files, {tokens:,} tokens</summary>\n\n{content}\n\n</details>"
            stats = f"<span class='{'warn' if tokens > MAX_RECOMMENDED_TOKENS else ''} stats-value'>~{tokens:,} tokens</span>"
            return display, stats, content
        
        load_ctx.click(load_context, [ctx_files, ctx_max], [ctx_box, ctx_stats, raw_context])
        clear_ctx.click(lambda: ("", "", ""), outputs=[ctx_box, ctx_stats, raw_context])
        
        def load_synth_ctx(paths):
            if not paths:
                return "", ""
            return f"Loaded {len(paths)} docs", resource_server.get_raw_content(paths)
        
        load_synth.click(load_synth_ctx, [synth_select], [synth_box, synth_context])
        clear_synth.click(lambda: ("", ""), outputs=[synth_box, synth_context])
        
        def estimate(paths, recipe_text, mc):
            if not paths:
                return "Select files"
            result = json.loads(resource_server.execute_tool("load_content", {"paths": paths, "max_chars": int(mc) if mc else 0}))
            tokens = result.get("total_tokens", 0) + estimate_tokens(recipe_text or "")
            return f"Tokens: <span class='{'warn' if tokens > MAX_RECOMMENDED_TOKENS else ''} stats-value'>~{tokens:,}</span>"
        
        for comp in [source_select, recipe, max_chars]:
            comp.change(estimate, [source_select, recipe, max_chars], [token_est])
        
        def refresh():
            resource_server.refresh_index()
            new_src, new_synth = resource_server.list_all_sources(), resource_server.list_synth_files()
            return gr.update(choices=new_src), gr.update(choices=new_src), gr.update(choices=new_synth)
        
        refresh_btn.click(refresh, outputs=[ctx_files, source_select, synth_select])
        refresh_btn2.click(refresh, outputs=[ctx_files, source_select, synth_select])
        
        def flip(selected):
            all_paths = [s[1] for s in resource_server.list_all_sources()]
            return [p for p in all_paths if p not in selected]
        
        flip_ctx.click(flip, [ctx_files], [ctx_files])
        flip_src.click(flip, [source_select], [source_select])
        
        # Chat events
        chat_evt = chat_send.click(chat_stream, [chat_input, chatbot, raw_context, synth_context, system_prompt, chat_temp, max_tokens, chat_model, tool_boxes], [chatbot, chat_input, chat_send, chat_busy])
        chat_evt.then(lambda: "", outputs=[chat_input])
        
        submit_evt = chat_input.submit(chat_stream, [chat_input, chatbot, raw_context, synth_context, system_prompt, chat_temp, max_tokens, chat_model, tool_boxes], [chatbot, chat_input, chat_send, chat_busy])
        submit_evt.then(lambda: "", outputs=[chat_input])
        
        def stop_chat(history):
            h = list(history or [])
            if h and h[-1].get("role") == "assistant" and "status-line" not in (h[-1].get("content") or ""):
                h[-1]["content"] += f"\n\n<div class='status-line'>{status_html('⛔ Cancelled', False)}</div>"
            return h, gr.update(visible=False), gr.update(interactive=True), gr.update(interactive=True)
        
        stop_btn.click(stop_chat, [chatbot], [chatbot, chat_busy, chat_input, chat_send], cancels=[chat_evt, submit_evt])
        clear_chat.click(lambda: ([{"role": "assistant", "content": "Cleared. Ready to help."}], gr.update(interactive=True), gr.update(interactive=True), gr.update(visible=False)), outputs=[chatbot, chat_input, chat_send, chat_busy])
        
        # Process events
        process_btn.click(lambda: gr.update(visible=True), outputs=[busy]).then(
            process_stream, [source_select, ops, recipe, proc_llm, proc_vlm, max_chars, proc_bot, auto_mode], [proc_bot]
        ).then(lambda: gr.update(visible=False), outputs=[busy])
        
        proc_clear.click(lambda: [{"role": "assistant", "content": "Ready."}], outputs=[proc_bot])
        save_btn.click(do_save, [proc_bot, save_name], [save_status])
        
        # Tool test
        def exec_tool(name, args_str):
            try:
                args = json.loads(args_str) if args_str.strip() else {}
            except json.JSONDecodeError as e:
                return {"error": f"Invalid JSON: {e}"}
            return json.loads(resource_server.execute_tool(name, args))
        
        run_tool.click(exec_tool, [tool_name, tool_args], [tool_result])
    
    return app

# =============================================================================
# SERVER
# =============================================================================
def create_app() -> FastAPI:
    app = FastAPI(title="DLI Content Processor")
    for name, path in [("imgs", IMG_DIR), ("slides", SLIDES_DIR)]:
        if os.path.exists(path):
            app.mount(f"/static/{name}", StaticFiles(directory=path), name=f"static-{name}")
    return gr.mount_gradio_app(app, build_interface().queue(), path="/", root_path=ROOT_PATH, css=CSS, theme=gr.themes.Soft(primary_hue="green"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--standalone", action="store_true")
    args = parser.parse_args()
    
    logger.info(f"Starting on port {args.port}")
    if args.standalone:
        build_interface().launch(server_name=args.host, server_port=args.port)
    else:
        uvicorn.run(create_app(), host=args.host, port=args.port)
