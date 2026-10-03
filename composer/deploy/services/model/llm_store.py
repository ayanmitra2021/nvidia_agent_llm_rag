"""Bounded, private SQLite response cache and credential-free request journal."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from llm_request import numeric_json


def canonical(body):
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class Store:
    def __init__(self, path, clean, ttl=86400, max_bytes=67108864, max_entries=2000, namespace="c-fx-25-v1:litellm-1.100.1"):
        if ttl < 0 or max_bytes < 1 or max_entries < 1:
            raise ValueError("invalid cache limits")
        self.namespace = namespace
        self.clean = clean
        self.ttl = ttl
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.lock = threading.Lock()
        self._activity_rows = []
        self._activity_truncated = False
        target = Path(path)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(target.parent, 0o700)
        self.db = sqlite3.connect(target, check_same_thread=False)
        os.chmod(target, 0o600)
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL,
            expires REAL NOT NULL, used REAL NOT NULL, size INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY, time REAL NOT NULL,
            method TEXT NOT NULL, url TEXT NOT NULL, body TEXT NOT NULL, source TEXT NOT NULL,
            status INTEGER);
        ''')
        self.db.commit()

    def key(self, url, body):
        return hashlib.sha256(canonical(["dli-llm-cache-v2-numeric", self.namespace, url, numeric_json(body)]).encode()).hexdigest()

    def legacy_key(self, url, body):
        return hashlib.sha256(canonical(["dli-llm-cache-v1", self.namespace, url, body]).encode()).hexdigest()

    def get(self, key, legacy_key=None):
        with self.lock:
            now = time.time()
            self.db.execute("DELETE FROM cache WHERE expires <= ?", (now,))
            row = self.db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
            if row is None and legacy_key is not None and legacy_key != key:
                # Promote only the caller's exact old key. Keep the original
                # response, expiry and size; never scan or guess representations.
                self.db.execute("UPDATE OR IGNORE cache SET key = ? WHERE key = ?", (key, legacy_key))
                row = self.db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
            if row:
                self.db.execute("UPDATE cache SET used = ? WHERE key = ?", (now, key))
            self.db.commit()
            return json.loads(row[0]) if row else None

    def put(self, key, value):
        encoded = canonical(self.clean(value))
        size = len(encoded.encode())
        if self.ttl <= 0 or size > min(self.max_bytes, 4194304):
            return
        with self.lock:
            now = time.time()
            self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?, ?)",
                (key, encoded, now + self.ttl, now, size))
            while True:
                count, total = self.db.execute("SELECT COUNT(*), COALESCE(SUM(size), 0) FROM cache").fetchone()
                if count <= self.max_entries and total <= self.max_bytes:
                    break
                self.db.execute("DELETE FROM cache WHERE key = (SELECT key FROM cache ORDER BY used LIMIT 1)")
            self.db.commit()

    def record(self, method, url, body, source="upstream", status=None):
        with self.lock:
            cursor = self.db.execute("INSERT INTO requests(time,method,url,body,source,status) VALUES(?,?,?,?,?,?)",
                (time.time(), method, self.clean(url), canonical(self.clean(body)), source, status))
            self.db.commit()
            return cursor.lastrowid

    def status(self, row, status):
        with self.lock:
            self.db.execute("UPDATE requests SET status = ? WHERE id = ?", (status, row))
            self.db.commit()

    def activity(self, window=3600, limit=100, model="", fallback_url=None, model_labels=None):
        """Project bounded journal metadata; never return request bodies or provider URLs.

        One row is a cache delivery, shared delivery, or provider HTTP attempt.
        Retries may add rows and HTTP 200 does not prove a stream completed.
        """
        now = time.time()
        with self.lock:
            after = self._activity_rows[0][0] if self._activity_rows else 0
            new_rows = self.db.execute("""SELECT id, time, method, url,
                CASE WHEN json_valid(body) THEN json_extract(body, '$.model') END,
                source, status FROM requests WHERE id > ? ORDER BY id DESC LIMIT 20001""", (after,)).fetchall()
            combined = new_rows + self._activity_rows
            self._activity_truncated = self._activity_truncated or len(combined) > 20000
            self._activity_rows = combined[:20000]
            # Pending HTTP observations can become known after the row was inserted.
            pending_ids = [row[0] for row in self._activity_rows if row[6] is None]
            updates = {}
            for offset in range(0, len(pending_ids), 500):
                ids = pending_ids[offset:offset + 500]
                updates.update(self.db.execute('SELECT id, status FROM requests WHERE id IN (' + ','.join('?' for _ in ids) + ')', ids).fetchall())
            if updates:
                self._activity_rows = [(*row[:6], updates.get(row[0], row[6])) for row in self._activity_rows]
            rows = list(self._activity_rows)
            truncated = self._activity_truncated
        counts = dict(course_cache=0, local_cache=0, shared=0, primary=0, fallback=0)
        width = window / 60
        start = now - window
        buckets = [{"time": start + i * width, "cache": 0, "primary": 0, "fallback": 0, "errors": 0} for i in range(60)]
        events, models, failures, pending = [], set(), 0, 0
        for row_id, when, method, url, model_id, source, status in rows:
            if method != "POST" or not url.endswith(("/chat/completions", "/embeddings", "/ranking", "/reranking")):
                continue
            model_id = self.clean((model_labels or {}).get(model_id, "Other / unavailable model")) if isinstance(model_id, str) else "Other / unavailable model"
            model_id = "".join(c for c in model_id if c.isprintable())[:256]
            models.add(model_id)
            if when < start or when > now or (model and model_id != model):
                continue
            category = {"course_bundle": "course_cache", "cache": "local_cache", "shared": "shared"}.get(source)
            if category is None:
                if source != "upstream":
                    continue
                category = "fallback" if fallback_url and url.startswith(fallback_url.rstrip('/') + '/') else "primary"
            counts[category] += 1
            failed = status is not None and status >= 400
            failures += int(failed)
            pending += int(status is None)
            bucket = buckets[min(59, int((when - start) / width))]
            bucket[category if category in {"primary", "fallback"} else "cache"] += 1
            bucket["errors"] += int(failed)
            events.append({"id": row_id, "time": when, "model": model_id,
                "operation": "embeddings" if url.endswith('/embeddings') else "ranking" if url.endswith(('/ranking', '/reranking')) else "chat",
                "source": category, "status": status})
        total = sum(counts.values())
        cached = counts['course_cache'] + counts['local_cache']
        oldest = min((r[1] for r in rows), default=None)
        return {"schema_version": 1, "as_of": now, "window_seconds": window,
            "bucket_seconds": width, "model_filter": model, "models": sorted(models),
            "counts": {**counts, "total": total, "cache_hits": cached,
                "provider_attempts": counts['primary'] + counts['fallback'],
                "http_errors": failures, "unconfirmed": pending},
            "operations_per_minute": total / (window / 60),
            "buckets": buckets, "recent": events[:limit], "matching_records": len(events),
            "journal": {"scanned_records": len(rows), "scan_limit": 20000,
                "truncated": truncated, "oldest_scanned": oldest,
                "window_complete": not truncated or (oldest is not None and oldest <= start)},
            "semantics": {"unit": "cache matches and provider HTTP attempts",
                "endpoint_classification": "Fallback endpoint uses the current configured address; it does not prove fallback routing. Other endpoints includes historical configurations.",
                "model_labels": "Known public catalog names and configured public aliases; unmapped historical identifiers are hidden.",
                "retry_counting": "A retry is another provider attempt; these are not unique learner requests.",
                "status": "HTTP status is captured at response headers. HTTP 200 does not prove a stream completed; missing status is unconfirmed, not necessarily active.",
                "privacy": "Prompts, responses, credentials and provider URLs are excluded.",
                "timing": "Per-request duration, TTFT and trace IDs are not stored in this journal. Use Jaeger for instrumented trace details."}}

    async def on_request(self, request):
        raw = await request.aread()
        body = json.loads(raw) if raw else None
        row = self.record(request.method, str(request.url), body)
        request.extensions["dli_record"] = row

    async def on_response(self, response):
        row = response.request.extensions.get("dli_record")
        if row is not None:
            self.status(row, response.status_code)
