"""Export/import a versioned response bundle. Never exports the request journal."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import tempfile

FORMAT = "dli-llm-cache-v1"


def export_bundle(database, output):
    connection = sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True)
    rows = connection.execute("SELECT key, value, expires FROM cache WHERE expires > ? ORDER BY key", (time.time(),)).fetchall()
    connection.close()
    entries = [{"key": key, "value": json.loads(value), "expires": expires} for key, value, expires in rows]
    data = json.dumps({"format": FORMAT, "entries": entries}, sort_keys=True, separators=(",", ":")).encode()
    artifact = gzip.compress(data, mtime=0)
    path = Path(output)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = stream.name
        stream.write(artifact)
    os.replace(temporary, path)
    manifest = {"format": FORMAT, "entries": len(entries), "compressed_bytes": len(artifact),
        "sha256": hashlib.sha256(artifact).hexdigest(), "request_journal_included": False,
        "suggested_storage": "git" if len(artifact) <= 5 * 1024 * 1024 else "object-storage"}
    Path(str(path) + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def import_bundle(database, bundle, digest):
    path = Path(bundle)
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("bundle exceeds compressed size limit")
    artifact = path.read_bytes()
    if hashlib.sha256(artifact).hexdigest() != digest:
        raise ValueError("bundle checksum mismatch")
    with gzip.open(path, "rb") as stream:
        raw = stream.read(64 * 1024 * 1024 + 1)
    if len(raw) > 64 * 1024 * 1024:
        raise ValueError("bundle exceeds expanded size limit")
    data = json.loads(raw)
    if data.get("format") != FORMAT or not isinstance(data.get("entries"), list):
        raise ValueError("unsupported bundle format")
    from llm_store import Store
    store = Store(database, lambda value: value)
    count = 0
    for entry in data["entries"]:
        key, value, expires = entry["key"], entry["value"], entry["expires"]
        if not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("invalid cache key")
        if not isinstance(value, (dict, list)) or type(expires) not in (int, float):
            raise ValueError("invalid cache entry")
        if expires > time.time():
            store.ttl = min(expires - time.time(), 86400)
            store.put(key, value)
            count += 1
    store.db.close()
    return {"format": FORMAT, "imported": count}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["export", "import"])
    parser.add_argument("bundle")
    parser.add_argument("--database", default="/var/lib/llm/requests.sqlite")
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if args.action == "export":
        result = export_bundle(args.database, args.bundle)
    else:
        if not args.sha256:
            parser.error("import requires --sha256 from a trusted manifest")
        result = import_bundle(args.database, args.bundle, args.sha256)
    print(json.dumps(result))
