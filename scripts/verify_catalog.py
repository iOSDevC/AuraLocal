#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""Check every models.json entry against live Hugging Face data.

    uv run scripts/verify_catalog.py                 # report only; exit 1 on any difference
    uv run scripts/verify_catalog.py --fix           # also write the fixable corrections to both catalogs
    uv run scripts/verify_catalog.py --only gemma4_12b,granite4_tiny

Standard library only, so `python3 scripts/verify_catalog.py` works too. Architecture facts (layers, KV heads,
head width, per-layer-type KV, experts, trained context, weight bytes) come from `aura models check <repo> --json`,
the same code that writes `aura models check --entry` (GGUF entries read their own file's header via --gguf);
build it first with `swift build --product aura`.

Per entry it checks: the repo exists (301/307 = renamed, `gated` = needs a token, 401/404 = gone or private);
the GGUF file name exactly as listed (case, subfolders, split parts); the size; numLayers / kvHeads / headDim /
kvLayers / experts / maxContextLength. Requests are capped at --max-requests per 300 s (HF allows 500) and a 429
backs off using Retry-After or the RateLimit header. HF_TOKEN, when set, is sent to huggingface.co only.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOGS = [ROOT / "models.json", ROOT / "Sources/AuraCore/Resources/models.json"]
EXCEPTIONS = ROOT / "scripts/catalog-exceptions.json"
HF = "https://huggingface.co"
WINDOW_SECONDS = 300
# CatalogEntry.jsonText order; `domain` sits after `id` in the hand-written entries.
KEY_ORDER = ["id", "domain", "repoID", "displayName", "category", "docTags", "format", "approximateSizeMB",
             "isUncensored", "ggufFilename", "defaultDocumentPrompt", "numLayers", "kvHeads", "headDim",
             "maxContextLength", "kvLayers", "experts"]
CONTEXT_CEILING = 32_768  # CatalogEntry.contextCeiling
SHARD = re.compile(r"-\d{5}-of-\d{5}\.gguf$")
NESTED_ORDER = ["fullAttentionLayers", "slidingLayers", "slidingWindow", "recurrentLayers", "recurrentStateBytes",
                "fullAttentionKVHeads", "fullAttentionHeadDim", "slidingPrefillStep", "routedExperts", "expertsPerToken",
                "activeSizeMB"]


# MARK: - HTTP


class RateLimiter:
    """Sliding window over every request this run makes, aura's included (charged at an upper bound)."""

    def __init__(self, limit: int, window: float = WINDOW_SECONDS):
        self.limit, self.window, self.sent = limit, window, collections.deque()
        self.total = 0

    def acquire(self, cost: int = 1) -> None:
        cost = min(cost, self.limit)
        while True:
            now = time.monotonic()
            while self.sent and now - self.sent[0] >= self.window:
                self.sent.popleft()
            if len(self.sent) + cost <= self.limit:
                break
            wait = self.window - (now - self.sent[0]) + 0.5
            log(f"  rate limit: {len(self.sent)} requests in the last {self.window:.0f} s, waiting {wait:.0f} s")
            time.sleep(wait)
        self.sent.extend([time.monotonic()] * cost)
        self.total += cost

    def penalize(self, seconds: float) -> None:
        log(f"  HTTP 429: backing off {seconds:.0f} s")
        time.sleep(seconds)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


OPENER = urllib.request.build_opener(NoRedirect)


@dataclass
class Response:
    status: int
    headers: dict
    body: bytes


def retry_after(headers: dict, attempt: int) -> float:
    if (value := headers.get("retry-after", "")).isdigit():
        return float(value) + 1
    # RateLimit: "api";r=0;t=55 — t is seconds until the window resets.
    if match := re.search(r"\bt=(\d+)", headers.get("ratelimit", "")):
        return float(match.group(1)) + 1
    return min(15 * 2 ** attempt, 300)


def get(url: str, limiter: RateLimiter, attempts: int = 6) -> Response:
    headers = {"User-Agent": "AuraLocal-verify-catalog", "Accept": "application/json"}
    if (token := os.environ.get("HF_TOKEN")) and url.startswith(HF + "/"):
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(attempts):
        limiter.acquire()
        try:
            with OPENER.open(urllib.request.Request(url, headers=headers), timeout=60) as reply:
                return Response(reply.status, {k.lower(): v for k, v in reply.headers.items()}, reply.read())
        except urllib.error.HTTPError as error:
            lowered = {k.lower(): v for k, v in error.headers.items()}
            if error.code == 429 and attempt + 1 < attempts:
                limiter.penalize(retry_after(lowered, attempt))
                continue
            return Response(error.code, lowered, error.read() or b"")
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt + 1 == attempts:
                return Response(0, {}, str(error).encode())
            time.sleep(5 * (attempt + 1))
    return Response(0, {}, b"retries exhausted")


# MARK: - Live data


@dataclass
class RepoState:
    status: str  # ok | renamed | gone | error
    canonical: str
    gated: str | None = None
    files: dict = field(default_factory=dict)  # path -> bytes
    detail: str = ""


def fetch_repo(repo: str, limiter: RateLimiter) -> RepoState:
    reply = get(f"{HF}/api/models/{repo}?blobs=true", limiter)
    canonical, status = repo, "ok"
    if reply.status in (301, 302, 307, 308):
        target = urllib.parse.urlparse(reply.headers.get("location", "")).path
        canonical = target.removeprefix("/api/models/").split("?")[0] or repo
        status = "renamed"
        reply = get(f"{HF}/api/models/{canonical}?blobs=true", limiter)
    if reply.status in (401, 403, 404):
        return RepoState("gone", repo, detail=f"HTTP {reply.status}: not found, or private (anonymous HF cannot tell)")
    if reply.status != 200:
        return RepoState("error", repo, detail=f"HTTP {reply.status}: {reply.body[:200].decode(errors='replace')}")
    info = json.loads(reply.body)
    canonical = info.get("id", canonical)
    files = {s["rfilename"]: (s.get("lfs") or {}).get("size", s.get("size", 0)) for s in info.get("siblings", [])}
    gated = info.get("gated") or None
    return RepoState(status, canonical, gated=gated if isinstance(gated, str) else None, files=files)


def aura_cost(files: dict) -> int:
    """Upper bound on aura's requests: listing, config, index, a GGUF header, the first shard's header again, and two
    reads (probe, then the rest) per safetensors header."""
    safetensors = sum(1 for path in files if path.endswith(".safetensors") and "/" not in path)
    return 5 + 2 * min(safetensors, 68)


def listed_gguf(entry: dict, files: dict) -> str | None:
    """The catalog's GGUF as the repo lists it (exact, else same name in another case), for aura's --gguf."""
    name = entry.get("ggufFilename") or ""
    if name in files:
        return name
    return next((p for p in files if p.lower() == name.lower()), None)


def aura_report(aura: str, repo: str, device: str, cost: int, limiter: RateLimiter,
                gguf: str | None = None) -> dict | None:
    command = [aura, "models", "check", repo, "--device", device, "--json"] + (["--gguf", gguf] if gguf else [])
    for attempt in range(4):
        limiter.acquire(cost)
        run = subprocess.run(command, capture_output=True, text=True, timeout=600)
        if run.returncode != 0:
            log(f"  aura failed ({run.returncode}): {run.stderr.strip()[-300:]}")
            return None
        report = json.loads(run.stdout)
        if not any("HTTP 429" in f.get("detail", "") for f in report.get("findings", [])):
            return report
        limiter.penalize(min(60 * 2 ** attempt, 300))
    return report


# MARK: - Comparison


@dataclass
class Diff:
    field: str
    catalog: object
    live: object
    fixable: bool
    note: str = ""


def mib(size: int) -> int:
    return round(size / 1_048_576)  # catalog "MB" are MiB: HardwareAnalyzer divides by 1024 for GiB


def ordered(nested: dict | None) -> dict | None:
    """aura's JSON sorts keys; the catalog keeps CatalogEntry's field order."""
    if nested is None:
        return None
    return {k: nested[k] for k in sorted(nested, key=lambda k: NESTED_ORDER.index(k) if k in NESTED_ORDER else 99)}


def uniform(layout: dict | None, layers: int | None) -> bool:
    if not layout:
        return True
    extra = {k: v for k, v in layout.items() if k not in ("fullAttentionLayers",) and v}
    return not extra and layout.get("fullAttentionLayers") == layers


def compare(entry: dict, state: RepoState, report: dict | None, tolerance: float, architecture: bool) -> list[Diff]:
    diffs: list[Diff] = []
    if state.status == "renamed" or state.canonical != entry["repoID"]:
        diffs.append(Diff("repoID", entry["repoID"], state.canonical, True, "renamed on HF (old name redirects)"))

    if entry["format"] == "gguf":
        size = gguf_file_diffs(entry, state, diffs)
    else:
        size = mlx_file_diffs(entry, state, report, diffs)
    if size:
        live = mib(size)
        if abs(entry["approximateSizeMB"] - live) > tolerance * live:
            diffs.append(Diff("approximateSizeMB", entry["approximateSizeMB"], live, True))

    if not architecture:
        return diffs
    arch = (report or {}).get("architecture") or {}
    layers = arch.get("layers")
    for key, live in (("numLayers", layers), ("kvHeads", arch.get("kvHeads")), ("headDim", arch.get("headDim"))):
        if live is None:
            diffs.append(Diff(key, entry.get(key), None, False, "not readable from config.json / GGUF header"))
        elif entry.get(key) != live:
            diffs.append(Diff(key, entry.get(key), live, True))

    if report is not None:
        layout = None if uniform(arch.get("kvLayers"), layers) else ordered(arch.get("kvLayers"))
        if entry.get("kvLayers") != layout:
            diffs.append(Diff("kvLayers", entry.get("kvLayers"), layout, True))
        if entry.get("experts") != ordered(arch.get("experts")):
            diffs.append(Diff("experts", entry.get("experts"), ordered(arch.get("experts")), True))

    trained, cap = arch.get("trainedContext"), entry.get("maxContextLength")
    if trained and cap and cap > trained:
        diffs.append(Diff("maxContextLength", cap, trained, True, "above the trained context"))
    elif trained and not cap and trained < CONTEXT_CEILING:
        diffs.append(Diff("maxContextLength", None, trained, True, "trained context below 32768 needs a cap"))
    return diffs


def gguf_file_diffs(entry: dict, state: RepoState, diffs: list[Diff]) -> int | None:
    name = entry.get("ggufFilename") or ""
    if name in state.files and not SHARD.search(name) and "/" not in name:
        return state.files[name]
    same_case = [p for p in state.files if p.lower() == name.lower() and "/" not in p]
    nested = [p for p in state.files if p.rsplit("/", 1)[-1].lower() == name.rsplit("/", 1)[-1].lower() and "/" in p]
    parts = sorted(p for p in state.files if SHARD.search(p) and SHARD.sub("", p).lower() == name.lower().removesuffix(".gguf"))
    if SHARD.search(name) or parts:
        total = sum(state.files[p] for p in parts) if parts else None
        diffs.append(Diff("ggufFilename", name, parts[0] if parts else name, False,
                          f"split into {len(parts) or '?'} parts; the GGUF downloader skips shards"
                          + (f" ({mib(total)} MiB)" if total else "")))
    elif same_case:
        diffs.append(Diff("ggufFilename", name, same_case[0], True, "case differs; resolve URLs are case-sensitive"))
        return state.files[same_case[0]]
    elif nested:
        diffs.append(Diff("ggufFilename", name, nested[0], False, "only in a subfolder; the downloader fetches the root"))
    else:
        quants = sorted(p for p in state.files if p.endswith(".gguf"))
        diffs.append(Diff("ggufFilename", name, None, False, f"not in the repo ({len(quants)} .gguf files listed)"))
    return None


def mlx_file_diffs(entry: dict, state: RepoState, report: dict | None, diffs: list[Diff]) -> int | None:
    if "config.json" not in state.files:
        diffs.append(Diff("files", "config.json", None, False, "no top-level config.json"))
    weights = {p: s for p, s in state.files.items() if p.endswith(".safetensors") and "/" not in p}
    if not weights:
        diffs.append(Diff("files", "*.safetensors", None, False, "no top-level safetensors"))
        return None
    return (report or {}).get("weightsBytes") or sum(weights.values())


# MARK: - Catalog I/O


def inline(value) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(f"{json.dumps(k)}: {inline(v)}" for k, v in value.items()) + "}"
    return json.dumps(value, ensure_ascii=False)


def dump(catalog: dict) -> str:
    """The catalog's own layout: two-space indent, nested objects on one line (CatalogEntry.jsonText)."""
    entries = []
    for entry in catalog["models"]:
        body = ",\n".join(f"      {json.dumps(k)}: {inline(v)}" for k, v in entry.items())
        entries.append("    {\n" + body + "\n    }")
    return ('{\n  "schemaVersion": ' + json.dumps(catalog["schemaVersion"]) + ',\n  "models": [\n'
            + ",\n".join(entries) + "\n  ]\n}\n")


def set_field(entry: dict, key: str, value) -> dict:
    """Set or drop `key`, placing a new key where CatalogEntry.jsonText would."""
    if value is None and key in ("kvLayers", "experts", "maxContextLength"):
        entry.pop(key, None)
        return entry
    if key in entry:
        entry[key] = value
        return entry
    rank = KEY_ORDER.index(key)
    items = list(entry.items())
    at = next((i for i, (k, _) in enumerate(items) if k in KEY_ORDER and KEY_ORDER.index(k) > rank), len(items))
    items.insert(at, (key, value))
    return dict(items)


def load_exceptions(path: Path) -> dict:
    """`{"entries": {"<id>": {"fields": ["approximateSizeMB"], "reason": "…"}}}`: differences reported, not failed."""
    if not path.exists():
        return {}
    return json.loads(path.read_text())["entries"]


# MARK: - Main


def log(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def find_aura(explicit: str | None) -> str | None:
    for candidate in [explicit, ROOT / ".build/debug/aura", ROOT / ".build/release/aura", shutil.which("aura")]:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fix", action="store_true", help="write fixable corrections to both catalogs")
    parser.add_argument("--only", help="comma-separated entry ids")
    parser.add_argument("--aura", help="aura binary (default: .build/debug/aura, .build/release/aura, PATH)")
    parser.add_argument("--no-aura", action="store_true", help="skip the architecture checks")
    parser.add_argument("--device", default="mac-64gb", help="aura device preset for the not-runnable notes")
    parser.add_argument("--size-tolerance", type=float, default=0.02, help="relative size drift allowed (default 0.02)")
    parser.add_argument("--max-requests", type=int, default=450, help="requests per 300 s (HF allows 500)")
    parser.add_argument("--exceptions", type=Path, default=EXCEPTIONS,
                        help="documented exceptions (default: scripts/catalog-exceptions.json, optional)")
    parser.add_argument("--report", type=Path, help="also write the differences as JSON here")
    args = parser.parse_args()

    texts = [path.read_text() for path in CATALOGS]
    if texts[0] != texts[1]:
        log(f"{CATALOGS[0]} and {CATALOGS[1]} differ; make them identical first")
        return 2
    catalog = json.loads(texts[0])
    if dump(catalog) != texts[0]:
        log("note: the catalog layout differs from this script's writer; --fix will reformat it")
    aura = None if args.no_aura else find_aura(args.aura)
    if not args.no_aura and not aura:
        log("aura not found: run `swift build --product aura`, pass --aura, or use --no-aura")
        return 2
    only = set(args.only.split(",")) if args.only else None
    exceptions = load_exceptions(args.exceptions)
    limiter = RateLimiter(args.max_requests)

    results, failing, fixed = [], 0, 0
    for index, entry in enumerate(catalog["models"]):
        if only and entry["id"] not in only:
            continue
        log(f"[{index + 1}/{len(catalog['models'])}] {entry['id']}  {entry['repoID']}")
        state = fetch_repo(entry["repoID"], limiter)
        report = None
        if aura and state.status in ("ok", "renamed"):
            gguf = listed_gguf(entry, state.files) if entry["format"] == "gguf" else None
            report = aura_report(aura, state.canonical, args.device, aura_cost(state.files), limiter, gguf)
        if state.status in ("gone", "error"):
            diffs = [Diff("repoID", entry["repoID"], None, False, state.detail)]
        else:
            diffs = compare(entry, state, report, args.size_tolerance, architecture=aura is not None)
        skipped = set((exceptions.get(entry["id"]) or {}).get("fields", []))
        open_diffs = [d for d in diffs if d.field not in skipped]
        verdict = (report or {}).get("verdict")
        notes = []
        if state.gated:
            notes.append(f"gated ({state.gated}): downloads need a Hugging Face token")
        if verdict == "notRunnable":
            notes.append(f"aura: not runnable — {report.get('headline')}")
        print_entry(entry, open_diffs, [d for d in diffs if d.field in skipped], notes)
        failing += bool(open_diffs)
        results.append({"id": entry["id"], "repoID": entry["repoID"], "status": state.status, "gated": state.gated,
                        "verdict": verdict, "notes": notes,
                        "differences": [d.__dict__ for d in open_diffs],
                        "excepted": [d.__dict__ for d in diffs if d.field in skipped]})
        if args.fix:
            for diff in open_diffs:
                if diff.fixable:
                    catalog["models"][index] = entry = set_field(entry, diff.field, diff.live)
                    fixed += 1

    checked = len(results)
    print(f"\n{checked} entries checked, {checked - failing} match, {failing} differ; "
          f"{limiter.total} HF requests (aura's counted at an upper bound)")
    if args.report:
        args.report.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    if args.fix and fixed:
        text = dump(catalog)
        for path in CATALOGS:
            path.write_text(text)
        print(f"--fix: wrote {fixed} corrections to {', '.join(os.path.relpath(p, ROOT) for p in CATALOGS)}; "
              "unfixable differences stay listed above")
    return 1 if failing else 0


def print_entry(entry: dict, diffs: list[Diff], excepted: list[Diff], notes: list[str]) -> None:
    mark = "DIFF" if diffs else "ok  "
    print(f"{mark} {entry['id']}  ({entry['repoID']})")
    for diff in diffs:
        fix = "" if diff.fixable else "  [manual]"
        note = f"  — {diff.note}" if diff.note else ""
        print(f"       {diff.field}: {inline(diff.catalog)} → {inline(diff.live)}{fix}{note}")
    for diff in excepted:
        print(f"       (excepted) {diff.field}: {inline(diff.catalog)} → {inline(diff.live)}")
    for note in notes:
        print(f"       note: {note}")


if __name__ == "__main__":
    sys.exit(main())
