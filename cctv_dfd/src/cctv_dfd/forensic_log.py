"""Forensic-ready logging.

Per-prediction:
- JSONL line with input SHA-256, model SHA-256, prediction, probability,
  degradation profile, enhancement mode, code git SHA, seed, timestamp,
  paths to inputs and explanation overlay.
- HTML report (Jinja2) with input thumbnail + attention overlay + hashes.

Phrasing in user-facing output uses "forensic-ready logging" /
"auditable inference trace" — **not** "court admissible".
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from jinja2 import Template

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hashing + git SHA
# ---------------------------------------------------------------------------


def sha256_file(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha(cwd: Optional[Path] = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(cwd) if cwd else None, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Record schema
# ---------------------------------------------------------------------------


@dataclass
class ForensicRecord:
    ts: str
    input_path: str
    input_sha256: str
    model_id: str
    model_sha256: str
    prediction: str            # "fake" | "real"
    prob_fake: float
    degradation_profile: str   # name from PROFILE_NAMES
    enhance_mode: str          # name from ENHANCE_MODES
    explanation_path: str = ""
    seed: int = 1337
    code_git_sha: str = field(default_factory=git_sha)


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------


def append_jsonl(record: ForensicRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def write_record(
    input_path: Path,
    model_path: Path,
    *,
    prob_fake: float,
    degradation_profile: str = "clean",
    enhance_mode: str = "none",
    explanation_path: str = "",
    log_path: Optional[Path] = None,
    seed: int = 1337,
) -> ForensicRecord:
    rec = ForensicRecord(
        ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        input_path=str(input_path),
        input_sha256=sha256_file(input_path),
        model_id=model_path.stem,
        model_sha256=sha256_file(model_path),
        prediction="fake" if prob_fake >= 0.5 else "real",
        prob_fake=float(prob_fake),
        degradation_profile=degradation_profile,
        enhance_mode=enhance_mode,
        explanation_path=explanation_path,
        seed=seed,
    )
    if log_path is None:
        log_path = Path.cwd() / "results" / "forensic.jsonl"
    append_jsonl(rec, log_path)
    return rec


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


_HTML_TEMPLATE = Template(
    """<!doctype html>
<html><head><meta charset="utf-8"><title>Forensic report — {{ rec.input_sha256[:12] }}</title>
<style>
body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;color:#222}
h1{font-size:1.3rem}
table{border-collapse:collapse;margin:1rem 0}
td,th{border:1px solid #ccc;padding:.4rem .8rem;font-size:.9rem;text-align:left}
.badge{display:inline-block;padding:.2rem .6rem;border-radius:.3rem;color:#fff;font-weight:600}
.fake{background:#b03030}.real{background:#308040}
img{max-width:380px;border:1px solid #ccc;margin:.5rem}
.note{font-size:.8rem;color:#666;margin-top:2rem;line-height:1.4}
</style></head><body>
<h1>Forensic-ready inference trace</h1>
<p><span class="badge {{ rec.prediction }}">{{ rec.prediction|upper }}</span>
&nbsp;p(fake) = <b>{{ '%.3f'|format(rec.prob_fake) }}</b></p>

<table>
<tr><th>Timestamp (UTC)</th><td>{{ rec.ts }}</td></tr>
<tr><th>Input path</th><td>{{ rec.input_path }}</td></tr>
<tr><th>Input SHA-256</th><td><code>{{ rec.input_sha256 }}</code></td></tr>
<tr><th>Model ID</th><td>{{ rec.model_id }}</td></tr>
<tr><th>Model SHA-256</th><td><code>{{ rec.model_sha256 }}</code></td></tr>
<tr><th>Degradation profile</th><td>{{ rec.degradation_profile }}</td></tr>
<tr><th>Enhancement mode</th><td>{{ rec.enhance_mode }}</td></tr>
<tr><th>Code git SHA</th><td><code>{{ rec.code_git_sha or '(no git)' }}</code></td></tr>
<tr><th>Seed</th><td>{{ rec.seed }}</td></tr>
</table>

{% if rec.explanation_path %}
<h2>Attention overlay</h2>
<img src="{{ rec.explanation_path }}" alt="attention rollout overlay">
{% endif %}

<p class="note">This document records an automated inference trace produced by
the <code>cctv_dfd</code> pipeline. The hashes establish the bytes of the input
and model that were processed at the recorded timestamp. <b>This is
forensic-ready logging, not a court-admissibility certification.</b> Use of the
output as evidence requires independent expert review and adherence to the
applicable evidentiary rules of the relevant jurisdiction.</p>
</body></html>
"""
)


def render_html_report(rec: ForensicRecord, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{rec.input_sha256[:16]}.html"
    out_path.write_text(_HTML_TEMPLATE.render(rec=rec))
    return out_path
