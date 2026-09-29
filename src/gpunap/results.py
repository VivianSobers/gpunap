"""Verdicts, check results, the run report, and file naming that never overwrites."""
from __future__ import annotations

import getpass
import json
import os
import re
import socket
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List

import gpunap

OK, HAZARD, SKIPPED, INVALID, ERROR = "ok", "hazard", "skipped", "invalid", "error"
VERDICTS = (OK, HAZARD, SKIPPED, INVALID, ERROR)


@dataclass
class CheckResult:
    name: str
    verdict: str
    summary: str
    data: Dict = field(default_factory=dict)
    seconds: float = 0.0

    def __post_init__(self):
        if self.verdict not in VERDICTS:
            raise ValueError(f"unknown verdict {self.verdict!r}")

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["seconds"] = round(self.seconds, 2)
        return d


@dataclass
class Report:
    header: Dict
    results: List[CheckResult] = field(default_factory=list)
    partial: bool = False

    def to_dict(self) -> Dict:
        counts = {v: sum(1 for x in self.results if x.verdict == v) for v in VERDICTS}
        return {"gpunap": gpunap.__version__, "header": self.header, "partial": self.partial,
                "counts": counts, "results": [x.to_dict() for x in self.results]}


def claim(path: str) -> str:
    """Create path exclusively and return it; if it exists, use <root>-2<ext>, <root>-3<ext>, ..."""
    root, ext = os.path.splitext(path)
    candidate, n = path, 1
    while True:
        try:
            with open(candidate, "x"):
                return candidate
        except FileExistsError:
            n += 1
            candidate = f"{root}-{n}{ext}"


def default_name(label: str, when: float) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-") or "unlabelled"
    return f"gpunap-check-{safe}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(when))}.json"


def _identity() -> Dict[str, str]:
    try:
        user = getpass.getuser()
    except Exception:
        user = ""
    return {"home": os.path.expanduser("~"), "user": user, "host": socket.gethostname(), "uid": str(os.getuid())}


def redact(text: str, ident: Dict[str, str]) -> str:
    """Replace the home directory, user name, host name and UID in text. Results are meant to be shared."""
    if ident.get("home") and ident["home"] != "/":
        text = text.replace(ident["home"], "~")
    if ident.get("uid"):
        text = re.sub(rf"user-{ident['uid']}\b", "user-UID", text)
        text = re.sub(rf"user@{ident['uid']}\b", "user@UID", text)
    for key, repl in (("host", "HOST"), ("user", "USER")):
        if len(ident.get(key) or "") >= 3:
            text = re.sub(rf"\b{re.escape(ident[key])}\b", repl, text)
    return text


def write(report: Report, out_dir: str, label: str, when: float) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = claim(os.path.join(out_dir, default_name(label, when)))
    text = redact(json.dumps(report.to_dict(), indent=1, default=str), _identity())
    with open(path, "w") as f:
        f.write(text)
    return path
