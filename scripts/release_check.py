"""Pre-release audit: what would ship, and whether it should.

Answers one question before a production release: if this repository were made
public right now, exactly which files would leave the machine, and is there
anything in them that should not?

Publishing is not reversible. A key, an absolute path, or a stray recording is
public the moment it is pushed, and deleting it afterwards does not remove it
from the history or from anything that mirrored it. So this refuses to pass on
anything it cannot verify, rather than assuming a clean tree means clean content.

    python -m scripts.release_check              # audit, list what ships
    python -m scripts.release_check --files      # just the manifest
    python -m scripts.release_check --strict     # warnings are failures too
    python -m scripts.release_check --json       # machine-readable

Exit codes: 0 clean, 1 warnings (0 unless --strict), 2 failures.

Standard library only, so it runs on a bare checkout with nothing installed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Content scanning skips this file: it necessarily contains every pattern it
# looks for, and would otherwise report itself.
SCAN_EXEMPT = {"scripts/release_check.py"}

TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".yml", ".yaml",
    ".html", ".css", ".js", ".json", ".sh", ".ps1", ".gitignore", ".gitattributes",
}

# Anything matching these should never be in a published tree, whatever the
# .gitignore claims. Checked against tracked paths, not the working directory.
FORBIDDEN_PATHS = [
    (re.compile(r"\.(wav|mp3|flac|m4a|ogg)$", re.I), "audio recording"),
    (re.compile(r"\.(pem|key|p12|pfx|keystore)$", re.I), "key material"),
    (re.compile(r"(^|/)\.env(\.|$)", re.I), "environment file"),
    (re.compile(r"(^|/)(venv|\.venv|env)/", re.I), "virtual environment"),
    (re.compile(r"(^|/)models?/", re.I), "model weights"),
    (re.compile(r"\.(bin|onnx|gguf|pt|pth|safetensors|ckpt)$", re.I), "model weights"),
    (re.compile(r"(^|/)__pycache__/|\.pyc$", re.I), "bytecode"),
    (re.compile(r"(^|/)(id_rsa|id_ed25519|\.npmrc|\.pypirc)$", re.I), "credentials"),
]

SECRET_PATTERNS = [
    (re.compile(r"[A-Za-z]:\\+Users\\+[^\\\s\"']+"), "Windows user path"),
    (re.compile(r"/(?:home|Users)/(?!runner\b)[A-Za-z0-9._-]+/"), "home directory path"),
    (re.compile(r"\b(?:sk|pk)-[A-Za-z0-9]{20,}"), "API key"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}"), "GitHub token"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "Google API key"),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), "Slack token"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (re.compile(r"""(?i)\b(?:password|passwd|secret|api[_-]?key|token)\s*[:=]\s*["'][^"'\s]{8,}["']"""),
     "hardcoded credential"),
]

# Personal email addresses. GitHub noreply is fine; it is the whole point of it.
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@(?!users\.noreply\.github\.com)[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Placeholder text that should not survive into a production release.
PLACEHOLDER_PATTERN = re.compile(r"\b(TBD|FIXME|XXX|HACK|TODO)\b")

LARGE_FILE_BYTES = 1_000_000


class Status(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class Check:
    name: str
    status: Status
    detail: str
    items: list[str] = field(default_factory=list)


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def tracked_files() -> list[str]:
    """Exactly what git would publish. Not a directory walk: the working tree
    contains plenty that is correctly ignored, and only tracked content ships."""
    listing = git("ls-files")
    return [line for line in listing.splitlines() if line]


def read_text(path: Path) -> str | None:
    if path.suffix.lower() not in TEXT_SUFFIXES and path.suffix:
        return None
    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except (UnicodeDecodeError, OSError):
        return None


# --- individual checks ---------------------------------------------------


def check_repo_state(files: list[str]) -> list[Check]:
    checks = []

    if not files:
        return [Check("git repository", Status.FAIL, "not a git repo, or nothing is tracked")]

    dirty = git("status", "--porcelain")
    checks.append(
        Check("working tree", Status.PASS, "clean") if not dirty
        else Check("working tree", Status.FAIL,
                   "uncommitted changes: a release must be reproducible from a commit",
                   dirty.splitlines())
    )

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    upstream = git("rev-parse", "--abbrev-ref", "@{upstream}")
    if not upstream:
        checks.append(Check("upstream", Status.WARN, f"branch '{branch}' tracks no remote"))
    else:
        ahead = git("rev-list", "--count", f"{upstream}..HEAD")
        behind = git("rev-list", "--count", f"HEAD..{upstream}")
        if ahead == "0" and behind == "0":
            checks.append(Check("upstream", Status.PASS, f"in sync with {upstream}"))
        else:
            checks.append(Check("upstream", Status.WARN,
                                f"{ahead} ahead, {behind} behind {upstream}"))

    return checks


def check_forbidden_paths(files: list[str]) -> Check:
    offenders = [
        f"{path}  ({reason})"
        for path in files
        for pattern, reason in FORBIDDEN_PATHS
        if pattern.search(path)
    ]
    if offenders:
        return Check("tracked file types", Status.FAIL,
                     "files that must never be published are tracked", offenders)
    return Check("tracked file types", Status.PASS, "no audio, weights, keys or venvs tracked")


def check_file_sizes(files: list[str]) -> Check:
    large = []
    total = 0
    for path in files:
        full = REPO_ROOT / path
        if not full.exists():
            continue
        size = full.stat().st_size
        total += size
        if size > LARGE_FILE_BYTES:
            large.append(f"{path}  ({size / 1_000_000:.1f} MB)")

    summary = f"{len(files)} files, {total / 1_000_000:.2f} MB total"
    if large:
        return Check("file sizes", Status.WARN, f"{summary}; large files present", large)
    return Check("file sizes", Status.PASS, summary)


def check_content(files: list[str]) -> list[Check]:
    secrets: list[str] = []
    emails: list[str] = []
    placeholders: list[str] = []

    for path in files:
        if path in SCAN_EXEMPT:
            continue
        content = read_text(REPO_ROOT / path)
        if content is None:
            continue

        for number, line in enumerate(content.splitlines(), start=1):
            for pattern, label in SECRET_PATTERNS:
                match = pattern.search(line)
                if match:
                    secrets.append(f"{path}:{number}  {label}: {match.group()[:60]}")
            match = EMAIL_PATTERN.search(line)
            if match:
                emails.append(f"{path}:{number}  {match.group()}")
            match = PLACEHOLDER_PATTERN.search(line)
            if match:
                placeholders.append(f"{path}:{number}  {match.group()}")

    checks = [
        Check("secrets and paths", Status.FAIL, "possible secrets or machine paths", secrets)
        if secrets else
        Check("secrets and paths", Status.PASS, "no keys, tokens or absolute paths found"),

        Check("email addresses", Status.WARN, "personal addresses in shipped files", emails)
        if emails else
        Check("email addresses", Status.PASS, "none outside github noreply"),

        Check("placeholders", Status.WARN, "unfinished markers in shipped files", placeholders)
        if placeholders else
        Check("placeholders", Status.PASS, "no TBD/TODO/FIXME markers"),
    ]
    return checks


def check_commit_authors() -> Check:
    """History ships too, and it is the part people forget to audit."""
    authors = {line for line in git("log", "--format=%ae").splitlines() if line}
    personal = sorted(a for a in authors if not a.endswith("users.noreply.github.com"))
    if personal:
        return Check("commit authors", Status.WARN,
                     "history exposes personal addresses; rewriting after publishing is far harder",
                     personal)
    return Check("commit authors", Status.PASS, f"{len(authors)} author address(es), all noreply")


def check_release_files(files: list[str]) -> list[Check]:
    present = set(files)
    checks = []

    licence = [f for f in present if f.upper().startswith(("LICENSE", "LICENCE"))]
    checks.append(
        Check("licence", Status.PASS, licence[0]) if licence
        else Check("licence", Status.FAIL,
                   "no LICENSE file: without one, nobody may legally use, fork or contribute")
    )

    checks.append(
        Check("readme", Status.PASS, "present") if "README.md" in present
        else Check("readme", Status.FAIL, "no README.md")
    )

    if any(f.startswith(".github/workflows/") for f in present):
        checks.append(Check("ci", Status.PASS, "workflow present"))
    else:
        checks.append(Check("ci", Status.WARN, "no CI workflow"))

    checks.append(
        Check("gitignore", Status.PASS, "present") if ".gitignore" in present
        else Check("gitignore", Status.FAIL, "no .gitignore")
    )

    return checks


def check_dependencies() -> Check:
    requirements = REPO_ROOT / "requirements.txt"
    if not requirements.exists():
        return Check("dependencies", Status.WARN, "no requirements.txt")

    unpinned = []
    for number, raw in enumerate(requirements.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        if "==" not in line:
            unpinned.append(f"requirements.txt:{number}  {line}")

    if unpinned:
        return Check("dependencies", Status.WARN,
                     "unpinned requirements make builds irreproducible", unpinned)
    return Check("dependencies", Status.PASS, "all requirements pinned")


def check_server_defaults() -> Check:
    """A local-first tool must not start by listening on every interface."""
    entrypoint = REPO_ROOT / "server" / "__main__.py"
    if not entrypoint.exists():
        return Check("server binding", Status.WARN, "server/__main__.py not found")

    content = entrypoint.read_text(encoding="utf-8")
    if re.search(r"""default\s*=\s*["']0\.0\.0\.0["']""", content):
        return Check("server binding", Status.FAIL,
                     "default host is 0.0.0.0: exposes the microphone feed to the network")
    if re.search(r"""default\s*=\s*["']127\.0\.0\.1["']""", content):
        return Check("server binding", Status.PASS, "defaults to 127.0.0.1")
    return Check("server binding", Status.WARN, "could not confirm the default bind address")


def check_tooling() -> list[Check]:
    """Run the suite and linter if they are installed. Absence is not a failure:
    this script must work on a bare checkout."""
    checks = []
    for name, command in (("tests", [sys.executable, "-m", "pytest", "-q"]),
                          ("lint", [sys.executable, "-m", "ruff", "check", "."])):
        result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        output = (result.stdout + result.stderr).strip()
        if "No module named" in output:
            checks.append(Check(name, Status.WARN, "not installed, skipped"))
        elif result.returncode == 0:
            checks.append(Check(name, Status.PASS, output.splitlines()[-1] if output else "passed"))
        else:
            checks.append(Check(name, Status.FAIL, "failed", output.splitlines()[-25:]))
    return checks


# --- reporting -----------------------------------------------------------


SYMBOLS = {Status.PASS: "PASS", Status.WARN: "WARN", Status.FAIL: "FAIL"}


def report(checks: list[Check], files: list[str], show_files: bool) -> int:
    width = max(len(c.name) for c in checks)

    print("\nRELEASE AUDIT\n" + "=" * 72)
    for check in checks:
        print(f"  [{SYMBOLS[check.status]}]  {check.name.ljust(width)}  {check.detail}")
        for item in check.items[:15]:
            print(f"            {item}")
        if len(check.items) > 15:
            print(f"            ... and {len(check.items) - 15} more")

    failures = [c for c in checks if c.status is Status.FAIL]
    warnings = [c for c in checks if c.status is Status.WARN]

    if show_files:
        print("\nWOULD PUBLISH\n" + "=" * 72)
        for path in files:
            full = REPO_ROOT / path
            size = full.stat().st_size if full.exists() else 0
            print(f"  {size / 1000:8.1f} KB  {path}")

    print("\n" + "=" * 72)
    if failures:
        print(f"  NOT READY: {len(failures)} failure(s), {len(warnings)} warning(s)")
        print("  Publishing is irreversible. Resolve the failures first.")
        return 2
    if warnings:
        print(f"  READY with {len(warnings)} warning(s) to review")
        return 1
    print(f"  READY: {len(files)} files cleared for release")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files", action="store_true", help="list every file that would be published")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    parser.add_argument("--skip-tooling", action="store_true", help="do not run pytest or ruff")
    args = parser.parse_args(argv)

    files = tracked_files()

    checks: list[Check] = []
    checks += check_repo_state(files)
    checks.append(check_forbidden_paths(files))
    checks.append(check_file_sizes(files))
    checks += check_content(files)
    checks.append(check_commit_authors())
    checks += check_release_files(files)
    checks.append(check_dependencies())
    checks.append(check_server_defaults())
    if not args.skip_tooling:
        checks += check_tooling()

    if args.json:
        print(json.dumps({
            "files": files,
            "checks": [
                {"name": c.name, "status": c.status.value, "detail": c.detail, "items": c.items}
                for c in checks
            ],
        }, indent=2))
        return 2 if any(c.status is Status.FAIL for c in checks) else 0

    code = report(checks, files, show_files=args.files)
    if args.strict and code == 1:
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
