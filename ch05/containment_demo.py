"""The path-containment flaw class from Section 5.8, in miniature, using a temporary directory."""

import tempfile
from pathlib import Path


def naive_inside(root: Path, requested: str) -> bool:
    """A string-prefix test: does the requested path *start with* the allowed root?"""
    return requested.startswith(str(root))


def inside(root: Path, requested: str) -> bool:
    """Resolve first (follow symlinks, collapse '..'), then compare whole path
    components."""
    return Path(requested).resolve().is_relative_to(root.resolve())


with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp).resolve()
    allowed = base / "allow_dir"
    sibling = base / "allow_dir_sensitive_credentials"   # shares the prefix, is NOT inside
    secret = base / "secret"
    for folder in (allowed, sibling, secret):
        folder.mkdir()
    (sibling / "creds.txt").write_text("hunter2")
    (secret / "keys.txt").write_text("k3y")
    (allowed / "notes.txt").write_text("fine")
    (allowed / "shortcut").symlink_to(secret / "keys.txt")   # a link inside pointing outside

    requests = {
        "a file inside":                str(allowed / "notes.txt"),
        "a sibling sharing the prefix": str(sibling / "creds.txt"),
        "a '..' escape":                str(allowed / ".." / "secret" / "keys.txt"),
        "a symlink pointing outside":   str(allowed / "shortcut"),
    }
    print(f"{'request':32} {'prefix test':>12} {'resolve + contain':>18}")
    for label, requested in requests.items():
        print(f"{label:32} {str(naive_inside(allowed, requested)):>12} "
              f"{str(inside(allowed, requested)):>18}")
