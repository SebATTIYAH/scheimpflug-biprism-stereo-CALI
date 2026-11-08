"""Check tracked source files for private inputs and accidental binary media."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
BLOCKED_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp",
    ".gif", ".heic", ".raw", ".pdf", ".npy", ".npz", ".h5",
    ".hdf5", ".mat", ".avi", ".mov", ".mp4", ".zip", ".whl",
}
BLOCKED_DIRECTORIES = {
    "private_data", "local_experiment_runs", "inputs", "work", "outputs",
    "tmp", "build", "dist", ".venv", "venv", "__pycache__",
}
PRIVATE_PATH = re.compile(
    r"(?i)(?:[A-Z]:[/\\](?:Users|Documents)[/\\]|[/\\](?:Users|home)[/\\][^/\\\s]+[/\\])"
)
PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")


def main() -> int:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    tracked = [name.decode("utf-8") for name in result.stdout.split(b"\0") if name]
    problems = []
    for name in tracked:
        path = ROOT / name
        if any(part in BLOCKED_DIRECTORIES for part in Path(name).parts):
            problems.append(f"private/output directory: {name}")
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            problems.append(f"binary media or output: {name}")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"non-text file: {name}")
            continue
        if PRIVATE_PATH.search(content):
            problems.append(f"personal local path: {name}")
        if PRIVATE_KEY.search(content):
            problems.append(f"private key marker: {name}")
    if problems:
        print("\n".join(sorted(set(problems))), file=sys.stderr)
        return 1
    print(f"Audited {len(tracked)} tracked source files; no blocked media or private paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
