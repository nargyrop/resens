from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    print(repo_root)
    docs_dir = repo_root
    source_dir = docs_dir / "source"
    build_dir = docs_dir / "build" / "html"

    if not source_dir.exists():
        raise SystemExit(f"Missing Sphinx source directory: {source_dir}")

    cmd = [
        sys.executable,
        "-m",
        "sphinx",
        "-b",
        "html",
        str(source_dir),
        str(build_dir),
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(repo_root))
    print(f"Built HTML docs at: {build_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
