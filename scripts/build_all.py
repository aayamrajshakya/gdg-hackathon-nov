"""One command to go from nothing to a demo-ready system:

    uv run python scripts/build_all.py

download dataset -> 80/20 split -> EmbeddingGemma vectors -> train + tune gate -> evaluate
"""

import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from sentineldesk import data

HERE = Path(__file__).resolve().parent


def run(script: str, *args: str) -> None:
    print(f"\n=== {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, str(HERE / script), *args], check=True)


if __name__ == "__main__":
    data.prepare_splits()
    run("build_embeddings.py")
    run("train.py")
    run("evaluate.py", *sys.argv[1:])
