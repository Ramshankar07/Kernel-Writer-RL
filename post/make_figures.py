"""Render the post's editable Excalidraw diagrams to PNG.

Run from the repository root: python3 post/make_figures.py
Requires Playwright, Chromium, and network access to the pinned export library.
"""

from pathlib import Path
import subprocess
import sys

POST = Path(__file__).resolve().parent
RENDERER = POST / "renderer" / "render_excalidraw.py"


def main():
    for name in ("grpo_group_signal", "grpo_training", "grpo_step_time"):
        subprocess.run(
            [sys.executable, str(RENDERER), str(POST / "figures" / f"{name}.excalidraw")],
            check=True,
        )


if __name__ == "__main__":
    main()
