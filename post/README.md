# The post

A first-person write-up of one GRPO setup, its verifier audit, and supervised fine-tuning results.
It is the whole website: `build.py` renders it to a single static page in `docs/`, which GitHub Pages serves from `main:/docs`.

| File | What it is |
|---|---|
| `index.md` | The post. Front matter sets `title`, `description`, `author` (blank = no byline) and `dated`. |
| `assets/style.css` | The page style: Newsreader body, IBM Plex Sans/Mono for tables and code, warm paper with a rust accent, automatic dark mode. |
| `figures/` | Editable `.excalidraw` diagrams and their rendered PNGs. |
| `build.py` | `python3 post/build.py` writes `docs/index.html`, copies the CSS and every linked figure, then opens it. Needs `pip install markdown`. |
| `make_figures.py`, `renderer/` | Re-export the Excalidraw diagrams (Playwright + Chromium, network access for the pinned export library). |
| `_FACTS.md` | Every number you might quote, with its source file. Not published. |

## Editing

1. Edit prose in `index.md`, or a diagram in `figures/*.excalidraw` (then `python3 post/make_figures.py`).
2. Run `python3 post/build.py` and read the page.
3. Commit `post/` and `docs/` together in a PR into `main`.

The renderer is adapted from `.agents/skills/excalidraw-diagram/references/`; its local template fixes a broken
transitive dependency URL by pinning Excalidraw and `sanitize-url`.
