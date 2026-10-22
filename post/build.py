"""Build the single-page blog: post/index.md -> docs/index.html (served by GitHub Pages from main:/docs).

    python3 post/build.py            # writes docs/, then opens it
    python3 post/build.py --no-open

Needs `pip install markdown`. Copies assets/style.css and every figure the post links to into docs/;
anything else in docs/ is left alone.
"""
import html
import re
import shutil
import sys
import webbrowser
from pathlib import Path

try:
    import markdown
except ImportError:
    sys.exit("pip install markdown")

POST = Path(__file__).resolve().parent
DOCS = POST.parent / "docs"

src = (POST / "index.md").read_text()
front = {}
m = re.match(r"^---\n(.*?)\n---\n", src, re.S)
if m:
    for line in m.group(1).splitlines():
        k, _, v = line.partition(":")
        front[k.strip()] = v.strip().strip('"')
    src = src[m.end():]

gaps = len(re.findall(r"\[Write", src))
if gaps:
    print(f"warning: {gaps} [Write] gaps left")

body = markdown.markdown(src, extensions=["tables", "fenced_code", "toc"])


def color_diff(match):
    lines = []
    for line in match.group(1).split("\n"):
        kind = {"+": "add", "-": "del"}.get(line[:1])
        lines.append(f'<span class="{kind}">{line}</span>' if kind else line)
    return '<code class="language-diff">' + "\n".join(lines) + "</code>"


body = re.sub(r'<code class="language-diff">(.*?)</code>', color_diff, body, flags=re.S)
body = body.replace("<table>", '<div class="table-scroll"><table>').replace("</table>", "</table></div>")
body = body.replace("<img ", '<img loading="lazy" ')

meta = " · ".join(x for x in (front.get("author"), front.get("dated")) if x)
title = html.escape(front.get("title", ""))
page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{html.escape(front.get("description", ""))}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400..700;1,6..72,400..600&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<link rel="stylesheet" href="assets/style.css">
</head>
<body>
<main>
<article>
<header>
<h1>{title}</h1>
<p class="meta">{html.escape(meta)}</p>
</header>
{body}
</article>
</main>
</body>
</html>
"""

DOCS.mkdir(exist_ok=True)
(DOCS / "index.html").write_text(page)
(DOCS / "assets").mkdir(exist_ok=True)
shutil.copy2(POST / "assets" / "style.css", DOCS / "assets" / "style.css")
(DOCS / ".nojekyll").touch()

linked = sorted(set(re.findall(r'(?:src|href)="(figures/[^"]+)"', body)))
for rel in linked:
    (DOCS / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(POST / rel, DOCS / rel)

words = len(re.sub(r"<[^>]+>", " ", body).split())
print(f"wrote {DOCS / 'index.html'}  ({words} words, {len(linked)} figure files)")
if "--no-open" not in sys.argv:
    webbrowser.open((DOCS / "index.html").as_uri())
