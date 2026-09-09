#!/usr/bin/env python3
"""Bundle the site into one self-contained HTML file.

Inlines the CSS, every script, and KaTeX (JS + CSS with woff2 fonts as data
URIs) so the result makes zero external requests and can be published anywhere.

    python3 site/build.py            -> site/dist/index.html
    python3 site/build.py --body     -> site/dist/artifact.html  (no <html>/<head>
                                        wrapper, for the Artifact publisher)
    python3 site/build.py --pages    -> docs/index.html          (what GitHub
                                        Pages serves at the repo's public URL)
"""
import base64
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).parent
DIST = ROOT / "dist"
DIST.mkdir(exist_ok=True)

html = (ROOT / "index.html").read_text()


def inline_katex_css() -> str:
    css = (ROOT / "vendor/katex/dist/katex.min.css").read_text()
    fonts = ROOT / "vendor/katex/dist/fonts"

    def repl(m):
        url = m.group(1).strip("\"'")
        if not url.startswith("fonts/") or not url.endswith(".woff2"):
            return ""  # drop ttf/woff variants entirely
        data = (fonts / pathlib.Path(url).name).read_bytes()
        b64 = base64.b64encode(data).decode()
        return f"url(data:font/woff2;base64,{b64}) format(\"woff2\")"

    css = re.sub(r"url\(([^)]+)\)\s*format\(\"woff2\"\)", repl, css)
    # remove now-dangling references to the pruned formats
    css = re.sub(r",?\s*url\([^)]*\.(ttf|woff)\)\s*format\(\"(truetype|woff)\"\)", "", css)
    return css


parts = {
    "katex_css": inline_katex_css(),
    "app_css": (ROOT / "css/app.css").read_text(),
}

scripts = re.findall(r'<script src="([^"]+)"></script>', html)
js_blobs = []
for src in scripts:
    p = ROOT / src
    if not p.exists():
        print(f"  ! missing {src}", file=sys.stderr)
        continue
    js_blobs.append(f"/* ==== {src} ==== */\n" + p.read_text())

body = re.search(r"<body>(.*)</body>", html, re.S).group(1)
body = re.sub(r'<script src="[^"]+"></script>\s*', "", body)

title = re.search(r"<title>(.*?)</title>", html, re.S).group(1)
desc = re.search(r'<meta name="description" content="([^"]*)"', html).group(1)

bundle_style = f"<style>\n{parts['katex_css']}\n</style>\n<style>\n{parts['app_css']}\n</style>"
bundle_script = "<script>\n" + "\n".join(js_blobs) + "\n</script>"

body_only = "--body" in sys.argv
if body_only:
    out = f"""<title>{title}</title>
{bundle_style}

{body}
{bundle_script}
"""
    target = DIST / "artifact.html"
else:
    out = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{title}</title>
<meta name="description" content="{desc}">
{bundle_style}
</head>
<body>
{body}
{bundle_script}
</body>
</html>
"""
    target = DIST / "index.html"

if "--pages" in sys.argv:
    # GitHub Pages serves this one; .nojekyll stops Jekyll touching it
    pages = ROOT.parent / "docs"
    pages.mkdir(exist_ok=True)
    (pages / ".nojekyll").write_text("")
    target = pages / "index.html"

target.write_text(out)
kb = len(out.encode()) / 1024
print(f"wrote {target.relative_to(ROOT.parent)}  {kb:,.0f} KB  ({len(js_blobs)} scripts inlined)")
