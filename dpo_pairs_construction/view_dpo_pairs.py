"""Script to render a DPO pairs file as a standalone HTML page, for readability.
Each pair is shown as the prompt (collapsible full context + the last turn always visible), then chosen vs rejected side by side.
Use it via python view_dpo_pairs.py --input dpo_pairs/onpolicy_pairs.aborted.<...>.json
"""

import argparse
import html
import json
import os
from collections import Counter
from pathlib import Path


def text_of(side):
    """A pair side is a list of {role, content}; join the assistant content(s)."""
    if isinstance(side, list):
        return "\n".join(m.get("content", "") for m in side)
    return str(side)


def turns_html(prompt):
    if not isinstance(prompt, list):
        return f"<pre>{html.escape(str(prompt))}</pre>"
    rows = []
    for m in prompt:
        role = m.get("role", "?")
        cls = "user" if role == "user" else "asst"
        rows.append(f'<div class="turn {cls}"><span class="role">{html.escape(role)}</span>'
                    f'<pre>{html.escape(str(m.get("content","")))}</pre></div>')
    return "".join(rows)


def diff_html(chosen, rejected):
    """Highlight the divergent tail of chosen and rejected after their common prefix."""
    i = 0
    while i < len(chosen) and i < len(rejected) and chosen[i] == rejected[i]:
        i += 1
    def render(full, color):
        common = html.escape(full[:i])
        tail = html.escape(full[i:])
        return (f'<span class="common">{common}</span>'
                f'<span class="{color}">{tail}</span>')
    return render(chosen, "ins"), render(rejected, "del")


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f5f5f5;color:#222}}
header{{position:sticky;top:0;background:#222;color:#fff;padding:10px 16px;z-index:10}}
header h1{{font-size:15px;margin:0 0 6px}}
header .meta{{font-size:12px;color:#bbb}}
#filter{{margin-top:8px;padding:4px 8px;width:280px;font-size:13px}}
.pair{{background:#fff;margin:14px;border:1px solid #ddd;border-radius:6px;overflow:hidden}}
.pair h2{{font-size:13px;margin:0;padding:8px 12px;background:#eef;border-bottom:1px solid #ddd}}
.pair h2 .game{{background:#447;color:#fff;border-radius:3px;padding:1px 7px;margin-right:8px;font-size:11px}}
details{{padding:8px 12px;border-bottom:1px solid #eee}}
summary{{cursor:pointer;font-size:12px;color:#666}}
.turn{{margin:4px 0}}
.turn .role{{font-size:10px;text-transform:uppercase;color:#888;display:block}}
.turn.user .role{{color:#960}}
.turn.asst .role{{color:#069}}
pre{{margin:2px 0;white-space:pre-wrap;word-break:break-word;font-family:ui-monospace,monospace;font-size:12px;line-height:1.4}}
.cols{{display:flex;gap:0}}
.col{{flex:1;padding:8px 12px}}
.col.chosen{{background:#f0fbf0;border-right:1px solid #ddd}}
.col.rejected{{background:#fdf1f1}}
.col h3{{font-size:11px;text-transform:uppercase;margin:0 0 4px}}
.col.chosen h3{{color:#2a7}}
.col.rejected h3{{color:#c44}}
.common{{color:#999}}
.ins{{color:#197d19;font-weight:bold;background:#d9f2d9}}
.del{{color:#b32020;font-weight:bold;background:#f6d6d6}}
.last-turn{{font-size:11px;color:#444;background:#fafafa;padding:6px 12px}}
</style>
<script>
function flt(){{var q=document.getElementById('filter').value.toLowerCase();
document.querySelectorAll('.pair').forEach(function(p){{
p.style.display = p.dataset.search.indexOf(q)>=0 ? '' : 'none';}});}}
</script>
</head><body>
<header><h1>{title}</h1><div class="meta">{meta}</div>
<input id="filter" placeholder="filter by game or text..." oninput="flt()"></header>
{body}
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=500, help="Max pairs to render (default 500).")
    args = ap.parse_args()

    data = json.load(open(args.input))
    total = len(data)
    games = Counter(d.get("game", "(anti-bleed)") for d in data)
    shown = data[:args.limit]

    blocks = []
    for k, d in enumerate(shown):
        game = d.get("game", "(anti-bleed)")
        pid = d.get("pair_id", str(k))
        chosen, rejected = text_of(d["chosen"]), text_of(d["rejected"])
        ch, rj = diff_html(chosen, rejected)
        prompt = d.get("prompt", [])
        last = ""
        if isinstance(prompt, list) and prompt:
            last = html.escape(str(prompt[-1].get("content", ""))[:300])
        search = html.escape((game + " " + chosen + " " + rejected).lower().replace('"', ''))
        blocks.append(f"""
<div class="pair" data-search="{search}">
  <h2><span class="game">{html.escape(game)}</span>{html.escape(pid)} <small>#{k}</small></h2>
  <div class="last-turn"><b>last prompt turn:</b> {last}</div>
  <details><summary>full prompt ({len(prompt) if isinstance(prompt,list) else 1} turns)</summary>{turns_html(prompt)}</details>
  <div class="cols">
    <div class="col chosen"><h3>chosen</h3><pre>{ch}</pre></div>
    <div class="col rejected"><h3>rejected</h3><pre>{rj}</pre></div>
  </div>
</div>""")

    by_game = ", ".join(f"{g}: {n}" for g, n in games.most_common())
    meta = f"{total} pairs total" + (f" (showing first {len(shown)})" if len(shown) < total else "") + f" - {by_game}"
    out = args.out or (os.path.splitext(args.input)[0] + ".html")
    Path(out).write_text(PAGE.format(title=os.path.basename(args.input), meta=meta, body="".join(blocks)))
    print(f"Wrote {out}  ({len(shown)}/{total} pairs)")


if __name__ == "__main__":
    main()
