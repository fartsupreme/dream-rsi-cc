"""The map: a short, hard-capped brief of everything tried so far.

It replaces reading the full history at boot. It is regenerated from the tree,
so it cannot drift from it and cannot be lost to compaction.
"""
from __future__ import annotations

from .families import family_stats

_ORDER = {"open": 0, "plateau": 1, "dead": 2, "untried": 3}


def _one(s, cap=110) -> str:
    s = " ".join(str(s or "").split()).replace("|", "/")
    return s if len(s) <= cap else s[:cap - 1] + "…"


def render_map(tree, families: dict | None, goal: str = "", max_chars: int = 16000, plateau: int = 3,
               recent: int = 10) -> str:
    nodes = tree.nodes()
    fams = (families or {}).get("families") or []
    head = [f"# Search map — {len(nodes)} attempts, {len([f for f in fams if f['id'] != 'F00'])} families",
            "", f"Goal: {_one(goal, 1500) or '(not stated)'}", "",
            "Protocol: read this map, then run `drsi check \"<your proposal>\"` BEFORE building anything. "
            "A duplicate verdict means pick something else. Record every attempt, including failures.", ""]

    fam_rows: list[str] = []
    if fams:
        stats = family_stats(tree, families, plateau)
        stats.sort(key=lambda s: (_ORDER.get(s["status"], 9), -(int(s["last"]) if str(s["last"]).isdigit() else 0),
                                  s["id"]))
        fam_rows = [f"| {s['id']} | {_one(s['name'], 60)} | {s['n']} | {s['status']} | "
                    f"{(s['best'] + ' @#' + s['best_at'] + ', ' + str(s['since_best']) + ' since') if s['best'] else '-'} | "
                    f"{_one(s['killed_by_top'], 50) or '-'} | "
                    f"{('#' + s['last'] + ' (' + (s['last_created'] or '')[:10] + ')') if s['last'] else '-'} |"
                    for s in stats]
    fam_head = ["## Families (open → plateau → dead → untried)", "",
                "| id | family | n | status | best own-test outcome (first reached, attempts since) | stopped most by | last |",
                "|---|---|---|---|---|---|---|"]

    frontier = (families or {}).get("frontier") or []
    front = []
    if frontier:
        front = ["", "## Untried directions (classifier suggestions — unverified, check before use)", ""]
        front += [f"- {_one(d['direction'], 240)} — avoids {', '.join(d.get('avoids', [])) or '-'}" for d in frontier]

    rec = ["", f"## Last {recent} attempts", ""]
    for n in nodes[-recent:]:
        fp = n.get("fingerprint") or {}
        stop = f" (stopped by {_one(fp.get('killed_by'), 50)})" if fp.get("killed_by") else ""
        rec.append(f"- #{n['id']} [{fp.get('family', '?')}] {fp.get('outcome', '?')} — "
                   f"{_one(fp.get('mechanism') or n.get('proposal'), 140)}{stop}")

    if not fams:
        fam_head = ["## Families", "", "Not built yet: run `drsi families` to group the attempts.", ""]

    def assemble(rows, omitted):
        body = head + fam_head + rows
        if omitted:
            body.append(f"| … | +{omitted} more families (run `drsi families --list`) | | | | | |")
        return "\n".join(body + front + rec) + "\n"

    text = assemble(fam_rows, 0)
    keep = len(fam_rows)
    while len(text) > max_chars and keep > 0:
        keep -= 1
        text = assemble(fam_rows[:keep], len(fam_rows) - keep)
    if len(text) > max_chars:
        text = text[:max_chars - 20] + "\n…(map truncated)\n"
    return text
