#!/usr/bin/env python3
"""One cow under two names - the pairs the night was unsure of, put to a person.

Each camera's gallery names a cow by the side it sees; the night (pipeline.merge_by_place, [mct] link_rule "llr")
sums where her names' tracks stood over the days. Sure pairs are merged; unsure ones (cow_review_llr) wait in the
store's merge_reviews for a person - three zones, as in a city's camera network: sure, ask, nothing.

    python herd/herd.py merges page  --config barn.toml --out merges.html     # both names' faces, per camera
    python herd/herd.py merges apply --config barn.toml --decisions decisions.json
    python herd/herd.py merges list  --config barn.toml

  page   every pending pair: the two names' faces (gallery/faces/, a confirmed burst's crop per camera every
         [identity] face_every_s), how many bursts each has per camera, the evidence and its margin over the
         next cow; S same cow, D different, Enter next; "Save decisions.json". The night's sure merges are
         listed below it, to look over.
  apply  "same": the old name aliased into the other at once (the reports follow; store.unalias_cow undoes it),
         the gallery merges the two the next night; "different": never proposed again.
  list   the pairs and their state.

--store / --gallery instead of --config: a trial's (trial/pass2/herd.sqlite, trial/pass2/gallery).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def paths(args):
    """(store path, gallery folder) from --config, or as given."""
    if args.store:
        return args.store, args.gallery or os.path.join(os.path.dirname(os.path.abspath(args.store)), "gallery")
    from common import load_config
    cfg = load_config(args.config)
    store = cfg["store"]["path"]
    return store, args.gallery or os.path.join(os.path.dirname(os.path.abspath(store)), "gallery")


def faces(gallery, cow, aliases, page_dir):
    """{camera: [face paths relative to the page]} of a name and every name aliased into it, newest first."""
    names = {cow} | {old for old, c in aliases.items() if c == cow}
    out = collections.defaultdict(list)
    for n in sorted(names):
        d = os.path.join(gallery, "faces", n)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".jpg") and "_" in f:
                cam, ts = f[:-4].rsplit("_", 1)
                out[cam].append((int(ts) if ts.isdigit() else 0,
                                 os.path.relpath(os.path.join(d, f), page_dir).replace(os.sep, "/")))
    return {cam: [p for _, p in sorted(v, reverse=True)] for cam, v in sorted(out.items())}


def bursts_by_camera(store):
    """{cow: {camera: confirmed bursts}}, aliases followed."""
    aliases = store.aliases()
    out = collections.defaultdict(collections.Counter)
    for cam, cow, n in store.query("SELECT cam, cow, COUNT(*) FROM bursts WHERE state='confirmed' GROUP BY cam, cow"):
        out[aliases.get(cow, cow)][cam] += n
    return out


def sure_merges(store, days=7.0):
    """The nights' own merges of the last days: [{ts, old, into}]."""
    out = []
    for ts, detail in store.query("SELECT ts, detail FROM events WHERE kind = 'merged-by-place' AND ts >= ? "
                                  "ORDER BY ts DESC", (time.time() - days * 86400,)):
        d = json.loads(detail)
        for old, into in (d.get("merged", []) if isinstance(d, dict) else d):
            out.append({"ts": ts, "old": old, "into": into})
    return out


def cmd_page(args):
    from store import Store
    store_path, gallery = paths(args)
    store = Store(store_path)
    page_dir = os.path.dirname(os.path.abspath(args.out))
    aliases = store.aliases()
    per_cam = bursts_by_camera(store)
    show = lambda c: {"name": c, "faces": faces(gallery, c, aliases, page_dir), "bursts": dict(per_cam.get(c, {}))}
    pending = [dict(r, a=show(r["old"]), b=show(r["into"])) for r in store.reviews("pending")]
    done = [r for r in store.reviews() if r["state"] != "pending"]
    sure = [dict(m, a=show(m["old"]), b=show(m["into"])) for m in sure_merges(store, args.days)]
    data = {"pending": pending, "decided": done, "sure": sure, "store": os.path.abspath(store_path)}
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(PAGE.replace("/*DATA*/null", json.dumps(data)))
    print(f"[merges] {len(pending)} pairs to look at, {len(done)} decided, {len(sure)} merged by the nights of the "
          f"last {args.days:g} days -> {args.out}  (open it in a browser; the faces are read from {gallery}/faces)")


def cmd_apply(args):
    from store import Store
    store_path, _ = paths(args)
    store = Store(store_path)
    rows = json.load(open(args.decisions, encoding="utf-8"))
    rows = rows.get("decisions", rows) if isinstance(rows, dict) else rows
    now = time.time()
    said = collections.Counter()
    for r in rows:
        old, into, ans = r["old"], r["into"], r.get("answer")
        if ans == "same":
            store.decide_review(old, into, "same", now)
            store.alias_cow(old, into, now)
        elif ans == "different":
            store.decide_review(old, into, "different", now)
        else:
            continue
        said[ans] += 1
    store.event(now, "merge-reviews", json.dumps({k: v for k, v in said.items()}))
    print(f"[merges] {said['same']} pairs one cow (aliased now; the gallery merges them the next night), "
          f"{said['different']} two cows (never proposed again)")


def cmd_list(args):
    from store import Store
    store = Store(paths(args)[0])
    rows = store.reviews()
    if not rows:
        print("[merges] nothing to review ([mct] link_rule = \"llr\" fills it at night)")
    for r in rows:
        ev = "" if r["llr"] is None else f"  evidence {r['llr']:.0f}, margin {r['margin']:.0f}"
        print(f"{r['state']:9s} {r['old']} -> {r['into']}{ev}")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>One cow, two names?</title>
<style>
:root{--bg:#fafaf7;--card:#fff;--fg:#1d1d1b;--mute:#6b6b66;--line:#e2e1da;--accent:#2f6f4f;--warn:#a33;--same:#e8f3ec;--diff:#f6e7e7;--sel:#c9a227}
@media (prefers-color-scheme: dark){:root{--bg:#171816;--card:#20221f;--fg:#ecebe6;--mute:#9a9a93;--line:#34362f;--accent:#7fc29c;--warn:#e88;--same:#1f3127;--diff:#3a2424;--sel:#d9b84a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px}
h1{font-size:17px;margin:0 0 6px}h2{font-size:15px;margin:22px 0 6px}.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
button{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer}
button.main{background:var(--accent);color:var(--bg);border-color:var(--accent)}
.help,.mute{color:var(--mute)}.help{font-size:13px;margin:4px 0}main{padding:8px 16px 40px}
.pair{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;margin:8px 0}
.pair.same{background:var(--same)}.pair.different{background:var(--diff)}.pair.cur{outline:2px solid var(--sel)}
.sides{display:grid;grid-template-columns:1fr 1fr;gap:12px}.side b{word-break:break-all}
.cam{margin:6px 0 2px;font-size:12px;color:var(--mute)}
.faces{display:flex;flex-wrap:wrap;gap:3px}.faces img{width:96px;height:96px;object-fit:cover;border-radius:4px;background:var(--line)}
.none{font-size:12px;color:var(--mute);padding:8px 0}.btns{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap}
@media (max-width:640px){header{position:static}.sides{grid-template-columns:1fr}.faces img{width:72px;height:72px}}
</style></head><body>
<header>
<h1>One cow, two names?</h1>
<p class="help">Each camera names a cow by the side it sees. These pairs stood together often, but not surely enough
to merge alone. Look at both names' faces: <b>S</b> the same cow, <b>D</b> two cows, <b>Enter</b> the next pair.
Answers are kept in this browser; <b>Save decisions.json</b>, then <code>herd.py merges apply --decisions decisions.json</code>.</p>
<div class="bar"><button class="main" id="save">Save decisions.json</button><span class="mute" id="stat"></span></div>
</header>
<main><div id="list"></div><h2 id="sureh"></h2><div id="sure"></div><h2 id="doneh"></h2><div id="done" class="mute"></div></main>
<script>
const DATA = /*DATA*/null;
const KEY = "herd-merges:" + DATA.store;
let ans = {};
try { ans = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { ans = {}; }
const keep = () => { try { localStorage.setItem(KEY, JSON.stringify(ans)); } catch (e) {} };
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const id = p => p.old + "→" + p.into;
let cur = 0;
function side(s) {
  const cams = Object.keys(s.faces);
  const n = Object.entries(s.bursts).map(([c, k]) => `${esc(c)} ${k}`).join(", ") || "no bursts";
  return `<div class="side"><b>${esc(s.name)}</b> <span class="mute">(${n})</span>` +
    (cams.length ? cams.map(c => `<div class="cam">${esc(c)}</div><div class="faces">${s.faces[c].map(
      f => `<img loading="lazy" src="${esc(f)}" alt="">`).join("")}</div>`).join("")
     : `<div class="none">no faces kept yet</div>`) + `</div>`;
}
function draw() {
  const L = document.getElementById("list");
  L.innerHTML = DATA.pending.length ? DATA.pending.map((p, k) => `<div class="pair ${ans[id(p)] || ""} ${k === cur ? "cur" : ""}" data-k="${k}">
    <div class="sides">${side(p.a)}${side(p.b)}</div>
    <div class="mute">evidence ${p.llr == null ? "-" : p.llr.toFixed(0)}, ahead of the next cow by ${p.margin == null ? "-" : p.margin.toFixed(0)}</div>
    <div class="btns"><button data-a="same">Same cow (S)</button><button data-a="different">Two cows (D)</button><button data-a="">Not sure</button></div></div>`).join("")
    : `<p class="mute">Nothing to look at: no pair the nights were unsure of.</p>`;
  const n = Object.values(ans).filter(Boolean).length;
  document.getElementById("stat").textContent = `${n} of ${DATA.pending.length} answered`;
  document.getElementById("sureh").textContent = DATA.sure.length ? `Merged by the nights themselves (${DATA.sure.length})` : "";
  document.getElementById("sure").innerHTML = DATA.sure.map(p => `<div class="pair"><div class="mute">${esc(new Date(p.ts * 1000).toLocaleString())}: ${esc(p.old)} into ${esc(p.into)}</div><div class="sides">${side(p.a)}${side(p.b)}</div></div>`).join("");
  document.getElementById("doneh").textContent = DATA.decided.length ? `Decided before (${DATA.decided.length})` : "";
  document.getElementById("done").innerHTML = DATA.decided.map(r => `<div>${esc(r.state)}: ${esc(r.old)} / ${esc(r.into)}</div>`).join("");
}
function answer(k, a) {
  const p = DATA.pending[k]; if (!p) return;
  ans[id(p)] = a; keep(); cur = Math.min(k + 1, DATA.pending.length - 1); draw();
  const el = document.querySelector(`.pair[data-k="${cur}"]`); if (el) el.scrollIntoView({block: "nearest"});
}
document.getElementById("list").addEventListener("click", e => {
  const b = e.target.closest("button"), card = e.target.closest(".pair"); if (!card) return;
  const k = +card.dataset.k;
  if (b) answer(k, b.dataset.a); else { cur = k; draw(); }
});
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "s" || e.key === "S") answer(cur, "same");
  else if (e.key === "d" || e.key === "D") answer(cur, "different");
  else if (e.key === "Enter") { cur = Math.min(cur + 1, DATA.pending.length - 1); draw(); }
});
document.getElementById("save").onclick = () => {
  const out = DATA.pending.filter(p => ans[id(p)]).map(p => ({old: p.old, into: p.into, answer: ans[id(p)]}));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify({decisions: out}, null, 1)], {type: "application/json"}));
  a.download = "decisions.json"; a.click();
};
draw();
</script></body></html>
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("page", "apply", "list"):
        s = sub.add_parser(name)
        s.add_argument("--config", help="barn.toml: its store and gallery")
        s.add_argument("--store", help="instead of --config: a store file (a trial's pass2/herd.sqlite)")
        s.add_argument("--gallery", help="the gallery folder (default: gallery/ beside the store)")
        if name == "page":
            s.add_argument("--out", default="merges.html")
            s.add_argument("--days", type=float, default=7.0, help="the nights' own merges of this many days, listed")
        if name == "apply":
            s.add_argument("--decisions", required=True, help="the page's decisions.json")
    args = p.parse_args(argv)
    if not args.config and not args.store:
        p.error("--config or --store")
    {"page": cmd_page, "apply": cmd_apply, "list": cmd_list}[args.cmd](args)


if __name__ == "__main__":
    main()
