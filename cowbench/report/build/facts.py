"""Collect every number the report quotes, straight from runs/.

Nothing in the document is typed by hand: if a figure appears in the report it
was computed here from results.jsonl / metrics.json and can be recomputed.
"""
import collections
import json
import os
import statistics
import sys

BENCH = r"C:\DEV\tools\cowbench"
ROOT = r"C:\Users\Work\Downloads\archive"
sys.path.insert(0, BENCH)
os.chdir(BENCH)

import cbvd  # noqa: E402
import compare  # noqa: E402
import scoring  # noqa: E402
import tracks  # noqa: E402

RUNS = {
    "clip371_f1": "runs/2026-09-25_clip371_w1280_f1",
    "clip371_f5": "runs/2026-09-25_clip371_w1280_f5",
    "val_1280": "runs/2026-09-25_val-full_w1280_f1",
    "val_1920": "runs/2026-09-25_val-full_w1920_f1",
    "s300_1280": "runs/2026-09-25_val300_w1280",
    "s300_1920": "runs/2026-09-25_val300_w1920",
    "s300_both": "runs/2026-09-25_val300_both",
    "s300_jaw8": "runs/2026-09-25_val300_jaw8",
    "s300_jaw20": "runs/2026-09-25_val300_jaw20",
}


def load(key):
    with open(os.path.join(RUNS[key], "results.jsonl"), encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def meta(key):
    with open(os.path.join(RUNS[key], "run_meta.json"), encoding="utf-8") as fh:
        return json.load(fh)


def ok(r):
    return r["posture"] == r["gt_posture"] and r["activity"] == r["gt_activity"]


def summary(results):
    m = scoring.score(results)
    return {
        "n": m["n_examples"],
        "keyframes": m["n_keyframes"],
        "clips": m["n_clips"],
        "failed": m["n_failed"],
        "exact": m["exact_match"]["error_rate"],
        "exact_ci": m["exact_match"]["ci95"],
        "posture": m["posture"]["error_rate"],
        "posture_ci": m["posture"]["ci95"],
        "activity": m["activity"]["error_rate"],
        "activity_ci": m["activity"]["ci95"],
        "base_posture": m["baseline"]["posture"]["error_rate"],
        "base_posture_label": m["baseline"]["posture"]["label"],
        "base_activity": m["baseline"]["activity"]["error_rate"],
        "base_activity_label": m["baseline"]["activity"]["label"],
        "per_class": m["per_class"],
        "predicted": m["predicted"],
        "support": m["support"],
        "by_size": m["by_size"],
        "by_clip": m["by_clip"],
        "confusion": m["confusion"],
    }


def paired(a, b):
    c = compare.compare(a, b)
    return {k: {"fixed": v["fixed_by_b"], "broken": v["broken_by_b"],
                "p": v["p_value"], "err_a": v["error_a"], "err_b": v["error_b"]}
            for k, v in c["axes"].items()} | {"per_class": c["per_class"]}


def rumination(results):
    rum = [r for r in results if r["gt_activity"] == "ruminating"]
    said = [r for r in results if r["activity"] == "ruminating"]
    return {"annotated": len(rum), "said": len(said),
            "correct": sum(1 for r in said if r["gt_activity"] == "ruminating"),
            "answers_on_rum": dict(collections.Counter(r["activity"] for r in rum))}


F = {}
R = {k: load(k) for k in RUNS}
for k in RUNS:
    F[k] = summary(R[k])
    F[k]["rum"] = rumination(R[k])
    md = meta(k)
    F[k]["meta"] = {x: md.get(x) for x in (
        "run_date", "model", "model_repo", "vllm_version", "render_mode", "frames",
        "span", "max_width", "min_width", "temperature", "max_tokens", "prompt_sha")}
    toks = [r["usage"]["prompt_tokens"] for r in R[k] if r.get("usage")]
    F[k]["prompt_tokens_median"] = statistics.median(toks) if toks else None

# voting on the final full-val run
voted, info = tracks.vote(R["val_1920"])
F["val_1920_vote"] = summary(voted)
F["val_1920_vote"]["rum"] = rumination(voted)
F["vote_info"] = info
voted1280, info1280 = tracks.vote(R["val_1280"])
F["val_1280_vote"] = summary(voted1280)

# paired comparisons quoted in the report
F["pair"] = {
    "clip371_f1_vs_f5": paired(R["clip371_f1"], R["clip371_f5"]),
    "s300_1280_vs_1920": paired(R["s300_1280"], R["s300_1920"]),
    "s300_1920_vs_both": paired(R["s300_1920"], R["s300_both"]),
    "s300_1920_vs_jaw8": paired(R["s300_1920"], R["s300_jaw8"]),
    "s300_jaw8_vs_jaw20": paired(R["s300_jaw8"], R["s300_jaw20"]),
    "s300_1920_vs_jaw20": paired(R["s300_1920"], R["s300_jaw20"]),
    "val_1280_vs_1920": paired(R["val_1280"], R["val_1920"]),
    "val_1920_vs_vote": paired(R["val_1920"], voted),
    "val_1280_vs_final": paired(R["val_1280"], voted),
    "val_1280_vs_1280vote": paired(R["val_1280"], voted1280),
}

# where the resolution gain sits: size quartiles on the 300-cow pair
a = {r["id"]: r for r in R["s300_1280"]}
b = {r["id"]: r for r in R["s300_1920"]}
ids = sorted(set(a) & set(b))
area = {i: (a[i]["bbox"][2] - a[i]["bbox"][0]) * (a[i]["bbox"][3] - a[i]["bbox"][1]) for i in ids}
qs = statistics.quantiles([area[i] for i in ids], n=4)
rows = []
for q in range(4):
    lo = -1 if q == 0 else qs[q - 1]
    hi = qs[q] if q < 3 else 9
    s = [i for i in ids if lo < area[i] <= hi]
    rows.append({"q": q + 1, "n": len(s),
                 "area_min": 100 * min(area[i] for i in s), "area_max": 100 * max(area[i] for i in s),
                 "err_a": sum(1 for i in s if not ok(a[i])) / len(s),
                 "err_b": sum(1 for i in s if not ok(b[i])) / len(s),
                 "fixed": sum(1 for i in s if not ok(a[i]) and ok(b[i])),
                 "broken": sum(1 for i in s if ok(a[i]) and not ok(b[i]))})
F["res_by_size"] = rows

# error budget of the final configuration
n = len(voted)
budget = collections.Counter()
for r in voted:
    if ok(r):
        continue
    p = r["posture"] != r["gt_posture"]
    act = r["activity"] != r["gt_activity"]
    if act and r["gt_activity"] == "ruminating":
        budget["missed_rum"] += 1
    elif p and act:
        budget["both"] += 1
    elif act:
        budget[f"act:{r['gt_activity']}->{r['activity']}"] += 1
    else:
        budget[f"pos:{r['gt_posture']}->{r['posture']}"] += 1
F["budget"] = {"n": n, "wrong": sum(budget.values()), "items": budget.most_common()}
nonrum = [r for r in voted if r["gt_activity"] != "ruminating"]
F["final_err_without_rum"] = sum(1 for r in nonrum if not ok(r)) / len(nonrum)
nonrum0 = [r for r in R["val_1280"] if r["gt_activity"] != "ruminating"]
F["start_err_without_rum"] = sum(1 for r in nonrum0 if not ok(r)) / len(nonrum0)

# dataset facts
tr_all = cbvd.load_boxes(os.path.join(ROOT, "annotations", "ava_train_v2.1.csv"))
va_all = cbvd.load_boxes(os.path.join(ROOT, "annotations", "ava_val_v2.1.csv"))
tr, tr_rej = cbvd.partition(tr_all)
va, va_rej = cbvd.partition(va_all)
test_rows = sum(1 for l in open(os.path.join(ROOT, "annotations", "ava_test_v2.1.csv")) if l.strip())
val_rows = sum(1 for l in open(os.path.join(ROOT, "annotations", "ava_val_v2.1.csv")) if l.strip())
test_boxes = len(cbvd.load_boxes(os.path.join(ROOT, "annotations", "ava_test_v2.1.csv")))
ov = {b.video_id for b in tr} & {b.video_id for b in va}
F["data"] = {
    "train_boxes_raw": len(tr_all), "val_boxes_raw": len(va_all),
    "train_usable": len(tr), "val_usable": len(va),
    "train_rejected": len(tr_rej), "val_rejected": len(va_rej),
    "train_keyframes": len({(b.video_id, b.timestamp) for b in tr}),
    "val_keyframes": len({(b.video_id, b.timestamp) for b in va}),
    "train_clips": len({b.video_id for b in tr}), "val_clips": len({b.video_id for b in va}),
    "test_rows": test_rows, "val_rows": val_rows, "test_boxes": test_boxes,
    "overlap_clips": sorted(ov, key=int),
    "overlap_val_boxes": sum(1 for b in va if b.video_id in ov),
    "overlap_keyframes": len({(b.video_id, b.timestamp) for b in tr} & {(b.video_id, b.timestamp) for b in va}),
    "val_rejected_reason": [why for _, why in va_rej],
}
# rumination is a clip-level property
def p_rum(bs):
    ly = [b for b in bs if b.posture == "lying"]
    return sum(1 for b in ly if b.activity == "ruminating"), len(ly)
F["rum_prior"] = {"train": p_rum(tr), "val": p_rum(va)}
per = collections.defaultdict(lambda: [0, 0])
for b in tr + va:
    if b.posture == "lying":
        per[b.video_id][0] += b.activity == "ruminating"
        per[b.video_id][1] += 1
big = [(r, t) for r, t in per.values() if t >= 10]
F["rum_by_clip"] = {"clips": len(big), "all": sum(1 for r, t in big if r == t),
                    "none": sum(1 for r, t in big if r == 0),
                    "median": statistics.median(r / t for r, t in big)}
ly_val = [r for r in R["val_1280"] if r["gt_posture"] == "lying"]
F["val_lying"] = {"n": len(ly_val),
                  "rum": sum(1 for r in ly_val if r["gt_activity"] == "ruminating"),
                  "rum_total": sum(1 for r in R["val_1280"] if r["gt_activity"] == "ruminating"),
                  "rum_lying": sum(1 for r in R["val_1280"] if r["gt_activity"] == "ruminating" and r["gt_posture"] == "lying")}
# track stability of labels (ground truth only)
gt_rows = [dict(r, posture=r["gt_posture"], activity=r["gt_activity"]) for r in R["val_1280"]]
tr_list = tracks.build(gt_rows)
F["track_gt"] = {"tracks": len(tr_list),
                 "posture_const": sum(1 for t in tr_list if len({r["posture"] for r in t}) == 1),
                 "activity_const": sum(1 for t in tr_list if len({r["activity"] for r in t}) == 1)}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "facts.json")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(F, fh, indent=1, ensure_ascii=False, default=str)
print("facts ->", out)
