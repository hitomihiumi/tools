"""Comparing two runs over the same examples.

Two error rates and their confidence intervals are the wrong tool here. The
runs share a manifest, so every example is answered twice and the comparison is
paired: what matters is not "30% vs 26%" but how many individual cows changed
answer and in which direction. On 200 examples an unpaired reading needs a
~9 point gap to call anything; McNemar on the same data notices far less,
because the examples both runs get right or both get wrong carry no
information about the difference and are correctly ignored.
"""

from __future__ import annotations

import math

from cbvd import ACTIVITIES, POSTURES


def _failed(r):
    return bool(r.get("error") or r.get("parse_error")
                or r.get("posture") is None or r.get("activity") is None)


def _correct(r, axis):
    if _failed(r):
        return False
    if axis == "posture":
        return r["posture"] == r["gt_posture"]
    if axis == "activity":
        return r["activity"] == r["gt_activity"]
    return r["posture"] == r["gt_posture"] and r["activity"] == r["gt_activity"]


def binom_two_sided(b: int, c: int) -> float:
    """Exact McNemar. b and c are the discordant counts.

    The exact test rather than the chi-square approximation: the interesting
    case is a handful of flips, where the approximation is at its worst.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def compare(results_a, results_b, label_a="A", label_b="B"):
    by_a = {r["id"]: r for r in results_a}
    by_b = {r["id"]: r for r in results_b}
    shared = [i for i in by_a if i in by_b]
    if not shared:
        raise SystemExit("the two runs share no example ids - different manifests?")

    out = {
        "label_a": label_a,
        "label_b": label_b,
        "n_paired": len(shared),
        "n_only_a": len(by_a) - len(shared),
        "n_only_b": len(by_b) - len(shared),
        "axes": {},
        "per_class": {},
        "flips": {},
    }

    for axis in ("exact", "posture", "activity"):
        a_ok = sum(1 for i in shared if _correct(by_a[i], axis))
        b_ok = sum(1 for i in shared if _correct(by_b[i], axis))
        # b: A right, B wrong. c: A wrong, B right. Only these two move the number.
        only_a = [i for i in shared
                  if _correct(by_a[i], axis) and not _correct(by_b[i], axis)]
        only_b = [i for i in shared
                  if not _correct(by_a[i], axis) and _correct(by_b[i], axis)]
        n = len(shared)
        out["axes"][axis] = {
            "error_a": (n - a_ok) / n,
            "error_b": (n - b_ok) / n,
            "delta": (a_ok - b_ok) / n,  # positive = B is worse
            "fixed_by_b": len(only_b),
            "broken_by_b": len(only_a),
            "unchanged": n - len(only_a) - len(only_b),
            "p_value": binom_two_sided(len(only_a), len(only_b)),
        }
        out["flips"][axis] = {
            "fixed_by_b": only_b[:40],
            "broken_by_b": only_a[:40],
        }

    for axis, values, gt_key, pred_key in (
        ("posture", POSTURES, "gt_posture", "posture"),
        ("activity", ACTIVITIES, "gt_activity", "activity"),
    ):
        for value in values:
            support = [i for i in shared if by_a[i][gt_key] == value]
            if not support:
                continue
            hit_a = sum(1 for i in support
                        if not _failed(by_a[i]) and by_a[i][pred_key] == value)
            hit_b = sum(1 for i in support
                        if not _failed(by_b[i]) and by_b[i][pred_key] == value)
            said_a = sum(1 for i in shared
                         if not _failed(by_a[i]) and by_a[i][pred_key] == value)
            said_b = sum(1 for i in shared
                         if not _failed(by_b[i]) and by_b[i][pred_key] == value)
            out["per_class"][f"{axis}:{value}"] = {
                "support": len(support),
                "recall_a": hit_a / len(support),
                "recall_b": hit_b / len(support),
                "predicted_a": said_a,
                "predicted_b": said_b,
            }

    return out


def _pct(x):
    return f"{100 * x:.1f}%"


def render(cmp_result, meta_a=None, meta_b=None) -> str:
    a, b = cmp_result["label_a"], cmp_result["label_b"]
    lines = []
    add = lines.append

    add(f"# {a} vs {b}")
    add("")
    add(f"{cmp_result['n_paired']} examples answered by both runs. Paired "
        f"comparison: the same cow, the same annotation, two settings.")
    if cmp_result["n_only_a"] or cmp_result["n_only_b"]:
        add("")
        add(f"Unpaired and excluded: {cmp_result['n_only_a']} only in {a}, "
            f"{cmp_result['n_only_b']} only in {b}.")
    add("")

    for meta, label in ((meta_a, a), (meta_b, b)):
        if not meta:
            continue
        frames = meta.get("frames") or 1
        temporal = ("single keyframe" if frames <= 1
                    else f"{frames} frames over {meta.get('span')}s")
        add(f"- **{label}** — image mode `{meta.get('render_mode')}`, max width "
            f"{meta.get('max_width')}px, {temporal}, prompt "
            f"`{meta.get('prompt_sha')}`, run {meta.get('run_date')}")
    add("")

    add("## Paired result")
    add("")
    add(f"| Axis | {a} error | {b} error | Fixed by {b} | Broken by {b} | p (McNemar) |")
    add("|---|---|---|---|---|---|")
    for axis in ("exact", "posture", "activity"):
        m = cmp_result["axes"][axis]
        add(f"| {axis} | {_pct(m['error_a'])} | {_pct(m['error_b'])} | "
            f"{m['fixed_by_b']} | {m['broken_by_b']} | {m['p_value']:.3f} |")
    add("")
    add("`Fixed` and `broken` are the only examples that carry information about "
        "the difference; everything both runs agree on is excluded from the test. "
        "A p-value above 0.05 means the two settings are not distinguishable on "
        "this many examples — which is a finding, not a failure.")
    add("")

    add("## Per class")
    add("")
    add(f"| Class | Support | Recall {a} | Recall {b} | Predicted {a} | Predicted {b} |")
    add("|---|---|---|---|---|---|")
    for key, m in cmp_result["per_class"].items():
        add(f"| `{key}` | {m['support']} | {_pct(m['recall_a'])} | "
            f"{_pct(m['recall_b'])} | {m['predicted_a']} | {m['predicted_b']} |")
    add("")

    return "\n".join(lines) + "\n"
