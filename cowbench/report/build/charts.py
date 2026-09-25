import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = "C:/DEV/tools/cowbench"
sys.path.insert(0, BENCH)
import tracks  # noqa: E402

F = json.load(open(os.path.join(HERE, "facts.json"), encoding="utf-8"))

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
S1 = "#2a78d6"   # categorical slot 1
S2 = "#eb6834"   # categorical slot 2

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
})

UA = {"posture:standing": "стоїть", "posture:lying": "лежить",
      "activity:feeding": "їсть", "activity:none": "нічого з переліченого",
      "activity:drinking": "п'є", "activity:ruminating": "жуйка"}
ORDER = ["posture:standing", "posture:lying", "activity:feeding",
         "activity:none", "activity:drinking", "activity:ruminating"]


def recall_chart():
    start = F["val_1280"]["per_class"]
    end = F["val_1920_vote"]["per_class"]
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=200)
    h = 0.36
    ys = list(range(len(ORDER)))[::-1]
    for y, k in zip(ys, ORDER):
        a, b = 100 * start[k]["recall"], 100 * end[k]["recall"]
        ax.barh(y + h / 2 + 0.01, a, height=h, color=S1, zorder=2)
        ax.barh(y - h / 2 - 0.01, b, height=h, color=S2, zorder=2)
        for val, yy in ((a, y + h / 2 + 0.01), (b, y - h / 2 - 0.01)):
            ax.text(val + 1.2, yy, (f"{val:.1f}%").replace(".", ","), va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{UA[k]}  (n={start[k]['support']})" for k in ORDER], color=INK)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("частка правильно названих корів класу (recall)")
    ax.legend(handles=[Patch(color=S1, label="на початку: 1280 px, кожен кадр окремо"),
                       Patch(color=S2, label="у підсумку: 1920 px + голосування по треку")],
              loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "chart_recall.png"), facecolor=SURFACE)
    plt.close(fig)


def size_chart():
    path = os.path.join(BENCH, "runs", "2026-09-25_val-full_w1920_f1", "results.jsonl")
    res = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    res, _ = tracks.vote(res)
    for r in res:
        x1, y1, x2, y2 = r["bbox"]
        r["_a"] = (x2 - x1) * (y2 - y1)
    res.sort(key=lambda r: (r["_a"], r["id"]))
    q = len(res) // 4
    bands = [res[i * q:(i + 1) * q if i < 3 else len(res)] for i in range(4)]

    def wrong(r):
        return r["posture"] != r["gt_posture"] or r["activity"] != r["gt_activity"]

    allv, norum, meta = [], [], []
    for b in bands:
        nr = [r for r in b if r["gt_activity"] != "ruminating"]
        allv.append(100 * sum(map(wrong, b)) / len(b))
        norum.append(100 * sum(map(wrong, nr)) / len(nr))
        lying = sum(1 for r in b if r["gt_posture"] == "lying")
        meta.append({"min": 100 * b[0]["_a"], "max": 100 * b[-1]["_a"], "n": len(b),
                     "rum": len(b) - len(nr), "lying": lying})
    with open(os.path.join(HERE, "size_split.json"), "w", encoding="utf-8") as fh:
        json.dump({"all": allv, "no_rum": norum, "meta": meta}, fh, indent=1)

    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
    w = 0.36
    for i in range(4):
        ax.bar(i - w / 2 - 0.01, allv[i], width=w, color=S1, zorder=2)
        ax.bar(i + w / 2 + 0.01, norum[i], width=w, color=S2, zorder=2)
        ax.text(i - w / 2 - 0.01, allv[i] + 1, (f"{allv[i]:.1f}%").replace(".", ","), ha="center", fontsize=8.5, color=INK)
        ax.text(i + w / 2 + 0.01, norum[i] + 1, (f"{norum[i]:.1f}%").replace(".", ","), ha="center", fontsize=8.5, color=INK)
    names = ["найменші", "менші", "більші", "найбільші"]
    ax.set_xticks(range(4))
    ax.set_xticklabels([n + "\n" + (f"{m['min']:.2f}–{m['max']:.2f}% кадру").replace(".", ",")
                        for n, m in zip(names, meta)], fontsize=8.5, color=INK)
    ax.set_ylim(0, max(allv) * 1.3)
    ax.set_ylabel("помилка (обидві відповіді)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.legend(handles=[Patch(color=S1, label="усі корови"),
                       Patch(color=S2, label="без корів, розмічених як «жуйка»")],
              loc="upper right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "chart_size.png"), facecolor=SURFACE)
    plt.close(fig)


recall_chart()
size_chart()
print("charts written")
