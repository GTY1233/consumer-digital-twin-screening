"""Regenerate manuscript Figures 3-6 from the frozen numbers."""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RUNS, read_jsonl  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
NUM = json.loads(
    (pathlib.Path(__file__).resolve().parent.parent / "final_numbers.json").read_text("utf-8")
)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.linewidth": 1.0,
        "figure.dpi": 200,
        "savefig.dpi": 200,
    }
)

BLUE, LBLUE, GREY, ORANGE, GREEN = "#1f4e79", "#5b8db8", "#a6a6a6", "#c8892a", "#2e6b34"


def random_lift() -> float:
    """Expected decision lift of a uniformly random choice among creatives."""
    tasks = read_jsonl(RUNS / "ad_confirmatory_tasks.jsonl")
    lifts = []
    for t in tasks:
        imp, clk = t["meta"]["impressions"], t["meta"]["clicks"]
        ctrs = [c / i for c, i in zip(clk, imp)]
        mean_ctr = sum(clk) / sum(imp)
        lifts.append(sum(ctrs) / len(ctrs) / mean_ctr - 1)
    return sum(lifts) / len(lifts)


def box(ax, x, y, w, h, title, body, face, edge="#1f4e79", title_size=12):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.4, edgecolor=edge, facecolor=face,
        )
    )
    ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color="#1f3864")
    if body:
        ax.text(x + w / 2, y + h * 0.30, body, ha="center", va="center",
                fontsize=9.5, color="#333333")


def arrow(ax, x0, y0, x1, y1, color=BLUE, lw=1.8, head=0.16):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                        shrinkA=0, shrinkB=0, mutation_scale=18),
    )


def figure1() -> None:
    """Four-layer architecture: generous gaps so the arrows stay legible."""
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    layers = [
        ("Data layer", "Versioned creative inventory; outcome store kept separate"),
        ("Panel layer", "Personas from published segmentation and decision-style constructs"),
        ("Inference layer", "Each creative put to every panel member; fixed JSON contract"),
        ("Decision layer", "Expected ordering, dispersion across panel members, validated scope"),
    ]
    h, gap, x0, w = 0.145, 0.105, 0.035, 0.93
    for i, (title, body) in enumerate(layers):
        y = 0.985 - h - i * (h + gap)
        box(ax, x0, y, w, h, title, body, "#eef2f7", title_size=13)
        if i < len(layers) - 1:
            arrow(ax, 0.5, y - 0.012, 0.5, y - gap + 0.012)
    ax.text(0.5, 0.02, "Design rule: the outcome data never reaches the inference layer",
            ha="center", va="center", fontsize=10.5, style="italic", color="#b03030")
    fig.savefig(OUT / "figure1.png", bbox_inches="tight", facecolor="white", pad_inches=0.18)
    plt.close(fig)


def figure2() -> None:
    """Persona construction: three sources merge on a collector line, then descend."""
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, "Generational cohort", "Gen Z / millennial\nGen X / boomer"),
        (0.35, "Decision style", "Analytic versus experiential\n(Sproles & Kendall 1986)"),
        (0.68, "Dispositions", "Need for cognition\nCuriosity-gap response"),
    ]
    top, h, w = 0.985, 0.315, 0.30
    for x, title, body in boxes:
        box(ax, x, top - h, w, h, title, body, "#e7f0e9", title_size=12.5)
    collector = top - h - 0.075
    for x, _, _ in boxes:
        arrow(ax, x + w / 2, top - h - 0.012, x + w / 2, collector + 0.004)
    ax.plot([0.17, 0.83], [collector, collector], color=BLUE, lw=1.8, solid_capstyle="round")
    arrow(ax, 0.5, collector, 0.5, collector - 0.085)
    panel_h = 0.235
    panel_top = collector - 0.085
    box(ax, 0.135, panel_top - panel_h, 0.73, panel_h, "Eight-persona panel",
        "4 cohorts x 2 decision styles", "#fdf3e3", title_size=12.5)
    arrow(ax, 0.5, panel_top - panel_h - 0.012, 0.5, panel_top - panel_h - 0.10)
    box(ax, 0.135, 0.015, 0.73, 0.175, "Independent judgment per creative",
        "Averaged to an expected ordering", "#fdf3e3", title_size=12.5)
    fig.savefig(OUT / "figure2.png", bbox_inches="tight", facecolor="white", pad_inches=0.18)
    plt.close(fig)


def figure3() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.02, 0.60, 0.28, 0.32, "Exploratory split",
        "4,873 tests\nmethod development only", "#eef2f7")
    box(ax, 0.36, 0.60, 0.28, 0.32, "Confirmatory split",
        "22,743 tests\ntwo recorded executions", "#fdf3e3")
    box(ax, 0.70, 0.60, 0.28, 0.32, "Holdout split",
        "4,871 tests\nnot used", "#f0f0f0")
    ax.annotate("", xy=(0.355, 0.80), xytext=(0.305, 0.80),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=BLUE))
    ax.text(0.33, 0.94, "protocol frozen", ha="center", va="bottom",
            fontsize=9.5, color=BLUE)
    ax.annotate("", xy=(0.50, 0.32), xytext=(0.50, 0.585),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=BLUE))
    box(ax, 0.06, 0.02, 0.88, 0.30, "Two recorded executions of the frozen protocol",
        "pairwise accuracy, top-1 and decision lift\nagainst real outcomes", "#e8eef6",
        title_size=11.5)
    fig.savefig(OUT / "figure3.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure4() -> None:
    a, i = NUM["run2_advertising"]["AGG"], NUM["run2_advertising"]["IND"]
    labels = ["AGG\n(primary)", "IND", "Random\nchoice"]
    vals = [a["pairwise_accuracy"] * 100, i["pairwise_accuracy"] * 100, 50.0]
    errs = [
        [(a["pairwise_accuracy"] - a["ci95"][0]) * 100, (a["ci95"][1] - a["pairwise_accuracy"]) * 100],
        [(i["pairwise_accuracy"] - i["ci95"][0]) * 100, (i["ci95"][1] - i["pairwise_accuracy"]) * 100],
        [0, 0],
    ]
    errs = [[e[0] for e in errs], [e[1] for e in errs]]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    bars = ax.bar(labels, vals, color=[BLUE, LBLUE, GREY], width=0.6,
                  yerr=errs, capsize=5, error_kw=dict(ecolor="#333333", lw=1.2))
    ax.axhline(50, color="#b03030", ls="--", lw=1.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.65, f"{v:.2f}",
                ha="center", fontsize=10.5, fontweight="bold")
    ax.set_ylim(47.5, 60)
    ax.set_ylabel("Pairwise accuracy (%)")
    ax.set_yticks([48, 50, 52, 54, 56, 58, 60])
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figure4.png", facecolor="white")
    plt.close(fig)


def figure5() -> None:
    a = NUM["run2_advertising"]["AGG"]
    rnd = random_lift() * 100
    sysv = a["decision_lift"] * 100
    oracle = a["oracle_lift"] * 100
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    vals = [rnd, sysv, oracle]
    bars = ax.bar(["Random\nchoice", "System\nchoice", "Perfect\nranking"], vals,
                  color=[GREY, BLUE, GREEN], width=0.6)
    for bar, v in zip(bars, vals):
        off = 0.8 if v >= 0 else -2.2
        ax.text(bar.get_x() + bar.get_width() / 2, v + off, f"{v:+.2f}%",
                ha="center", fontsize=10.5, fontweight="bold")
    # Bracket the gap between the system's choice and a perfect ranking, drawn in
    # the empty column between the two bars so the head never lands inside a bar.
    bx = 1.50
    ax.plot([bx, bx], [sysv, oracle], color="#555555", lw=1.5)
    for y in (sysv, oracle):
        ax.plot([bx - 0.055, bx + 0.055], [y, y], color="#555555", lw=1.5)
    ax.text(bx - 0.10, (sysv + oracle) / 2,
            f"{a['share_of_oracle'] * 100:.1f}% of the gain\navailable to a perfect\nranking is captured",
            ha="right", va="center", fontsize=9.5, color="#555555")
    ax.axhline(0, color="#333333", lw=1.0)
    ax.set_ylabel("Click-through-rate lift over the test average (%)")
    ax.set_ylim(-3, 42)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figure5.png", facecolor="white")
    plt.close(fig)


def figure6() -> None:
    """Both domains, with the persona panel shown under the design's wrapper."""
    groups = ["Aggregate", "Individual", "Persona panel"]
    adv = [57.18, 55.02, 57.47]
    npv = [60.78, 58.45, 63.04]
    x = range(len(groups))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], adv, w, label="Advertising creative", color=BLUE)
    b2 = ax.bar([i + w / 2 for i in x], npv, w, label="New-product screening", color=ORANGE)
    goal = 61.54
    for bars in (b1, b2):
        for bar in bars:
            y = bar.get_height()
            label_y = y + 0.35
            # keep the value clear of the dotted goal line, which would otherwise
            # run through the label of any bar close to it
            if goal - 0.25 < label_y < goal + 0.85:
                label_y = goal + 0.85
            ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                    f"{y:.2f}", ha="center", fontsize=9.2)
    ax.axhline(goal, color="#7a7a7a", ls=":", lw=1.4)
    ax.text(-0.45, goal - 0.30,
            "funding goal alone", color="#5a5a5a", fontsize=9.2, ha="left", va="top")
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylim(52.0, 65.2)
    ax.set_ylabel("Pairwise accuracy (%)")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figure6.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    print("random-choice decision lift: %.4f%%" % (random_lift() * 100))
    print("figures written to", OUT)
