#!/usr/bin/env python3
"""
analyze_metrics.py — read scenarios.csv + metrics.csv, build plots & summary.

Reads:
  experiments/scenarios.csv  — one row per scenario (n_initial, n_picked, ...)
  experiments/metrics.csv    — one row per individual pick attempt

Writes:
  experiments/plots/*.png
  experiments/summary.txt
"""
import csv
import os
import sys
from collections import defaultdict, Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
SCENARIOS_CSV = os.path.join(ROOT, "scenarios.csv")
METRICS_CSV   = os.path.join(ROOT, "metrics.csv")
PLOTS = os.path.join(ROOT, "plots")
os.makedirs(PLOTS, exist_ok=True)


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def split_yolo_baseline(rows):
    yolo = [r for r in rows if int(r["use_yolo"]) == 1]
    base = [r for r in rows if int(r["use_yolo"]) == 0]
    return yolo, base


# ── Scenario-level metrics ──────────────────────────────────────────────────

def scenario_metrics(rows):
    if not rows:
        return None
    n = len(rows)
    pick_total = sum(int(r["n_picked"]) for r in rows)
    init_total = sum(int(r["n_initial"]) for r in rows)
    full_succ  = sum(int(r["full_success"]) for r in rows)
    times      = [float(r["scenario_time_sec"]) for r in rows]
    return {
        "n_scenarios": n,
        "pick_rate": 100.0 * pick_total / init_total if init_total else 0.0,
        "full_success_rate": 100.0 * full_succ / n,
        "avg_picked": pick_total / n,
        "avg_initial": init_total / n,
        "avg_time": sum(times) / n,
        "median_time": sorted(times)[n // 2],
    }


# ── Plot 1: pick rate (YOLO vs baseline) bar ────────────────────────────────

def plot_pick_rate_bar(yolo_m, base_m):
    labels, values, colors = [], [], []
    if yolo_m:
        labels.append("С YOLO")
        values.append(yolo_m["pick_rate"])
        colors.append("#2b7bba")
    if base_m:
        labels.append("Baseline")
        values.append(base_m["pick_rate"])
        colors.append("#b8860b")
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, values, color=colors)
    ax.axhline(80, color="red", linestyle="--", linewidth=1, label="Цель 80%")
    for i, v in enumerate(values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Pick rate, %")
    ax.set_ylim(0, 110)
    ax.set_title("Доля успешных захватов (по всем сценариям)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "pick_rate.png"), dpi=150)
    plt.close()


# ── Plot 2: full-scenario success rate bar ──────────────────────────────────

def plot_full_success(yolo_m, base_m):
    labels, values, colors = [], [], []
    if yolo_m:
        labels.append("С YOLO")
        values.append(yolo_m["full_success_rate"])
        colors.append("#2b7bba")
    if base_m:
        labels.append("Baseline")
        values.append(base_m["full_success_rate"])
        colors.append("#b8860b")
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, values, color=colors)
    for i, v in enumerate(values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("% сценариев")
    ax.set_ylim(0, 110)
    ax.set_title("Доля полностью успешных сценариев")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "full_success.png"), dpi=150)
    plt.close()


# ── Plot 3: scenario time histogram ─────────────────────────────────────────

def plot_scenario_time(rows, name):
    if not rows:
        return
    times = [float(r["scenario_time_sec"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(times, bins=12, color="#2b7bba", edgecolor="black", alpha=0.8)
    avg = sum(times) / len(times)
    ax.axvline(avg, color="red", linestyle="--", label=f"Среднее: {avg:.1f}с")
    ax.set_xlabel("Время сценария, с")
    ax.set_ylabel("Количество сценариев")
    ax.set_title(f"Распределение времени сценария ({name})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f"scenario_time_{name}.png"), dpi=150)
    plt.close()


# ── Plot 4: pick attempts by size class (from metrics.csv) ──────────────────

def plot_attempts_by_size(metrics_rows):
    if not metrics_rows:
        return
    yolo_rows = [r for r in metrics_rows if int(r["use_yolo"]) == 1]
    base_rows = [r for r in metrics_rows if int(r["use_yolo"]) == 0]

    sizes = ["small", "medium", "large"]
    def rate_by_size(rows, sz):
        bucket = [r for r in rows if r["box_size_class"] == sz]
        if not bucket:
            return 0.0
        return 100.0 * sum(int(r["success"]) for r in bucket) / len(bucket)

    yolo_rates = [rate_by_size(yolo_rows, s) for s in sizes]
    base_rates = [rate_by_size(base_rows, s) for s in sizes]

    x = range(len(sizes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    if any(yolo_rates):
        ax.bar([i - w/2 for i in x], yolo_rates, w, label="С YOLO", color="#2b7bba")
    if any(base_rates):
        ax.bar([i + w/2 for i in x], base_rates, w, label="Baseline", color="#b8860b")
    ax.axhline(80, color="red", linestyle="--", linewidth=1, label="Цель 80%")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Малые", "Средние", "Большие"])
    ax.set_ylabel("Success rate захвата, %")
    ax.set_ylim(0, 110)
    ax.set_title("Success rate по размерам коробок (per-attempt)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "attempts_by_size.png"), dpi=150)
    plt.close()


# ── Plot 5: failure step pie ────────────────────────────────────────────────

def plot_failures(metrics_rows, name):
    rows = [r for r in metrics_rows
            if int(r["use_yolo"]) == (1 if name == "yolo" else 0)]
    fails = [r["fail_step"] for r in rows
             if int(r["success"]) == 0 and r["fail_step"]]
    if not fails:
        return
    counts = Counter(fails)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(counts.values(), labels=counts.keys(),
           autopct="%1.0f%%", startangle=90, colors=plt.cm.Set3.colors)
    ax.set_title(f"Анализ ошибок ({name})")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f"failures_{name}.png"), dpi=150)
    plt.close()


# ── Summary ─────────────────────────────────────────────────────────────────

def write_summary(yolo_m, base_m, n_attempts_yolo, n_attempts_base):
    lines = []
    def block(m, n_att, name):
        if m is None:
            lines.append(f"--- {name}: нет данных ---")
            return
        lines.append(f"--- {name} ---")
        lines.append(f"Сценариев:                {m['n_scenarios']}")
        lines.append(f"Индивид. попыток:         {n_att}")
        lines.append(f"Средн. коробок/сценарий:  {m['avg_initial']:.2f}")
        lines.append(f"Средн. перенесено:        {m['avg_picked']:.2f}")
        lines.append(f"Pick rate (всего):        {m['pick_rate']:.1f}%")
        lines.append(f"Full-scenario success:    {m['full_success_rate']:.1f}%")
        lines.append(f"Средн. время сценария:    {m['avg_time']:.1f}с")
        lines.append(f"Медиана времени:          {m['median_time']:.1f}с")
        lines.append("")
    block(yolo_m, n_attempts_yolo, "С YOLO")
    block(base_m, n_attempts_base, "Baseline (без YOLO)")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(ROOT, "summary.txt"), "w") as f:
        f.write(text)


def main():
    scenarios = load_csv(SCENARIOS_CSV)
    metrics = load_csv(METRICS_CSV)
    print(f"Loaded {len(scenarios)} scenarios, {len(metrics)} attempts")

    s_yolo, s_base = split_yolo_baseline(scenarios) if scenarios else ([], [])
    m_yolo = scenario_metrics(s_yolo)
    m_base = scenario_metrics(s_base)

    plot_pick_rate_bar(m_yolo, m_base)
    plot_full_success(m_yolo, m_base)
    plot_scenario_time(s_yolo, "yolo")
    plot_scenario_time(s_base, "baseline")
    plot_attempts_by_size(metrics)
    plot_failures(metrics, "yolo")
    plot_failures(metrics, "baseline")

    n_att_yolo = sum(1 for r in metrics if int(r["use_yolo"]) == 1)
    n_att_base = sum(1 for r in metrics if int(r["use_yolo"]) == 0)
    write_summary(m_yolo, m_base, n_att_yolo, n_att_base)

    print(f"\nPlots → {PLOTS}")


if __name__ == "__main__":
    main()
