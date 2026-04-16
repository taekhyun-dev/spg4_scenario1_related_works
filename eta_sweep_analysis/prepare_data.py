#!/usr/bin/env python3
"""
eta_sweep 데이터를 분석용으로 정리.
- 각 케이스별 CSV를 통일된 네이밍으로 복사
- JSON에서 요약 지표 추출 → summary CSV
- 전 케이스 accuracy를 하나의 CSV로 병합 (비교 플롯용)
"""

import json
import csv
import shutil
import numpy as np
from pathlib import Path

SRC = Path(__file__).parent.parent / "eta_sweep"
DST = Path(__file__).parent / "data"
DST.mkdir(exist_ok=True)

CASES = [
    ("alpha0.1_eta0.1", 0.1, 0.1),
    ("alpha0.1_eta0.3", 0.1, 0.3),
    ("alpha0.1_eta0.5", 0.1, 0.5),
    ("alpha0.5_eta0.1", 0.5, 0.1),
    ("alpha0.5_eta0.3", 0.5, 0.3),
    ("alpha0.5_eta0.5", 0.5, 0.5),
]

# ── 1. 각 케이스별 CSV 복사 (통일된 네이밍) ──
for folder, alpha, eta in CASES:
    src_dir = SRC / folder
    prefix = f"a{alpha}_eta{eta}"

    for fname in ["fedpda_accuracy.csv", "fedpda_staleness.csv", "fedpda_plane_contributions.csv"]:
        src_file = src_dir / fname
        dst_name = f"{prefix}_{fname.replace('fedpda_', '')}"
        if src_file.exists():
            shutil.copy2(src_file, DST / dst_name)

print("CSV 복사 완료")

# ── 2. JSON → 요약 테이블 ──
summary_rows = []
for folder, alpha, eta in CASES:
    jpath = SRC / folder / "fedpda_metrics.json"
    with open(jpath) as f:
        m = json.load(f)

    acc_hist = [h["accuracy"] for h in m["accuracy_history"]]
    n = len(acc_hist)
    late = acc_hist[n * 2 // 3:]

    # 70%, 80% 도달 시간
    t70, t80 = None, None
    for h in m["accuracy_history"]:
        if t70 is None and h["accuracy"] >= 70.0:
            t70 = h["sim_hours"]
        if t80 is None and h["accuracy"] >= 80.0:
            t80 = h["sim_hours"]

    # staleness 평균
    spath = SRC / folder / "fedpda_staleness.csv"
    staleness_mean = None
    if spath.exists():
        import csv as csvmod
        with open(spath) as sf:
            reader = csvmod.DictReader(sf)
            vals = [float(r["mean"]) for r in reader]
            staleness_mean = np.mean(vals) if vals else None

    summary_rows.append({
        "alpha": alpha,
        "eta_g": eta,
        "preservation": f"{(1 - eta) * 100:.0f}%",
        "best_acc": m["best_accuracy"],
        "final_acc": m["final_accuracy"],
        "late_mean": round(np.mean(late), 2),
        "late_std": round(np.std(late), 2),
        "total_rounds": n * 5,
        "time_to_70pct": round(t70, 1) if t70 else "N/A",
        "time_to_80pct": round(t80, 1) if t80 else "N/A",
        "avg_staleness": round(staleness_mean, 2) if staleness_mean else "N/A",
    })

# 요약 CSV
summary_path = DST / "summary.csv"
fields = list(summary_rows[0].keys())
with open(summary_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(summary_rows)
print(f"요약 테이블: {summary_path}")

# ── 3. 전 케이스 accuracy 병합 (비교용) ──
# α=0.1 그룹
for alpha_val in [0.1, 0.5]:
    merged_path = DST / f"merged_accuracy_a{alpha_val}.csv"
    # 모든 eta 케이스의 accuracy를 round 기준으로 병합
    all_data = {}
    etas = []
    for folder, alpha, eta in CASES:
        if alpha != alpha_val:
            continue
        etas.append(eta)
        apath = SRC / folder / "fedpda_accuracy.csv"
        with open(apath) as af:
            reader = csv.DictReader(af)
            for r in reader:
                rd = int(r["round"])
                if rd not in all_data:
                    all_data[rd] = {"round": rd, "sim_hours": float(r["sim_hours"])}
                all_data[rd][f"acc_eta{eta}"] = float(r["accuracy"])
                all_data[rd][f"loss_eta{eta}"] = float(r["loss"])

    with open(merged_path, "w", newline="") as f:
        cols = ["round", "sim_hours"]
        for e in sorted(etas):
            cols += [f"acc_eta{e}", f"loss_eta{e}"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for rd in sorted(all_data.keys()):
            w.writerow(all_data[rd])
    print(f"병합 accuracy: {merged_path}")

# ── 4. 전 케이스 staleness 병합 ──
for alpha_val in [0.1, 0.5]:
    merged_path = DST / f"merged_staleness_a{alpha_val}.csv"
    all_data = {}
    etas = []
    for folder, alpha, eta in CASES:
        if alpha != alpha_val:
            continue
        etas.append(eta)
        spath = SRC / folder / "fedpda_staleness.csv"
        with open(spath) as sf:
            reader = csv.DictReader(sf)
            for r in reader:
                rd = int(r["round"])
                if rd not in all_data:
                    all_data[rd] = {"round": rd, "sim_hours": float(r["sim_hours"])}
                all_data[rd][f"mean_eta{eta}"] = float(r["mean"])
                all_data[rd][f"max_eta{eta}"] = int(r["max"])

    with open(merged_path, "w", newline="") as f:
        cols = ["round", "sim_hours"]
        for e in sorted(etas):
            cols += [f"mean_eta{e}", f"max_eta{e}"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for rd in sorted(all_data.keys()):
            w.writerow(all_data[rd])
    print(f"병합 staleness: {merged_path}")

print("\n데이터 정리 완료!")
