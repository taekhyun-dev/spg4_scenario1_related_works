"""
FedAsync 시뮬레이션 결과 분석 스크립트
=============================================
추출 항목:
  1. Plane별 집계 기여 (Per-Plane Aggregation Contribution)
     - FedAsync는 1:1 즉시 집계 → 버퍼 다양성 대신 plane별 기여 편중 분석
  2. 통신 기회 활용률 (Communication Utilization)
"""

import json
import re
import csv
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

# ============================================================
# 설정
# ============================================================
METRICS_JSON = "./results/fedasync_metrics.json"
LOG_FILE = "./logs/simulation_20260218_020534.log"
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_PLANES = 17
SATS_PER_PLANE = 14

def correct_plane_id_1indexed(sat_id: int) -> int:
    return (sat_id // SATS_PER_PLANE) + 1


# ============================================================
# 1. Plane별 집계 기여 분석
# ============================================================
def analyze_plane_contribution():
    """
    FedAsync: GS 접촉 즉시 1:1 집계이므로,
    어떤 위성이 집계에 참여했는지를 로그에서 추출.

    로그 패턴:
      📡 [...] SAT_XXX : 지상국 접속
      ⚡ [FedAsync] α=0.3×s(τ=0)=1.000 → α_eff=0.3000
    → SAT_XXX가 해당 라운드의 유일한 참여 위성
    """
    print("=" * 60)
    print("1. Plane별 집계 기여 (FedAsync: 1:1 즉시 집계)")
    print("=" * 60)

    gs_pattern = re.compile(r"SAT_(\d+) : 지상국 접속")
    async_pattern = re.compile(r"FedAsync")

    aggregation_sats = []  # 집계에 참여한 위성 ID 리스트 (순서대로)
    current_gs_sat = None

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            gs_match = gs_pattern.search(line)
            if gs_match:
                current_gs_sat = int(gs_match.group(1))
                continue

            if current_gs_sat is not None and async_pattern.search(line):
                aggregation_sats.append(current_gs_sat)
                current_gs_sat = None
                continue

            # 다른 이벤트가 나오면 리셋
            if "다운로드" in line or "Skip" in line or "Stale" in line or "IoT" in line:
                current_gs_sat = None

    total_aggs = len(aggregation_sats)
    print(f"\n총 집계 횟수: {total_aggs}")

    # Plane별 기여
    plane_counter = Counter(correct_plane_id_1indexed(sid) for sid in aggregation_sats)
    sat_counter = Counter(aggregation_sats)

    print(f"\nPlane별 집계 기여:")
    print(f"  {'Plane':>7} {'기여':>7} {'비율':>7}")
    print(f"  {'-'*7} {'-'*7} {'-'*7}")
    for p in range(1, NUM_PLANES + 1):
        count = plane_counter.get(p, 0)
        pct = count / total_aggs * 100 if total_aggs > 0 else 0
        bar = "█" * int(pct)
        print(f"  Plane {p:>2}: {count:>5}  {pct:5.1f}% {bar}")

    # 기여 위성 수
    contributing_sats = len(sat_counter)
    print(f"\n집계 기여 위성: {contributing_sats}/238 ({contributing_sats/238*100:.1f}%)")

    # 위성별 기여 분포
    contrib_counts = [sat_counter.get(sid, 0) for sid in range(238)]
    nonzero = [c for c in contrib_counts if c > 0]
    if nonzero:
        print(f"  평균 기여 횟수: {np.mean(nonzero):.1f}")
        print(f"  최대 기여 횟수: {np.max(nonzero)}")
        print(f"  Gini 계수: {gini_coefficient(contrib_counts):.3f}")

    # 상위/하위 plane
    sorted_planes = sorted(plane_counter.items(), key=lambda x: -x[1])
    top3 = sorted_planes[:3]
    bot3 = sorted_planes[-3:]
    top3_share = sum(c for _, c in top3) / total_aggs * 100 if total_aggs > 0 else 0
    bot3_share = sum(c for _, c in bot3) / total_aggs * 100 if total_aggs > 0 else 0
    print(f"\n  상위 3 Plane 기여 비중: {top3_share:.1f}% {top3}")
    print(f"  하위 3 Plane 기여 비중: {bot3_share:.1f}% {bot3}")

    # 라운드별 참여 plane 시계열 (연속된 집계가 같은 plane에서 오는 패턴 확인)
    plane_sequence = [correct_plane_id_1indexed(sid) for sid in aggregation_sats]
    consecutive_same = sum(1 for i in range(1, len(plane_sequence))
                          if plane_sequence[i] == plane_sequence[i-1])
    print(f"\n  연속 동일 Plane 집계: {consecutive_same}/{total_aggs-1} "
          f"({consecutive_same/(total_aggs-1)*100:.1f}%)")

    # CSV 저장
    csv_path = OUTPUT_DIR / "fedasync_plane_contributions_corrected.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["plane_id", "aggregation_contributions", "pct_of_total"])
        for p in range(1, NUM_PLANES + 1):
            count = plane_counter.get(p, 0)
            pct = count / total_aggs * 100 if total_aggs > 0 else 0
            writer.writerow([p, count, f"{pct:.1f}"])
    print(f"\n저장: {csv_path}")

    # 위성별 기여 CSV
    sat_csv_path = OUTPUT_DIR / "fedasync_sat_contributions.csv"
    with open(sat_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sat_id", "correct_plane", "aggregation_contributions"])
        for sid in range(238):
            writer.writerow([sid, correct_plane_id_1indexed(sid), sat_counter.get(sid, 0)])
    print(f"저장: {sat_csv_path}")

    return aggregation_sats, plane_counter


# ============================================================
# 2. 통신 기회 활용률
# ============================================================
def analyze_comm_utilization():
    print("\n" + "=" * 60)
    print("2. 통신 기회 활용률 (Communication Utilization)")
    print("=" * 60)

    with open(METRICS_JSON, "r") as f:
        metrics = json.load(f)

    total_gs = metrics["total_gs_contacts"]
    uploads = metrics["total_gsl_uploads"]
    downloads = metrics["total_gsl_downloads"]
    skips = total_gs - uploads - downloads

    print(f"\n전체 GS 접촉: {total_gs:,}")
    print(f"  Upload  (학습 모델 전송): {uploads:>6,} ({uploads/total_gs*100:5.1f}%)")
    print(f"  Download (글로벌 모델 수신): {downloads:>6,} ({downloads/total_gs*100:5.1f}%)")
    print(f"  Skip     (미학습 & 최신):  {skips:>6,} ({skips/total_gs*100:5.1f}%)")
    print(f"\n학습 기여율 (Upload only): {uploads/total_gs*100:.1f}%")

    # 위성별 GS 접촉 파싱
    per_sat_contacts = Counter()
    per_sat_uploads = Counter()
    gs_pattern = re.compile(r"SAT_(\d+) : 지상국 접속")
    async_pattern = re.compile(r"FedAsync")

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        current_sat = None
        for line in f:
            gs_match = gs_pattern.search(line)
            if gs_match:
                current_sat = int(gs_match.group(1))
                per_sat_contacts[current_sat] += 1
                continue

            if current_sat is not None:
                if async_pattern.search(line):
                    per_sat_uploads[current_sat] += 1
                    current_sat = None
                elif "다운로드" in line or "Skip" in line or "Stale" in line:
                    current_sat = None

    # Plane별 통신 활용률
    plane_stats = defaultdict(lambda: {"contacts": 0, "uploads": 0})
    for sid in range(238):
        plane = correct_plane_id_1indexed(sid)
        plane_stats[plane]["contacts"] += per_sat_contacts.get(sid, 0)
        plane_stats[plane]["uploads"] += per_sat_uploads.get(sid, 0)

    print(f"\nPlane별 GS 접촉 및 Upload 기여:")
    print(f"  {'Plane':>7} {'접촉':>7} {'Upload':>7} {'기여율':>7}")
    print(f"  {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for p in range(1, NUM_PLANES + 1):
        s = plane_stats[p]
        rate = s["uploads"] / s["contacts"] * 100 if s["contacts"] > 0 else 0
        print(f"  Plane {p:>2}: {s['contacts']:>5}  {s['uploads']:>5}  {rate:>5.1f}%")

    # 위성별 불균형
    upload_counts = [per_sat_uploads.get(sid, 0) for sid in range(238)]
    nonzero = [c for c in upload_counts if c > 0]
    print(f"\n위성별 Upload 분포:")
    print(f"  Upload 경험 위성: {len(nonzero)}/238 ({len(nonzero)/238*100:.1f}%)")
    if nonzero:
        print(f"  평균: {np.mean(nonzero):.1f}, 최대: {np.max(nonzero)}")
        print(f"  Gini 계수: {gini_coefficient(upload_counts):.3f}")
        sorted_up = sorted(upload_counts, reverse=True)
        top10_share = sum(sorted_up[:24]) / sum(upload_counts) * 100
        print(f"  상위 10% 위성 기여 비중: {top10_share:.1f}%")

    # CSV 저장
    csv_path = OUTPUT_DIR / "fedasync_comm_utilization.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sat_id", "correct_plane", "gs_contacts", "uploads", "upload_rate"])
        for sid in range(238):
            plane = correct_plane_id_1indexed(sid)
            contacts = per_sat_contacts.get(sid, 0)
            ups = per_sat_uploads.get(sid, 0)
            rate = ups / contacts * 100 if contacts > 0 else 0
            writer.writerow([sid, plane, contacts, ups, f"{rate:.1f}"])
    print(f"\n저장: {csv_path}")

    plane_csv = OUTPUT_DIR / "fedasync_plane_comm.csv"
    with open(plane_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["plane_id", "gs_contacts", "uploads", "upload_rate"])
        for p in range(1, NUM_PLANES + 1):
            s = plane_stats[p]
            rate = s["uploads"] / s["contacts"] * 100 if s["contacts"] > 0 else 0
            writer.writerow([p, s["contacts"], s["uploads"], f"{rate:.1f}"])
    print(f"저장: {plane_csv}")

    return plane_stats


def gini_coefficient(values):
    values = np.array(values, dtype=float)
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_vals) - (n + 1) * np.sum(sorted_vals)) / (n * np.sum(sorted_vals))


# ============================================================
# 3. 종합 요약
# ============================================================
def print_summary(aggregation_sats, plane_counter, plane_stats):
    print("\n" + "=" * 60)
    print("종합 요약 & 한계점 시사점")
    print("=" * 60)

    with open(METRICS_JSON, "r") as f:
        metrics = json.load(f)

    total_aggs = len(aggregation_sats)
    sorted_planes = sorted(plane_counter.items(), key=lambda x: -x[1])
    top3_share = sum(c for _, c in sorted_planes[:3]) / total_aggs * 100

    print(f"""
전략: {metrics['strategy'].upper()}
최종 정확도: {metrics['best_accuracy']}%
총 집계 라운드: {metrics['total_aggregation_rounds']}

[한계점 1: 1:1 집계의 낮은 효율성]
  총 {total_aggs}회 집계, 매번 위성 1개만 반영
  → 라운드당 반영 데이터 다양성 최소 (단일 위성의 Non-IID 데이터)
  → 수렴에 많은 라운드 필요 → 통신 비용 ↑

[한계점 2: Plane별 기여 편중]
  17개 plane 중 상위 3개가 전체의 {top3_share:.1f}% 기여
  → GS 접촉 타이밍이 유리한 궤도면에 집중
  → 나머지 plane의 데이터 반영 지연

[한계점 3: 낮은 통신 기여율]
  전체 GS 접촉: {metrics['total_gs_contacts']:,}
  실제 Upload:  {metrics['total_gsl_uploads']:,} ({metrics['total_gsl_uploads']/metrics['total_gs_contacts']*100:.1f}%)
  → {100 - metrics['total_gsl_uploads']/metrics['total_gs_contacts']*100:.1f}%의 GS 접촉 미활용

[한계점 4: Staleness]
  전체 Staleness 평균: {metrics['staleness_overall_mean']}
  → 단일 GS에서는 staleness가 항상 0 → 가중치 조절 무의미
""")

    summary_path = OUTPUT_DIR / "fedasync_analysis_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"FedAsync 한계점 분석 요약\n{'='*50}\n\n")
        f.write(f"전략: {metrics['strategy'].upper()}\n")
        f.write(f"최종 정확도: {metrics['best_accuracy']}%\n")
        f.write(f"총 집계 라운드: {metrics['total_aggregation_rounds']}\n\n")
        f.write(f"[1:1 집계] {total_aggs}회, 매번 단일 위성\n")
        f.write(f"[Plane 편중] 상위 3 plane이 {top3_share:.1f}% 기여\n")
        f.write(f"[통신 활용률] Upload {metrics['total_gsl_uploads']/metrics['total_gs_contacts']*100:.1f}%\n")
        f.write(f"[Staleness] 평균 {metrics['staleness_overall_mean']} (무의미)\n")
    print(f"저장: {summary_path}")


if __name__ == "__main__":
    print("FedAsync 시뮬레이션 결과 분석\n")

    aggregation_sats, plane_counter = analyze_plane_contribution()
    plane_stats = analyze_comm_utilization()
    print_summary(aggregation_sats, plane_counter, plane_stats)