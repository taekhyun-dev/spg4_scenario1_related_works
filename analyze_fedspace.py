"""
FedSpace 시뮬레이션 결과 분석 스크립트
=============================================
추출 항목:
  1. 버퍼 구성 다양성 (Buffer Plane Diversity)
  2. 통신 기회 활용률 (Communication Utilization)
  3. 동적 Threshold 효과 분석 (FedSpace 고유)
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
METRICS_JSON = "./results/fedspace_metrics.json"
LOG_FILE = "./logs/simulation_20260222_021702.log"
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_PLANES = 17
SATS_PER_PLANE = 14

def plane_of(sat_id: int) -> int:
    return (sat_id // SATS_PER_PLANE) + 1

def gini(values):
    v = np.array(values, dtype=float)
    if len(v) == 0 or v.sum() == 0:
        return 0.0
    s = np.sort(v)
    n = len(s)
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * s) - (n + 1) * np.sum(s)) / (n * np.sum(s))


# ============================================================
# 1. 버퍼 구성 다양성
# ============================================================
def analyze_buffer_diversity():
    print("=" * 60)
    print("1. 버퍼 구성 다양성 (Buffer Plane Diversity)")
    print("=" * 60)

    # FedSpace는 FedBuff와 동일한 로그 형식
    pattern = re.compile(r"FedBuff Round #(\d+)\] K=(\d+): \[([^\]]+)\]")

    rounds_data = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                round_num = int(match.group(1))
                k = int(match.group(2))
                sat_ids = [int(x.strip()) for x in match.group(3).split(",")]

                planes = [plane_of(sid) for sid in sat_ids]
                unique_planes = set(planes)

                rounds_data.append({
                    "round": round_num,
                    "k": k,
                    "sat_ids": sat_ids,
                    "planes": planes,
                    "num_unique_planes": len(unique_planes),
                    "unique_planes": sorted(unique_planes),
                    "plane_counts": dict(Counter(planes)),
                })

    if not rounds_data:
        print("FedBuff Round 데이터를 찾을 수 없습니다.")
        return None

    diversities = [r["num_unique_planes"] for r in rounds_data]
    k_values = [r["k"] for r in rounds_data]
    total_rounds = len(rounds_data)

    print(f"\n총 Flush 횟수: {total_rounds}")
    print(f"버퍼 크기 K: 평균 {np.mean(k_values):.1f}, 범위 [{min(k_values)}, {max(k_values)}]")

    # K 분포 (FedSpace는 동적 threshold → K가 가변)
    k_counter = Counter(k_values)
    print(f"\nK 크기 분포 (동적 threshold 효과):")
    for k in sorted(k_counter.keys()):
        pct = k_counter[k] / total_rounds * 100
        bar = "█" * int(pct / 2)
        print(f"  K={k:>2}: {k_counter[k]:>4}회 ({pct:5.1f}%) {bar}")

    print(f"\nPlane 다양성:")
    print(f"  평균: {np.mean(diversities):.2f} / {NUM_PLANES}")
    print(f"  중앙값: {np.median(diversities):.1f}")
    print(f"  최소: {np.min(diversities)}, 최대: {np.max(diversities)}")

    div_counter = Counter(diversities)
    print(f"\n다양성 분포:")
    for d in sorted(div_counter.keys()):
        pct = div_counter[d] / total_rounds * 100
        bar = "█" * int(pct / 2)
        print(f"  {d}개 plane: {div_counter[d]:>4}회 ({pct:5.1f}%) {bar}")

    # Plane별 버퍼 참여
    plane_participation = Counter()
    for r in rounds_data:
        for p in r["planes"]:
            plane_participation[p] += 1

    total_slots = sum(plane_participation.values())
    print(f"\nPlane별 버퍼 참여 빈도 (총 {total_slots} 슬롯):")
    for p in range(1, NUM_PLANES + 1):
        count = plane_participation.get(p, 0)
        pct = count / total_slots * 100 if total_slots > 0 else 0
        bar = "█" * int(pct)
        print(f"  Plane {p:>2}: {count:>5}회 ({pct:5.1f}%) {bar}")

    # 단일 plane 지배
    dominated_count = 0
    for r in rounds_data:
        max_plane_count = max(r["plane_counts"].values())
        if max_plane_count >= r["k"] * 0.5:
            dominated_count += 1
    print(f"\n단일 Plane 지배 비율 (50%+): "
          f"{dominated_count}/{total_rounds} ({dominated_count/total_rounds*100:.1f}%)")

    # K와 다양성의 상관관계
    if len(set(k_values)) > 1:
        correlation = np.corrcoef(k_values, diversities)[0, 1]
        print(f"\nK↔다양성 상관계수: {correlation:.3f}")
        k_div_map = defaultdict(list)
        for k, d in zip(k_values, diversities):
            k_div_map[k].append(d)
        print(f"K별 평균 다양성:")
        for k in sorted(k_div_map.keys()):
            print(f"  K={k}: 평균 {np.mean(k_div_map[k]):.2f}")

    # CSV 저장
    csv_path = OUTPUT_DIR / "fedspace_buffer_diversity.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "k", "num_unique_planes", "dominant_plane", "dominant_count"])
        for r in rounds_data:
            dominant = max(r["plane_counts"], key=r["plane_counts"].get)
            writer.writerow([r["round"], r["k"], r["num_unique_planes"],
                             dominant, r["plane_counts"][dominant]])
    print(f"\n저장: {csv_path}")

    buf_csv = OUTPUT_DIR / "fedspace_plane_buffer_participation.csv"
    with open(buf_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["plane_id", "buffer_participations", "pct_of_total"])
        for p in range(1, NUM_PLANES + 1):
            count = plane_participation.get(p, 0)
            pct = count / total_slots * 100 if total_slots > 0 else 0
            writer.writerow([p, count, f"{pct:.1f}"])
    print(f"저장: {buf_csv}")

    return rounds_data


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

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        current_sat = None
        for line in f:
            gs_match = gs_pattern.search(line)
            if gs_match:
                current_sat = int(gs_match.group(1))
                per_sat_contacts[current_sat] += 1
                continue
            if current_sat is not None:
                if "버퍼 추가" in line:
                    per_sat_uploads[current_sat] += 1
                    current_sat = None
                elif "Skip" in line or "다운로드" in line or "Stale" in line:
                    current_sat = None

    # Plane별 통신 활용률
    plane_stats = defaultdict(lambda: {"contacts": 0, "uploads": 0})
    for sid in range(238):
        p = plane_of(sid)
        plane_stats[p]["contacts"] += per_sat_contacts.get(sid, 0)
        plane_stats[p]["uploads"] += per_sat_uploads.get(sid, 0)

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
        print(f"  Gini 계수: {gini(upload_counts):.3f}")
        sorted_up = sorted(upload_counts, reverse=True)
        top10_share = sum(sorted_up[:24]) / sum(upload_counts) * 100
        print(f"  상위 10% 위성 기여 비중: {top10_share:.1f}%")

    # CSV 저장
    csv_path = OUTPUT_DIR / "fedspace_comm_utilization.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sat_id", "plane_id", "gs_contacts", "uploads", "upload_rate"])
        for sid in range(238):
            contacts = per_sat_contacts.get(sid, 0)
            ups = per_sat_uploads.get(sid, 0)
            rate = ups / contacts * 100 if contacts > 0 else 0
            writer.writerow([sid, plane_of(sid), contacts, ups, f"{rate:.1f}"])
    print(f"\n저장: {csv_path}")

    plane_csv = OUTPUT_DIR / "fedspace_plane_comm.csv"
    with open(plane_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["plane_id", "gs_contacts", "uploads", "upload_rate"])
        for p in range(1, NUM_PLANES + 1):
            s = plane_stats[p]
            rate = s["uploads"] / s["contacts"] * 100 if s["contacts"] > 0 else 0
            writer.writerow([p, s["contacts"], s["uploads"], f"{rate:.1f}"])
    print(f"저장: {plane_csv}")

    return plane_stats


# ============================================================
# 3. 동적 Threshold 효과 분석 (FedSpace 고유)
# ============================================================
def analyze_dynamic_threshold():
    print("\n" + "=" * 60)
    print("3. 동적 Threshold 효과 분석 (FedSpace 고유)")
    print("=" * 60)

    flush_pattern = re.compile(
        r"\[FedSpace\] 집계 결정: 버퍼=(\d+) ≥ threshold=(\d+) \(향후 접촉 (\d+)개\)"
    )

    flush_events = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            m = flush_pattern.search(line)
            if m:
                flush_events.append({
                    "buffer_size": int(m.group(1)),
                    "threshold": int(m.group(2)),
                    "upcoming_contacts": int(m.group(3)),
                })

    if not flush_events:
        print("FedSpace 집계 결정 로그를 찾을 수 없습니다.")
        return None

    thresholds = [e["threshold"] for e in flush_events]
    upcoming = [e["upcoming_contacts"] for e in flush_events]
    buf_sizes = [e["buffer_size"] for e in flush_events]

    print(f"\n총 Flush 이벤트: {len(flush_events)}")

    print(f"\n동적 Threshold 분포:")
    print(f"  평균: {np.mean(thresholds):.1f}")
    print(f"  최소: {min(thresholds)}, 최대: {max(thresholds)}")
    th_counter = Counter(thresholds)
    for t in sorted(th_counter.keys()):
        pct = th_counter[t] / len(flush_events) * 100
        bar = "█" * int(pct / 2)
        print(f"  threshold={t:>2}: {th_counter[t]:>4}회 ({pct:5.1f}%) {bar}")

    print(f"\n향후 예상 접촉 수:")
    print(f"  평균: {np.mean(upcoming):.1f}")
    print(f"  최소: {min(upcoming)}, 최대: {max(upcoming)}")

    print(f"\nFlush 시점 버퍼 크기:")
    print(f"  평균: {np.mean(buf_sizes):.1f}")
    print(f"  최소: {min(buf_sizes)}, 최대: {max(buf_sizes)}")

    # Threshold↔접촉 수 관계
    if len(set(upcoming)) > 1:
        corr = np.corrcoef(upcoming, thresholds)[0, 1]
        print(f"\n향후 접촉↔Threshold 상관계수: {corr:.3f}")

    # CSV 저장
    csv_path = OUTPUT_DIR / "fedspace_dynamic_threshold.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["flush_idx", "buffer_size", "threshold", "upcoming_contacts"])
        for i, e in enumerate(flush_events):
            writer.writerow([i + 1, e["buffer_size"], e["threshold"], e["upcoming_contacts"]])
    print(f"\n저장: {csv_path}")

    return flush_events


# ============================================================
# 4. 종합 요약
# ============================================================
def print_summary(rounds_data, plane_stats, flush_events):
    print("\n" + "=" * 60)
    print("종합 요약 & 한계점 시사점")
    print("=" * 60)

    with open(METRICS_JSON, "r") as f:
        metrics = json.load(f)

    diversities = [r["num_unique_planes"] for r in rounds_data] if rounds_data else []
    k_values = [r["k"] for r in rounds_data] if rounds_data else []

    dominated = 0
    if rounds_data:
        for r in rounds_data:
            if max(r["plane_counts"].values()) >= r["k"] * 0.5:
                dominated += 1

    print(f"""
전략: {metrics['strategy'].upper()}
최종 정확도: {metrics['best_accuracy']}%
총 집계 라운드: {metrics['total_aggregation_rounds']}

[한계점 1: 동적 threshold로 인한 낮은 다양성]
  평균 K: {np.mean(k_values):.1f} (동적 조절)
  평균 Plane 다양성: {np.mean(diversities):.1f}/17 planes
  단일 Plane 지배율: {dominated}/{len(rounds_data)} ({dominated/len(rounds_data)*100:.1f}%)
  → 궤도 인식 스케줄링이 K를 줄여 빠른 flush 유도
  → 같은 궤도면 위성끼리 묶여 다양성 ↓
  → FedBuff(K=10, 다양성 2.9) 대비 오히려 악화

[한계점 2: 낮은 통신 기여율]
  전체 GS 접촉: {metrics['total_gs_contacts']:,}
  실제 Upload:  {metrics['total_gsl_uploads']:,} ({metrics['total_gsl_uploads']/metrics['total_gs_contacts']*100:.1f}%)
  → {100 - metrics['total_gsl_uploads']/metrics['total_gs_contacts']*100:.1f}%의 GS 접촉 미활용

[한계점 3: Staleness]
  전체 Staleness 평균: {metrics['staleness_overall_mean']}
  → 단일 GS + 높은 접촉 빈도 → staleness 항상 0
  → FedSpace의 staleness-idleness trade-off 최적화 무의미
""")

    if flush_events:
        thresholds = [e["threshold"] for e in flush_events]
        print(f"[동적 Threshold 효과]")
        print(f"  평균 threshold: {np.mean(thresholds):.1f}")
        print(f"  → 접촉 밀집 구간에서도 threshold가 충분히 올라가지 않음")
        print(f"  → 궤도 예측의 이점이 제한적")

    summary_path = OUTPUT_DIR / "fedspace_analysis_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"FedSpace 한계점 분석 요약\n{'='*50}\n\n")
        f.write(f"전략: {metrics['strategy'].upper()}\n")
        f.write(f"최종 정확도: {metrics['best_accuracy']}%\n")
        f.write(f"총 집계 라운드: {metrics['total_aggregation_rounds']}\n\n")
        f.write(f"[버퍼 다양성] 평균 {np.mean(diversities):.1f}/17 planes, K 평균 {np.mean(k_values):.1f}\n")
        f.write(f"[단일 Plane 지배] {dominated}/{len(rounds_data)} ({dominated/len(rounds_data)*100:.1f}%)\n")
        f.write(f"[통신 활용률] Upload {metrics['total_gsl_uploads']/metrics['total_gs_contacts']*100:.1f}%\n")
        f.write(f"[Staleness] 평균 {metrics['staleness_overall_mean']} (무의미)\n")
    print(f"저장: {summary_path}")


if __name__ == "__main__":
    print("FedSpace 시뮬레이션 결과 분석\n")

    rounds_data = analyze_buffer_diversity()
    plane_stats = analyze_comm_utilization()
    flush_events = analyze_dynamic_threshold()

    if rounds_data:
        print_summary(rounds_data, plane_stats, flush_events)
