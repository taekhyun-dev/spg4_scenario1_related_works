"""
FedBuff 시뮬레이션 결과 분석 스크립트
=============================================
추출 항목:
  1. 버퍼 구성 다양성 (Plane Diversity per Flush)
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
METRICS_JSON = "./results/fedbuff_metrics.json"
LOG_FILE = "./logs/simulation_20260219_202354.log"
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_PLANES = 17
SATS_PER_PLANE = 14

def correct_plane_id(sat_id: int) -> int:
    """올바른 plane 매핑 (0-indexed): 17 planes × 14 sats"""
    return sat_id // SATS_PER_PLANE

def correct_plane_id_1indexed(sat_id: int) -> int:
    """1-indexed plane (논문용)"""
    return (sat_id // SATS_PER_PLANE) + 1


# ============================================================
# 1. 버퍼 구성 다양성 (Buffer Plane Diversity)
# ============================================================
def analyze_buffer_diversity():
    """
    시뮬레이션 로그에서 FedBuff Round 라인을 파싱하여
    각 flush의 참여 위성이 몇 개의 서로 다른 plane에서 왔는지 분석.
    
    의미: 다양성이 낮으면 → 같은 궤도면의 유사 데이터만 집계
         → Non-IID 완화 효과 ↓ → 수렴 불안정
    """
    print("=" * 60)
    print("1. 버퍼 구성 다양성 (Buffer Plane Diversity)")
    print("=" * 60)

    pattern = re.compile(r"FedBuff Round #(\d+)\] K=(\d+): \[([^\]]+)\]")
    
    rounds_data = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                round_num = int(match.group(1))
                k = int(match.group(2))
                sat_ids = [int(x.strip()) for x in match.group(3).split(",")]
                
                planes = [correct_plane_id_1indexed(sid) for sid in sat_ids]
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
    total_rounds = len(rounds_data)
    
    print(f"\n총 Flush 횟수: {total_rounds}")
    print(f"버퍼 크기 K: {rounds_data[0]['k']}")
    print(f"\nPlane 다양성 (K={rounds_data[0]['k']}개 위성이 몇 개 plane에서 왔는가):")
    print(f"  평균: {np.mean(diversities):.2f} / {NUM_PLANES}")
    print(f"  중앙값: {np.median(diversities):.1f}")
    print(f"  최소: {np.min(diversities)}")
    print(f"  최대: {np.max(diversities)}")
    print(f"  표준편차: {np.std(diversities):.2f}")

    div_counter = Counter(diversities)
    print(f"\n다양성 분포:")
    for d in sorted(div_counter.keys()):
        pct = div_counter[d] / total_rounds * 100
        bar = "█" * int(pct / 2)
        print(f"  {d}개 plane: {div_counter[d]:>4}회 ({pct:5.1f}%) {bar}")

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

    dominated_count = 0
    for r in rounds_data:
        max_plane_count = max(r["plane_counts"].values())
        if max_plane_count >= r["k"] * 0.5:
            dominated_count += 1
    
    print(f"\n단일 Plane 지배 비율 (버퍼의 50%+ 차지): "
          f"{dominated_count}/{total_rounds} ({dominated_count/total_rounds*100:.1f}%)")

    csv_path = OUTPUT_DIR / "analysis_buffer_diversity.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "k", "num_unique_planes", "unique_planes", "dominant_plane", "dominant_plane_count"])
        for r in rounds_data:
            dominant = max(r["plane_counts"], key=r["plane_counts"].get)
            writer.writerow([
                r["round"], r["k"], r["num_unique_planes"],
                ";".join(map(str, r["unique_planes"])),
                dominant, r["plane_counts"][dominant]
            ])
    print(f"\n저장: {csv_path}")
    
    return rounds_data


# ============================================================
# 2. 통신 기회 활용률 (Communication Utilization)
# ============================================================
def analyze_comm_utilization():
    """
    전체 GS 접촉 이벤트 중 실제 upload / download / skip 비율 분석.
    
    의미: 활용률이 낮으면 → 대부분의 GS 접촉이 낭비
         → 집계 빈도 ↓ → 수렴 속도 ↓
    """
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
    print(f"\n실질 활용률 (Upload+Download): {(uploads+downloads)/total_gs*100:.1f}%")
    print(f"학습 기여율 (Upload only):     {uploads/total_gs*100:.1f}%")

    # 위성별 GS 접촉 파싱
    per_sat_contacts = Counter()
    per_sat_uploads = Counter()
    
    gs_contact_pattern = re.compile(r"SAT_(\d+) : 지상국 접속")
    
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        current_sat = None
        for line in f:
            gs_match = gs_contact_pattern.search(line)
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
    for sat_id in range(238):
        plane = correct_plane_id_1indexed(sat_id)
        plane_stats[plane]["contacts"] += per_sat_contacts.get(sat_id, 0)
        plane_stats[plane]["uploads"] += per_sat_uploads.get(sat_id, 0)

    print(f"\nPlane별 GS 접촉 및 Upload 기여:")
    print(f"  {'Plane':>7} {'접촉':>7} {'Upload':>7} {'기여율':>7}")
    print(f"  {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for p in range(1, NUM_PLANES + 1):
        s = plane_stats[p]
        rate = s["uploads"] / s["contacts"] * 100 if s["contacts"] > 0 else 0
        print(f"  Plane {p:>2}: {s['contacts']:>5}  {s['uploads']:>5}  {rate:>5.1f}%")

    # 위성별 불균형 분석
    upload_counts = [per_sat_uploads.get(sid, 0) for sid in range(238)]
    nonzero_uploads = [c for c in upload_counts if c > 0]

    print(f"\n위성별 Upload 분포:")
    print(f"  Upload 경험 위성: {len(nonzero_uploads)}/238 ({len(nonzero_uploads)/238*100:.1f}%)")
    if nonzero_uploads:
        print(f"  평균 Upload 횟수: {np.mean(nonzero_uploads):.1f}")
        print(f"  최대 Upload 횟수: {np.max(nonzero_uploads)}")
        print(f"  Gini 계수: {gini_coefficient(upload_counts):.3f}")

    sorted_uploads = sorted(upload_counts, reverse=True)
    top_10pct = sorted_uploads[:24]
    total_uploads_all = sum(upload_counts)
    if total_uploads_all > 0:
        top_10_share = sum(top_10pct) / total_uploads_all * 100
        print(f"  상위 10% 위성 기여 비중: {top_10_share:.1f}%")

    csv_path = OUTPUT_DIR / "analysis_comm_utilization.csv"
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

    return plane_stats


def gini_coefficient(values):
    """Gini 계수 (0=완전균등, 1=완전불균등)"""
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
def print_summary(rounds_data, plane_stats):
    print("\n" + "=" * 60)
    print("종합 요약 & 한계점 시사점")
    print("=" * 60)

    with open(METRICS_JSON, "r") as f:
        metrics = json.load(f)

    diversities = [r["num_unique_planes"] for r in rounds_data] if rounds_data else []
    
    print(f"""
전략: {metrics['strategy'].upper()}
최종 정확도: {metrics['best_accuracy']}%
총 집계 라운드: {metrics['total_aggregation_rounds']}

[한계점 1: 버퍼 구성의 낮은 다양성]
  평균 Plane 다양성: {np.mean(diversities):.1f}/17 planes
  → K=10 버퍼에 평균 {np.mean(diversities):.1f}개 plane만 포함
  → 나머지 {17 - np.mean(diversities):.1f}개 plane의 데이터 미반영
  → Non-IID 환경에서 편향된 집계 → 수렴 불안정

[한계점 2: 낮은 통신 기여율]
  전체 GS 접촉: {metrics['total_gs_contacts']:,}
  실제 Upload:  {metrics['total_gsl_uploads']:,} ({metrics['total_gsl_uploads']/metrics['total_gs_contacts']*100:.1f}%)
  → {100 - metrics['total_gsl_uploads']/metrics['total_gs_contacts']*100:.1f}%의 GS 접촉이 모델 업로드에 미활용
  → 특정 궤도면에 편중된 기여 → 공정성 문제

[한계점 3: Staleness 패턴]
  전체 Staleness 평균: {metrics['staleness_overall_mean']}
  → 단일 GS + 높은 접촉 빈도 → staleness 항상 0
  → FedBuff의 staleness 가중치 s(τ)=1로 사실상 무효
""")

    summary_path = OUTPUT_DIR / "analysis_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"FedBuff 한계점 분석 요약\n{'='*50}\n\n")
        f.write(f"전략: {metrics['strategy'].upper()}\n")
        f.write(f"최종 정확도: {metrics['best_accuracy']}%\n")
        f.write(f"총 집계 라운드: {metrics['total_aggregation_rounds']}\n\n")
        f.write(f"[버퍼 다양성] 평균 {np.mean(diversities):.1f}/17 planes per flush\n")
        f.write(f"[통신 활용률] Upload {metrics['total_gsl_uploads']/metrics['total_gs_contacts']*100:.1f}%\n")
        f.write(f"[Staleness] 평균 {metrics['staleness_overall_mean']} (사실상 무의미)\n")
    print(f"저장: {summary_path}")


if __name__ == "__main__":
    print("FedBuff 시뮬레이션 결과 분석\n")
    
    rounds_data = analyze_buffer_diversity()
    plane_stats = analyze_comm_utilization()
    
    if rounds_data:
        print_summary(rounds_data, plane_stats)