"""
LEO 위성 연합학습 전략 분석 통합 스크립트
=============================================
지원 전략: FedAsync, FedBuff, FedSpace, FedOrbit

사용법:
  python3 analyze_fl.py <metrics.json> <simulation.log>

출력 (outputs/ 디렉토리):
  - {strategy}_plane_contributions_corrected.csv
  - {strategy}_comm_utilization.csv
  - {strategy}_plane_comm.csv
  - {strategy}_analysis_summary.txt
  + FedBuff/FedSpace 전용:
    - {strategy}_buffer_diversity.csv
    - {strategy}_plane_buffer_participation.csv
  + FedOrbit 전용:
    - {strategy}_isl_aggregation.csv
"""

import json
import re
import csv
import sys
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

# ============================================================
# 설정
# ============================================================
NUM_PLANES = 17
SATS_PER_PLANE = 14

def plane_of(sat_id: int) -> int:
    """1-indexed plane ID"""
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
# 로그 파서: 전략별 집계 참여 위성 추출
# ============================================================

def parse_fedasync(log_path):
    """FedAsync: GS 접속 → ⚡ [FedAsync] 라인이 뒤따르면 해당 위성이 집계 참여"""
    gs_pat = re.compile(r"SAT_(\d+) : 지상국 접속")
    agg_pat = re.compile(r"FedAsync")
    rounds = []  # [{"sat_ids": [int], "round": int}]
    current_sat = None
    round_num = 0

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = gs_pat.search(line)
            if m:
                current_sat = int(m.group(1))
                continue
            if current_sat is not None and agg_pat.search(line):
                round_num += 1
                rounds.append({"round": round_num, "sat_ids": [current_sat]})
                current_sat = None
                continue
            if any(kw in line for kw in ["다운로드", "Skip", "Stale", "IoT"]):
                current_sat = None

    return rounds


def parse_fedbuff(log_path):
    """FedBuff/FedSpace: ⚡ [FedBuff Round #N] K=X: [sat_ids]"""
    pat = re.compile(r"FedBuff Round #(\d+)\] K=(\d+): \[([^\]]+)\]")
    rounds = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if m:
                round_num = int(m.group(1))
                sat_ids = [int(x.strip()) for x in m.group(3).split(",")]
                rounds.append({"round": round_num, "sat_ids": sat_ids, "k": int(m.group(2))})

    return rounds


def parse_fedorbit(log_path):
    """FedOrbit: 🚀 [FedOrbit] Plane X Master SAT_Y → GS Upload (Nge 위성)
    + ISL 집계: 🔗 [FedOrbit ISL] Plane X: Nge 위성 intra-plane 집계 완료"""
    upload_pat = re.compile(r"\[FedOrbit\] Plane (\d+) Master SAT_(\d+).*?(\d+)개 위성")
    isl_pat = re.compile(r"\[FedOrbit ISL\] Plane (\d+): (\d+)개 위성")
    
    rounds = []
    isl_events = []
    round_num = 0

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = upload_pat.search(line)
            if m:
                round_num += 1
                plane_id = int(m.group(1))
                master_sat = int(m.group(2))
                n_sats = int(m.group(3))
                # 정확한 참여 위성 ID는 로그에 없으므로 plane 기반 추정
                plane_sats = list(range(plane_id * SATS_PER_PLANE, (plane_id + 1) * SATS_PER_PLANE))
                rounds.append({
                    "round": round_num,
                    "sat_ids": plane_sats[:n_sats],  # 근사
                    "plane_id": plane_id,
                    "master_sat": master_sat,
                    "n_participants": n_sats,
                })
                continue

            m = isl_pat.search(line)
            if m:
                isl_events.append({
                    "plane_id": int(m.group(1)),
                    "n_sats": int(m.group(2)),
                })

    return rounds, isl_events


# ============================================================
# 공통 분석 1: Plane별 집계 기여
# ============================================================

def analyze_plane_contribution(strategy, rounds, output_dir):
    print("=" * 60)
    print(f"1. Plane별 집계 기여 [{strategy.upper()}]")
    print("=" * 60)

    # 모든 집계 참여 위성 수집
    all_sats = []
    for r in rounds:
        all_sats.extend(r["sat_ids"])

    total_participations = len(all_sats)
    total_rounds = len(rounds)
    print(f"\n총 집계 횟수: {total_rounds}")
    print(f"총 위성 참여 슬롯: {total_participations}")

    if total_participations == 0:
        print("참여 데이터 없음")
        return

    # Plane별 기여
    plane_counter = Counter(plane_of(sid) for sid in all_sats)
    sat_counter = Counter(all_sats)

    print(f"\nPlane별 집계 기여:")
    print(f"  {'Plane':>7} {'기여':>7} {'비율':>7}")
    print(f"  {'-'*7} {'-'*7} {'-'*7}")
    for p in range(1, NUM_PLANES + 1):
        count = plane_counter.get(p, 0)
        pct = count / total_participations * 100
        bar = "█" * int(pct)
        print(f"  Plane {p:>2}: {count:>5}  {pct:5.1f}% {bar}")

    # 위성별 기여 분포
    contrib_counts = [sat_counter.get(sid, 0) for sid in range(238)]
    nonzero = [c for c in contrib_counts if c > 0]
    print(f"\n집계 기여 위성: {len(nonzero)}/238 ({len(nonzero)/238*100:.1f}%)")
    if nonzero:
        print(f"  평균 기여: {np.mean(nonzero):.1f}, 최대: {np.max(nonzero)}")
        print(f"  Gini 계수: {gini(contrib_counts):.3f}")
        sorted_up = sorted(contrib_counts, reverse=True)
        top10_share = sum(sorted_up[:24]) / sum(contrib_counts) * 100
        print(f"  상위 10% 위성 기여 비중: {top10_share:.1f}%")

    # Plane 편중도
    sorted_planes = sorted(plane_counter.items(), key=lambda x: -x[1])
    top3_share = sum(c for _, c in sorted_planes[:3]) / total_participations * 100
    bot3_share = sum(c for _, c in sorted_planes[-3:]) / total_participations * 100
    print(f"\n  상위 3 Plane: {top3_share:.1f}% {sorted_planes[:3]}")
    print(f"  하위 3 Plane: {bot3_share:.1f}% {sorted_planes[-3:]}")

    # CSV 저장
    csv_path = output_dir / f"{strategy}_plane_contributions_corrected.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plane_id", "contributions", "pct_of_total"])
        for p in range(1, NUM_PLANES + 1):
            count = plane_counter.get(p, 0)
            pct = count / total_participations * 100
            w.writerow([p, count, f"{pct:.1f}"])
    print(f"\n저장: {csv_path}")

    sat_csv = output_dir / f"{strategy}_sat_contributions.csv"
    with open(sat_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sat_id", "plane_id", "contributions"])
        for sid in range(238):
            w.writerow([sid, plane_of(sid), sat_counter.get(sid, 0)])
    print(f"저장: {sat_csv}")

    return plane_counter


# ============================================================
# 공통 분석 2: 통신 기회 활용률
# ============================================================

def analyze_comm_utilization(strategy, metrics_json_path, log_path, output_dir):
    print("\n" + "=" * 60)
    print(f"2. 통신 기회 활용률 [{strategy.upper()}]")
    print("=" * 60)

    with open(metrics_json_path, "r") as f:
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
    gs_pat = re.compile(r"SAT_(\d+) : 지상국 접속")

    # 전략별 upload 감지 패턴
    if strategy == "fedasync":
        upload_indicator = lambda line: "FedAsync" in line
    elif strategy in ("fedbuff", "fedspace"):
        upload_indicator = lambda line: "버퍼 추가" in line
    elif strategy == "fedorbit":
        upload_indicator = lambda line: "[FedOrbit]" in line and "GS Upload" in line
    else:
        upload_indicator = lambda line: False

    with open(log_path, "r", encoding="utf-8") as f:
        current_sat = None
        for line in f:
            m = gs_pat.search(line)
            if m:
                current_sat = int(m.group(1))
                per_sat_contacts[current_sat] += 1
                continue
            if current_sat is not None:
                if upload_indicator(line):
                    per_sat_uploads[current_sat] += 1
                    current_sat = None
                elif any(kw in line for kw in ["다운로드", "Skip", "Stale"]):
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
    csv_path = output_dir / f"{strategy}_comm_utilization.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sat_id", "plane_id", "gs_contacts", "uploads", "upload_rate"])
        for sid in range(238):
            contacts = per_sat_contacts.get(sid, 0)
            ups = per_sat_uploads.get(sid, 0)
            rate = ups / contacts * 100 if contacts > 0 else 0
            w.writerow([sid, plane_of(sid), contacts, ups, f"{rate:.1f}"])
    print(f"\n저장: {csv_path}")

    plane_csv = output_dir / f"{strategy}_plane_comm.csv"
    with open(plane_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plane_id", "gs_contacts", "uploads", "upload_rate"])
        for p in range(1, NUM_PLANES + 1):
            s = plane_stats[p]
            rate = s["uploads"] / s["contacts"] * 100 if s["contacts"] > 0 else 0
            w.writerow([p, s["contacts"], s["uploads"], f"{rate:.1f}"])
    print(f"저장: {plane_csv}")

    return metrics, plane_stats


# ============================================================
# 전략별 추가 분석
# ============================================================

def analyze_buffer_diversity(strategy, rounds, output_dir):
    """FedBuff/FedSpace 전용: 버퍼 구성 다양성"""
    print("\n" + "=" * 60)
    print(f"3. 버퍼 구성 다양성 [{strategy.upper()}]")
    print("=" * 60)

    if not rounds or "k" not in rounds[0]:
        print("버퍼 데이터 없음")
        return

    total_rounds = len(rounds)
    diversities = []
    plane_participation = Counter()

    for r in rounds:
        planes = [plane_of(sid) for sid in r["sat_ids"]]
        unique = set(planes)
        r["_planes"] = planes
        r["_unique"] = len(unique)
        r["_plane_counts"] = dict(Counter(planes))
        diversities.append(len(unique))
        for p in planes:
            plane_participation[p] += 1

    print(f"\n총 Flush 횟수: {total_rounds}, K={rounds[0]['k']}")
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

    # 단일 plane 지배
    dominated = sum(1 for r in rounds if max(r["_plane_counts"].values()) >= r["k"] * 0.5)
    print(f"\n단일 Plane 지배 비율 (50%+): {dominated}/{total_rounds} ({dominated/total_rounds*100:.1f}%)")

    # Plane별 버퍼 참여
    total_slots = sum(plane_participation.values())
    print(f"\nPlane별 버퍼 참여 빈도:")
    for p in range(1, NUM_PLANES + 1):
        count = plane_participation.get(p, 0)
        pct = count / total_slots * 100 if total_slots > 0 else 0
        bar = "█" * int(pct)
        print(f"  Plane {p:>2}: {count:>5}회 ({pct:5.1f}%) {bar}")

    # CSV
    csv_path = output_dir / f"{strategy}_buffer_diversity.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "k", "num_unique_planes", "dominant_plane", "dominant_count"])
        for r in rounds:
            dom = max(r["_plane_counts"], key=r["_plane_counts"].get)
            w.writerow([r["round"], r["k"], r["_unique"], dom, r["_plane_counts"][dom]])
    print(f"\n저장: {csv_path}")

    buf_csv = output_dir / f"{strategy}_plane_buffer_participation.csv"
    with open(buf_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plane_id", "buffer_participations", "pct_of_total"])
        for p in range(1, NUM_PLANES + 1):
            count = plane_participation.get(p, 0)
            pct = count / total_slots * 100 if total_slots > 0 else 0
            w.writerow([p, count, f"{pct:.1f}"])
    print(f"저장: {buf_csv}")

    return diversities


def analyze_fedorbit_isl(strategy, isl_events, output_dir):
    """FedOrbit 전용: ISL 집계 분석"""
    print("\n" + "=" * 60)
    print(f"3. ISL Intra-Plane 집계 분석 [{strategy.upper()}]")
    print("=" * 60)

    if not isl_events:
        print("ISL 이벤트 없음")
        return

    print(f"\n총 ISL 집계 이벤트: {len(isl_events)}")

    plane_isl = Counter(e["plane_id"] for e in isl_events)
    avg_sats = np.mean([e["n_sats"] for e in isl_events])
    print(f"평균 참여 위성/집계: {avg_sats:.1f}")

    print(f"\nPlane별 ISL 집계 횟수:")
    for p in range(NUM_PLANES):
        count = plane_isl.get(p, 0)
        print(f"  Plane {p+1:>2}: {count}회")

    csv_path = output_dir / f"{strategy}_isl_aggregation.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plane_id", "isl_aggregations", "avg_participants"])
        plane_sats_list = defaultdict(list)
        for e in isl_events:
            plane_sats_list[e["plane_id"]].append(e["n_sats"])
        for p in range(NUM_PLANES):
            vals = plane_sats_list.get(p, [])
            w.writerow([p + 1, len(vals), f"{np.mean(vals):.1f}" if vals else "0"])
    print(f"\n저장: {csv_path}")


# ============================================================
# 종합 요약
# ============================================================

def print_summary(strategy, metrics, rounds, output_dir, diversities=None):
    print("\n" + "=" * 60)
    print(f"종합 요약 [{strategy.upper()}]")
    print("=" * 60)

    total_aggs = metrics["total_aggregation_rounds"]
    total_gs = metrics["total_gs_contacts"]
    uploads = metrics["total_gsl_uploads"]
    upload_rate = uploads / total_gs * 100 if total_gs > 0 else 0

    # 1:1 vs 버퍼 특성
    if strategy == "fedasync":
        agg_desc = f"1:1 즉시 집계 {total_aggs}회, 매번 단일 위성"
    elif strategy in ("fedbuff", "fedspace"):
        k = rounds[0]["k"] if rounds and "k" in rounds[0] else "?"
        agg_desc = f"K={k} 버퍼 집계 {len(rounds)}회 flush"
        if diversities:
            agg_desc += f", 평균 {np.mean(diversities):.1f}/17 plane 다양성"
    elif strategy == "fedorbit":
        agg_desc = f"Plane 기반 집계 {total_aggs}회 (ISL → Master → GS)"
    else:
        agg_desc = f"{total_aggs}회 집계"

    lines = f"""
전략: {strategy.upper()}
최종 정확도: {metrics.get('best_accuracy', 'N/A')}%
총 집계 라운드: {total_aggs}

[집계 방식] {agg_desc}
[통신 활용률] Upload {uploads:,}/{total_gs:,} ({upload_rate:.1f}%)
[Staleness] 평균 {metrics.get('staleness_overall_mean', 'N/A')}
"""
    print(lines)

    summary_path = output_dir / f"{strategy}_analysis_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"{strategy.upper()} 한계점 분석 요약\n{'='*50}\n")
        f.write(lines)
    print(f"저장: {summary_path}")


# ============================================================
# 메인
# ============================================================

def detect_strategy_from_log(log_path):
    """로그 첫 부분에서 전략명 감지"""
    with open(log_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > 30:
                break
            low = line.lower()
            if "strategy: fedasync" in low or "[fedasync]" in low:
                return "fedasync"
            if "strategy: fedbuff" in low or "[fedbuff]" in low:
                return "fedbuff"
            if "strategy: fedspace" in low or "[fedspace]" in low:
                return "fedspace"
            if "strategy: fedorbit" in low or "[fedorbit]" in low:
                return "fedorbit"
    return None


def main():
    if len(sys.argv) < 2:
        print("사용법:")
        print("  python3 analyze_fl.py <metrics.json> <simulation.log>")
        print("  python3 analyze_fl.py <simulation.log>  (JSON 없이 로그만)")
        sys.exit(1)

    # 인자 파싱: JSON + Log 또는 Log만
    if len(sys.argv) >= 3:
        metrics_path = sys.argv[1]
        log_path = sys.argv[2]
    else:
        metrics_path = None
        log_path = sys.argv[1]

    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)

    # 전략 자동 감지
    if metrics_path and Path(metrics_path).exists():
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)
        strategy = metrics_data["strategy"].lower()
    else:
        metrics_path = None
        strategy = detect_strategy_from_log(log_path)
        if not strategy:
            print("전략을 감지할 수 없습니다. metrics.json을 함께 제공해주세요.")
            sys.exit(1)
        metrics_data = None
        print(f"⚠️  JSON 없음 — 로그에서 전략 감지: {strategy.upper()}")

    print(f"LEO 위성 연합학습 분석 [{strategy.upper()}]")
    print(f"Metrics: {metrics_path or '(없음 - 로그만 사용)'}")
    print(f"Log: {log_path}\n")

    # 전략별 파싱
    isl_events = None
    if strategy == "fedasync":
        rounds = parse_fedasync(log_path)
    elif strategy in ("fedbuff", "fedspace"):
        rounds = parse_fedbuff(log_path)
    elif strategy == "fedorbit":
        rounds, isl_events = parse_fedorbit(log_path)
    else:
        print(f"알 수 없는 전략: {strategy}")
        sys.exit(1)

    # 공통 분석
    analyze_plane_contribution(strategy, rounds, output_dir)

    metrics = None
    if metrics_path:
        metrics, plane_stats = analyze_comm_utilization(strategy, metrics_path, log_path, output_dir)
    else:
        print("\n⚠️  통신 활용률 분석에는 metrics.json이 필요합니다. (스킵)")

    # 전략별 추가 분석
    diversities = None
    if strategy in ("fedbuff", "fedspace"):
        diversities = analyze_buffer_diversity(strategy, rounds, output_dir)
    elif strategy == "fedorbit" and isl_events:
        analyze_fedorbit_isl(strategy, isl_events, output_dir)

    # 종합 요약
    if metrics:
        print_summary(strategy, metrics, rounds, output_dir, diversities)

    print(f"\n✅ 분석 완료: {output_dir}")


if __name__ == "__main__":
    main()
