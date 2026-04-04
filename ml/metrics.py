# ml/metrics.py
# ============================================================
# FL 실험 성능 지표 수집기
#
# 수집 지표:
#   1. 모델 성능: Accuracy vs Round, Accuracy vs Time
#   2. 통신 효율: Communication Rounds, GSL Usage, Target Acc 도달 시간
#   3. Staleness: 라운드별 평균/분포
#   4. 위성 활용률: Participation Rate, Per-Plane Contribution
#   5. Idle Time: 학습 완료→집계 대기 시간
#   6. 에너지/통신 비용: Upload Count, Effective Updates per GSL
# ============================================================

import json
import csv
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


class MetricsCollector:
    """
    FL 시뮬레이션 전 구간에서 성능 지표를 수집하고
    실험 종료 시 CSV/JSON으로 저장합니다.
    """

    def __init__(self, strategy: str, num_planes: int, sats_per_plane: int,
                 sim_start_time: datetime):
        self.strategy = strategy
        self.num_planes = num_planes
        self.sats_per_plane = sats_per_plane
        self.sim_start_time = sim_start_time

        # ── 1. 모델 성능 ──
        # (round, sim_time_hours, accuracy, loss)
        self.accuracy_history: List[dict] = []

        # ── 2. 통신 효율 ──
        self.total_aggregation_rounds = 0
        self.total_gsl_uploads = 0        # GS에 모델 업로드 횟수
        self.total_gsl_downloads = 0      # GS에서 모델 다운로드 횟수
        self.total_gs_contacts = 0        # GS 접촉 총 횟수 (업/다운/스킵 모두)
        self.target_acc_times: Dict[int, Optional[float]] = {}  # {target%: sim_hours or None}
        self._target_thresholds = [50, 60, 70, 80, 85, 90, 93, 95]

        # ── 3. Staleness ──
        # 라운드별 staleness 값 목록
        self.staleness_per_round: List[dict] = []  # {"round", "values": [...], "mean", "max"}

        # ── 4. 위성 활용률 ──
        self.satellite_contributions: Dict[int, int] = defaultdict(int)  # sat_id → 기여 횟수
        self.plane_contributions: Dict[int, int] = defaultdict(int)      # plane_id → 기여 횟수
        self.satellite_train_count: Dict[int, int] = defaultdict(int)    # sat_id → 학습 횟수

        # ── 5. Idle Time ──
        # 학습 완료 시점 기록 → 집계 시점과의 차이로 idle time 계산
        self.satellite_last_train_time: Dict[int, Optional[datetime]] = defaultdict(lambda: None)
        self.idle_times: List[float] = []  # seconds

        # ── 6. 에너지/통신 비용 ──
        self.effective_updates_per_gsl: List[int] = []  # GSL 접촉 1회당 반영 위성 수

        # ── 이벤트별 타임라인 ──
        self.event_timeline: List[dict] = []

    # ================================================================
    # 이벤트 기록 메서드 (satellite.py에서 호출)
    # ================================================================

    def record_train(self, sat_id: int, plane_id: int, sim_time: datetime):
        """IOT_TRAIN 완료 시 호출"""
        self.satellite_train_count[sat_id] += 1
        self.satellite_last_train_time[sat_id] = sim_time
        self.event_timeline.append({
            "type": "train",
            "sat_id": sat_id,
            "plane_id": plane_id,
            "sim_time": sim_time.isoformat(),
            "sim_hours": self._to_sim_hours(sim_time),
        })

    def record_gs_contact(self, sat_id: int, sim_time: datetime, action: str):
        """
        GS 접촉 시 호출.
        action: "upload" | "download" | "skip"
        """
        self.total_gs_contacts += 1
        if action == "upload":
            self.total_gsl_uploads += 1
        elif action == "download":
            self.total_gsl_downloads += 1

    def record_aggregation(self, round_num: int, sim_time: datetime,
                            accuracy: Optional[float], loss: Optional[float],
                            participating_ids: List[int],
                            staleness_values: List[float],
                            plane_id: Optional[int] = None):
        """
        글로벌 모델 집계 완료 시 호출.
        accuracy=None이면 평가 스킵된 라운드.
        """
        self.total_aggregation_rounds = round_num
        sim_hours = self._to_sim_hours(sim_time)

        # 1. 모델 성능
        if accuracy is not None:
            self.accuracy_history.append({
                "round": round_num,
                "sim_hours": round(sim_hours, 2),
                "accuracy": round(accuracy, 4),
                "loss": round(loss, 6) if loss else None,
            })
            # Target accuracy 도달 시간 체크
            for tgt in self._target_thresholds:
                if tgt not in self.target_acc_times and accuracy >= tgt:
                    self.target_acc_times[tgt] = round(sim_hours, 2)

        # 3. Staleness
        if staleness_values:
            self.staleness_per_round.append({
                "round": round_num,
                "sim_hours": round(sim_hours, 2),
                "values": staleness_values,
                "mean": round(float(np.mean(staleness_values)), 2),
                "max": int(max(staleness_values)),
                "min": int(min(staleness_values)),
            })

        # 4. 위성/plane 기여
        for sid in participating_ids:
            self.satellite_contributions[sid] += 1
        if plane_id is not None:
            self.plane_contributions[plane_id] += 1
        else:
            for sid in participating_ids:
                pid = sid // self.sats_per_plane
                self.plane_contributions[pid] += 1

        # 5. Idle time
        for sid in participating_ids:
            if self.satellite_last_train_time[sid] is not None:
                idle = (sim_time - self.satellite_last_train_time[sid]).total_seconds()
                if idle >= 0:
                    self.idle_times.append(idle)

        # 6. Effective updates per GSL
        self.effective_updates_per_gsl.append(len(participating_ids))

    # ================================================================
    # 요약 및 저장
    # ================================================================

    def get_summary(self, total_satellites: int) -> dict:
        """전체 실험 요약 딕셔너리 반환"""
        participating_sats = set(self.satellite_contributions.keys())
        all_staleness = []
        for s in self.staleness_per_round:
            all_staleness.extend(s["values"])

        # Staleness 히스토그램 (구간: 0, 1-2, 3-5, 6-10, 11-20, 21+)
        bins = [0, 1, 3, 6, 11, 21, float('inf')]
        bin_labels = ["0", "1-2", "3-5", "6-10", "11-20", "21+"]
        staleness_hist = {label: 0 for label in bin_labels}
        for τ in all_staleness:
            for j in range(len(bins) - 1):
                if bins[j] <= τ < bins[j + 1]:
                    staleness_hist[bin_labels[j]] += 1
                    break

        final_acc = self.accuracy_history[-1]["accuracy"] if self.accuracy_history else 0.0

        return {
            "strategy": self.strategy,

            # 1. 모델 성능
            "final_accuracy": final_acc,
            "best_accuracy": max((h["accuracy"] for h in self.accuracy_history), default=0.0),
            "accuracy_history": self.accuracy_history,

            # 2. 통신 효율
            "total_aggregation_rounds": self.total_aggregation_rounds,
            "total_gs_contacts": self.total_gs_contacts,
            "total_gsl_uploads": self.total_gsl_uploads,
            "total_gsl_downloads": self.total_gsl_downloads,
            "target_accuracy_times": {
                f"{k}%": v for k, v in sorted(self.target_acc_times.items())
            },
            "target_accuracy_not_reached": [
                f"{k}%" for k in self._target_thresholds
                if k not in self.target_acc_times
            ],

            # 3. Staleness
            "staleness_overall_mean": round(float(np.mean(all_staleness)), 2) if all_staleness else 0,
            "staleness_overall_max": int(max(all_staleness)) if all_staleness else 0,
            "staleness_histogram": staleness_hist,
            "staleness_per_round": self.staleness_per_round,

            # 4. 위성 활용률
            "total_satellites": total_satellites,
            "participating_satellites": len(participating_sats),
            "participation_rate": round(len(participating_sats) / max(total_satellites, 1) * 100, 1),
            "per_plane_contributions": dict(sorted(self.plane_contributions.items())),
            "top10_contributing_sats": dict(
                sorted(self.satellite_contributions.items(), key=lambda x: -x[1])[:10]
            ),
            "bottom10_contributing_sats": dict(
                sorted(self.satellite_contributions.items(), key=lambda x: x[1])[:10]
            ),
            "satellites_never_contributed": total_satellites - len(participating_sats),

            # 5. Idle Time
            "idle_time_mean_sec": round(float(np.mean(self.idle_times)), 1) if self.idle_times else 0,
            "idle_time_median_sec": round(float(np.median(self.idle_times)), 1) if self.idle_times else 0,
            "idle_time_max_sec": round(float(max(self.idle_times)), 1) if self.idle_times else 0,
            "idle_time_mean_hours": round(float(np.mean(self.idle_times)) / 3600, 2) if self.idle_times else 0,

            # 6. 에너지/통신 비용
            "avg_updates_per_gsl": round(
                float(np.mean(self.effective_updates_per_gsl)), 2
            ) if self.effective_updates_per_gsl else 0,
            "max_updates_per_gsl": max(self.effective_updates_per_gsl, default=0),
        }

    def save(self, output_dir: str = "./results/fedpda_isl"):
        """결과를 JSON + CSV로 저장"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        total_sats = self.num_planes * self.sats_per_plane
        summary = self.get_summary(total_sats)

        # 1. 전체 요약 JSON
        json_path = out / f"{self.strategy}_metrics.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        # 2. Accuracy 곡선 CSV (그래프용)
        acc_csv_path = out / f"{self.strategy}_accuracy.csv"
        with open(acc_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["round", "sim_hours", "accuracy", "loss"])
            writer.writeheader()
            for row in self.accuracy_history:
                writer.writerow(row)

        # 3. Staleness 곡선 CSV
        stale_csv_path = out / f"{self.strategy}_staleness.csv"
        with open(stale_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["round", "sim_hours", "mean", "max", "min"])
            writer.writeheader()
            for row in self.staleness_per_round:
                writer.writerow({
                    "round": row["round"], "sim_hours": row["sim_hours"],
                    "mean": row["mean"], "max": row["max"], "min": row["min"]
                })

        # 4. Plane별 기여 CSV
        plane_csv_path = out / f"{self.strategy}_plane_contributions.csv"
        with open(plane_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["plane_id", "contributions"])
            writer.writeheader()
            for pid in range(1, self.num_planes + 1):
                writer.writerow({
                    "plane_id": pid,
                    "contributions": self.plane_contributions.get(pid, 0)
                })

        return {
            "json": str(json_path),
            "accuracy_csv": str(acc_csv_path),
            "staleness_csv": str(stale_csv_path),
            "plane_csv": str(plane_csv_path),
        }

    def print_summary(self, total_satellites: int, logger=None):
        """콘솔/로거에 요약 출력"""
        s = self.get_summary(total_satellites)
        lines = [
            "",
            "=" * 65,
            f"  📊 실험 결과 요약 [{s['strategy'].upper()}]",
            "=" * 65,
            "",
            "── 1. 모델 성능 ──",
            f"  Final Accuracy:   {s['final_accuracy']:.2f}%",
            f"  Best Accuracy:    {s['best_accuracy']:.2f}%",
            f"  Target Acc Times: {s['target_accuracy_times']}",
            f"  Not Reached:      {s['target_accuracy_not_reached']}",
            "",
            "── 2. 통신 효율 ──",
            f"  Aggregation Rounds:  {s['total_aggregation_rounds']}",
            f"  GS Contacts (total): {s['total_gs_contacts']}",
            f"  GSL Uploads:         {s['total_gsl_uploads']}",
            f"  GSL Downloads:       {s['total_gsl_downloads']}",
            "",
            "── 3. Staleness ──",
            f"  Mean τ:       {s['staleness_overall_mean']}",
            f"  Max τ:        {s['staleness_overall_max']}",
            f"  Distribution: {s['staleness_histogram']}",
            "",
            "── 4. 위성 활용률 ──",
            f"  Participating: {s['participating_satellites']}/{s['total_satellites']} "
            f"({s['participation_rate']}%)",
            f"  Never contributed: {s['satellites_never_contributed']}",
            f"  Top-10 sats:    {s['top10_contributing_sats']}",
            f"  Per-Plane:      {s['per_plane_contributions']}",
            "",
            "── 5. Idle Time ──",
            f"  Mean:   {s['idle_time_mean_sec']:.0f}s ({s['idle_time_mean_hours']:.2f}h)",
            f"  Median: {s['idle_time_median_sec']:.0f}s",
            f"  Max:    {s['idle_time_max_sec']:.0f}s",
            "",
            "── 6. 통신 비용 ──",
            f"  Avg updates/GSL: {s['avg_updates_per_gsl']}",
            f"  Max updates/GSL: {s['max_updates_per_gsl']}",
            "=" * 65,
        ]

        text = "\n".join(lines)
        if logger:
            logger.info(text)
        else:
            print(text)

    # ================================================================
    # 내부 유틸리티
    # ================================================================

    def _to_sim_hours(self, sim_time: datetime) -> float:
        """시뮬레이션 시작 기준 경과 시간 (시간 단위)"""
        return (sim_time - self.sim_start_time).total_seconds() / 3600.0