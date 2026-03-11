# object/satellite.py
# ============================================================
# LEO 위성 비동기 연합학습 비교 실험
# 570km Walker-Delta: 17 planes × 14 sats = 238 satellites
#
# 4개 전략 비교 (config.py AGGREGATION_STRATEGY로 전환):
#   1. FedAsync  - 1:1 즉시 비동기 (Xie et al., 2019)
#   2. FedBuff   - K-버퍼 pseudo-gradient averaging (Nguyen et al., 2022)
#   3. FedSpace  - 궤도 인식 동적 스케줄링 (So et al., 2022)
#   4. FedOrbit  - Plane 클러스터 + 마스터 위성 (Jabbarpour et al., 2024)
#   5. FedPDA    - Plane-Diversity-Aware Adaptive Buffering (Proposed)
#
# 공통 개선사항:
#   - 미학습 위성 필터링
#   - Cosine Annealing LR
#   - LOCAL_TRAIN 후 평가 제거
#   - GLOBAL_TEST 평가 주기 조절
# ============================================================

import asyncio
import torch
import numpy as np
import math
import random
from datetime import datetime, timedelta, timezone
from utils.skyfield_utils import EarthSatellite
from utils.logging_setup import setup_loggers, KST
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from torch.utils.data import DataLoader
from collections import defaultdict, OrderedDict
from skyfield.api import load, wgs84

from config_fedpda import (
    IOT_FLYOVER_THRESHOLD_DEG, GS_FLYOVER_THRESHOLD_DEG, LOCAL_EPOCHS,
    AGGREGATION_STRATEGY, NUM_PLANES, SATS_PER_PLANE, ORBIT_PERIOD_SEC,
    # FedAsync
    FEDASYNC_STALENESS_FUNC, FEDASYNC_ALPHA_MAX,
    # FedBuff
    FEDBUFF_K, FEDBUFF_SERVER_LR, FEDBUFF_SERVER_MOMENTUM,
    # FedSpace
    FEDSPACE_PREDICT_WINDOW_SEC, FEDSPACE_MIN_BUFFER, FEDSPACE_STALENESS_WEIGHT, FEDSPACE_SERVER_MOMENTUM,
    # FedOrbit
    FEDORBIT_INTRA_AGG_INTERVAL_SEC, FEDORBIT_SERVER_LR,
    # FedPDA (Proposed)
    FEDPDA_PREDICT_WINDOW_SEC, FEDPDA_MIN_BUFFER, FEDPDA_STALENESS_WEIGHT,
    FEDPDA_MIN_DIVERSITY, FEDPDA_MAX_BUFFER, FEDPDA_TIMEOUT_SEC, FEDPDA_SERVER_MOMENTUM, FEDPDA_SERVER_LR,
    # Common
    BASE_LR, MIN_LR, EVAL_EVERY_N_ROUNDS, STALENESS_THRESHOLD,
    NUM_CLIENTS, DIRICHLET_ALPHA, BATCH_SIZE, SAMPLES_PER_CLIENT,
    # Simulation time
    SIM_START_TIME, SIM_DURATION_DAYS,
)

from ml.data import get_cifar10_loaders
from ml.model import create_resnet9, PyTorchModel
from ml.training import train_model
from ml.aggregation import weighted_update
from ml.metrics import MetricsCollector

SEED = 42

class Satellite_Manager:
    """
    위성 연합학습 시뮬레이션 매니저.
    570km Walker-Delta (17×14) constellation에서
    FedAsync / FedBuff / FedSpace / FedOrbit / FedPDA 5가지 전략을 비교 실험합니다.
    """

    def __init__(self, start_time: datetime, end_time: datetime, sim_logger, perf_logger):
        self.start_time = start_time
        self.end_time = end_time
        self.sim_logger = sim_logger
        self.perf_logger = perf_logger

        self.satellites: Dict[int, EarthSatellite] = {}
        self.satellite_models: Dict[int, PyTorchModel] = {}
        self.satellite_performances: Dict[int, float] = {}
        self.satellite_last_trained_version: Dict[int, float] = {}

        # 위성별 글로벌 모델 다운로드 시점 기록 (시간 기반 staleness용)
        self.satellite_download_time: Dict[int, datetime] = {}

        # 학습 시작 전 base state 저장 (pseudo-gradient 정확 계산용)
        self.satellite_base_state: Dict[int, OrderedDict] = {}

        self.check_arr = defaultdict(list)

        # --- FL 설정 ---
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_satellites = NUM_CLIENTS
        self.NUM_CLASSES = 10
        self.strategy = AGGREGATION_STRATEGY

        # --- Aggregation 상태 ---
        self.aggregation_round = 0
        self.total_rounds = 0

        # FedBuff: pseudo-gradient 버퍼 + 서버 모멘텀
        self.gs_buffer: List[dict] = []
        self.server_momentum_state: Optional[OrderedDict] = None

        # FedSpace: 접촉 예측 캐시
        self.gs_contact_schedule: List[dict] = []

        # FedOrbit: plane 클러스터 상태
        self.plane_buffers: Dict[int, List[dict]] = defaultdict(list)
        self.plane_masters: Dict[int, int] = {}
        self.last_intra_agg_time: Dict[int, datetime] = {}

        # FedPDA: plane-diversity-aware 버퍼 + 별도 모멘텀
        self.pda_buffer: List[dict] = []
        self.pda_momentum_state: Optional[OrderedDict] = None

        self.sim_logger.info(f"Strategy: {self.strategy.upper()}")
        self.sim_logger.info("CIFAR-10 데이터셋 로드 및 샘플링 중...")

        self.avg_data_count, self.client_subsets, self.val_loader, _ = get_cifar10_loaders(
            num_clients=self.num_satellites,
            dirichlet_alpha=DIRICHLET_ALPHA,
            data_root='./data',
            samples_per_client=SAMPLES_PER_CLIENT
        )
        self.sim_logger.info(f"데이터셋 로드 완료. 위성당 데이터: {self.avg_data_count:.0f}장")

        self.global_model_net = create_resnet9(num_classes=self.NUM_CLASSES)
        self.global_model_net.to('cpu')
        self.global_model_wrapper = PyTorchModel.from_model(self.global_model_net, version=0.0)
        self.best_acc = 0.0

        # --- 성능 지표 수집기 ---
        self.metrics = MetricsCollector(
            strategy=self.strategy,
            num_planes=NUM_PLANES,
            sats_per_plane=SATS_PER_PLANE,
            sim_start_time=self.start_time,
        )

        self.sim_logger.info("위성 관리자 생성 완료.")

    # ================================================================
    # Walker-Delta Constellation 유틸리티
    # ================================================================

    @staticmethod
    def get_plane_id(sat_id: int) -> int:
        """TLE 순번 기반: sat_id를 SATS_PER_PLANE(14)으로 나눠 plane 결정.
        sat_id 0~13 → plane 0, 14~27 → plane 1, ..., 224~237 → plane 16"""
        return sat_id // SATS_PER_PLANE

    @staticmethod
    def get_position_in_plane(sat_id: int) -> int:
        """plane 내 위치 (0~13)"""
        return sat_id % SATS_PER_PLANE

    def get_plane_satellites(self, plane_id: int) -> List[int]:
        """특정 plane에 속하는 모든 위성 ID 반환"""
        return [sid for sid in self.satellites.keys() if sid // SATS_PER_PLANE == plane_id]

    # ================================================================
    # 궤도/통신 스케줄 (모든 전략 공통)
    # ================================================================

    def load_constellation(self):
        tle_path = "constellation.tle"
        satellites = {}
        try:
            with open(tle_path, "r") as f:
                lines = [line.strip() for line in f.readlines()]
                i = 0
                while i < len(lines):
                    if not lines[i]:
                        i += 1
                        continue
                    name, line1, line2 = lines[i:i + 3]
                    sat_id = int(name.replace("SAT", "").replace("_", ""))
                    satellites[sat_id] = EarthSatellite(line1, line2, name)
                    i += 3
            self.satellites = satellites
            self.sim_logger.info(
                f"Constellation 로드: {len(satellites)}개 위성, "
                f"{NUM_PLANES} planes × {SATS_PER_PLANE} sats"
            )
        except Exception as e:
            self.sim_logger.error(f"TLE 파일 로드 실패: {e}")
            raise e

    async def run(self):
        self.sim_logger.info("위성 관리자 운영 시작.")
        self.load_constellation()

        for sat_id in self.satellites.keys():
            self.satellite_models[sat_id] = PyTorchModel.from_model(self.global_model_net, version=0.0)
            self.satellite_performances[sat_id] = 0.0
            self.satellite_last_trained_version[sat_id] = -1.0
            self.satellite_download_time[sat_id] = self.start_time

        await self.propagate_orbit(self.start_time, self.end_time)
        self.sim_logger.info(f"궤도 전파 완료 ({len(self.times)} steps).")

        await self.check_iot_comm()
        await self.check_gs_comm()
        self.sim_logger.info("모든 통신 스케줄 계산 완료.")

        if self.strategy == "fedorbit":
            self._fedorbit_init_masters()

        await self.manage_fl_process()
        self.sim_logger.info("모든 시뮬레이션 종료.")

    async def propagate_orbit(self, start_time, end_time):
        step = timedelta(seconds=10)
        self.times = []
        curr = start_time
        while curr < end_time:
            self.times.append(curr)
            curr += step
        ts = load.timescale()
        self.t_vector = ts.from_datetimes(self.times)

    async def check_iot_comm(self):
        self.sim_logger.info("IoT 통신 가능 시간 분석 시작...")
        iot_devices = [
            {"name": "Amazon_Forest", "loc": wgs84.latlon(-3.47, -62.37, elevation_m=100)},
            {"name": "Great_Barrier_Reef", "loc": wgs84.latlon(-18.29, 147.77, elevation_m=0)},
            {"name": "Abisko Tundra", "loc": wgs84.latlon(68.35, 18.79, elevation_m=420)},
        ]
        for iot in iot_devices:
            for sat_id, satellite in self.satellites.items():
                difference = satellite - iot['loc']
                topocentric = difference.at(self.t_vector)
                alt, _, _ = topocentric.altaz()
                visible_indices = np.where(alt.degrees > IOT_FLYOVER_THRESHOLD_DEG)[0]
                if len(visible_indices) == 0:
                    continue
                breaks = np.where(np.diff(visible_indices) > 1)[0] + 1
                windows = np.split(visible_indices, breaks)
                for window in windows:
                    st = self.times[window[0]]
                    et = self.times[window[-1]]
                    dur = (et - st).total_seconds()
                    if dur == 0:
                        dur = 10
                    self.check_arr[sat_id].append({
                        "type": "IOT_TRAIN", "start_time": st, "end_time": et,
                        "duration": dur, "target": iot['name']
                    })

    async def check_gs_comm(self):
        self.sim_logger.info("지상국 통신 가능 시간 분석 시작...")
        gs = {"name": "Ground Station", "loc": wgs84.latlon(37.5665, 126.9780, elevation_m=34)}
        for sat_id, satellite in self.satellites.items():
            difference = satellite - gs['loc']
            topocentric = difference.at(self.t_vector)
            alt, _, _ = topocentric.altaz()
            visible_indices = np.where(alt.degrees > GS_FLYOVER_THRESHOLD_DEG)[0]
            if len(visible_indices) == 0:
                continue
            breaks = np.where(np.diff(visible_indices) > 1)[0] + 1
            windows = np.split(visible_indices, breaks)
            for window in windows:
                st = self.times[window[0]]
                et = self.times[window[-1]]
                dur = (et - st).total_seconds()
                if dur == 0:
                    dur = 10
                self.check_arr[sat_id].append({
                    "type": "GS_AGGREGATE", "start_time": st, "end_time": et,
                    "duration": dur, "target": gs['name']
                })

    # ================================================================
    # 공통 유틸리티
    # ================================================================

    def _get_cosine_lr(self) -> float:
        """Cosine Annealing LR: BASE_LR → MIN_LR"""
        progress = min(self.aggregation_round / max(self.total_rounds, 1), 1.0)
        return MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1 + math.cos(math.pi * progress))

    def _evaluate_direct(self, model, data_loader, sat_id, version, stage):
        model.to(self.device)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
        self.perf_logger.info(
            f"{datetime.now(KST).isoformat()},{stage},{sat_id},{version:.2f},"
            f"{self.strategy},{acc:.4f},{avg_loss:.6f},0.0000"
        )
        model.to('cpu')
        return acc, avg_loss

    def _is_trained_since_global(self, sat_id) -> bool:
        return self.satellite_last_trained_version[sat_id] > 0

    @staticmethod
    def _is_gradient_param(key: str, tensor: torch.Tensor) -> bool:
        """pseudo-gradient 연산 대상인지 판별.
        BatchNorm의 num_batches_tracked(int64) 등 non-float 텐서는 제외."""
        return tensor.is_floating_point()

    def _staleness_function(self, staleness: float) -> float:
        """s(τ) = 1/(1+τ)^0.5 — FedAsync/FedBuff 공통"""
        if FEDASYNC_STALENESS_FUNC == "poly":
            return (1.0 + staleness) ** (-0.5)
        elif FEDASYNC_STALENESS_FUNC == "hinge":
            return 1.0 if staleness <= STALENESS_THRESHOLD else 0.0
        else:
            return 1.0

    def _compute_staleness(self, local_wrapper, event_time: datetime) -> Tuple[float, float]:
        """버전 기반 staleness 계산.
        Returns: (τ_version, τ_time_normalized)"""
        tau_ver = max(0, self.global_model_wrapper.version - int(local_wrapper.version))
        return tau_ver, 0.0

    def _update_global_and_evaluate(self, new_state_dict, new_version,
                                     participating_ids, temp_model, force_eval=False,
                                     staleness_values=None, sim_time=None,
                                     plane_id=None):
        """글로벌 모델 업데이트 + 평가 + 체크포인트 + 메트릭 기록 (공통)"""
        self.global_model_net.load_state_dict(new_state_dict)

        g_acc, g_loss = None, None
        if force_eval or (self.aggregation_round % EVAL_EVERY_N_ROUNDS == 0):
            temp_model.load_state_dict(new_state_dict)
            g_acc, g_loss = self._evaluate_direct(
                temp_model, self.val_loader, sat_id="GS",
                version=new_version, stage="GLOBAL_TEST"
            )
            if g_acc > self.best_acc:
                prev = self.best_acc
                self.best_acc = g_acc
                save_dir = Path("./checkpoints")
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model_state_dict': new_state_dict,
                    'version': new_version,
                    'accuracy': g_acc, 'loss': g_loss,
                    'round': self.aggregation_round,
                    'strategy': self.strategy,
                    'participants': participating_ids,
                }, save_dir / f"{self.strategy}_v{int(new_version)}_acc{g_acc:.2f}.pth")
                self.sim_logger.info(f"   💾 New Best! ({prev:.2f}% → {g_acc:.2f}%)")
            self.sim_logger.info(
                f"   📊 Round #{self.aggregation_round}: v{new_version:.0f} Acc: {g_acc:.2f}%"
            )
        else:
            self.sim_logger.info(
                f"   📊 Round #{self.aggregation_round}: v{new_version:.0f} (평가 스킵)"
            )

        self.global_model_wrapper = PyTorchModel(
            version=new_version,
            model_state_dict=new_state_dict,
            trained_by=self.global_model_wrapper.trained_by + participating_ids
        )

        self.metrics.record_aggregation(
            round_num=self.aggregation_round,
            sim_time=sim_time or self.start_time,
            accuracy=g_acc,
            loss=g_loss,
            participating_ids=participating_ids,
            staleness_values=staleness_values or [],
            plane_id=plane_id,
        )

    # ================================================================
    # Strategy 1: FedAsync (Xie et al., 2019)
    #
    # GS 접촉 즉시 1:1 가중 평균 (논문 Algorithm 1):
    #   α_t = α × s(τ),  s(τ) = (1+τ)^{-0.5}
    #   x_t = (1 - α_t) * x_global + α_t * x_local
    # ================================================================

    def _fedasync_aggregate(self, sat_id, local_wrapper, temp_model, event_time):
        self.aggregation_round += 1
        new_version = round(self.global_model_wrapper.version + 1.0, 1)

        tau_ver, _ = self._compute_staleness(local_wrapper, event_time)
        s_tau = self._staleness_function(tau_ver)

        alpha_eff = FEDASYNC_ALPHA_MAX * s_tau

        self.sim_logger.info(
            f"   ⚡ [FedAsync] α={FEDASYNC_ALPHA_MAX}×s(τ={tau_ver})={s_tau:.3f} → α_eff={alpha_eff:.4f}"
        )

        new_sd = weighted_update(
            self.global_model_wrapper.model_state_dict,
            local_wrapper.model_state_dict, alpha_eff
        )
        self._update_global_and_evaluate(
            new_sd, new_version, [sat_id], temp_model,
            staleness_values=[tau_ver], sim_time=event_time,
        )

        self.metrics.record_gs_contact(sat_id, event_time, "upload")

        self.satellite_models[sat_id] = PyTorchModel.from_model(
            self.global_model_net, version=new_version
        )
        self.satellite_last_trained_version[sat_id] = -1.0
        self.satellite_download_time[sat_id] = event_time

    # ================================================================
    # Strategy 2: FedBuff (Nguyen et al., 2022)
    #
    # 논문 원본 pseudo-gradient averaging:
    #   Client: Δ_i = w_before - w_after (pseudo-gradient)
    #   Server: K개 모이면 Δ_avg = (1/K) Σ s(τ_i)·Δ_i
    #           w_{t+1} = w_t - η_g · Δ_avg
    #   서버 모멘텀: m_t = β·m_{t-1} + Δ_avg, w_{t+1} = w_t - η_g·m_t
    # ================================================================

    def _fedbuff_collect(self, sat_id, local_wrapper, event_time):
        """버퍼에 pseudo-gradient용 데이터 수집"""
        tau_ver, _ = self._compute_staleness(local_wrapper, event_time)
        s_tau = self._staleness_function(tau_ver)
        loader_idx = sat_id % len(self.client_subsets)

        self.gs_buffer.append({
            "sat_id": sat_id,
            "state_dict": local_wrapper.model_state_dict,
            "base_state_dict": self.satellite_base_state.get(sat_id, {}),
            "base_version": int(local_wrapper.version),
            "staleness": tau_ver,
            "s_tau": s_tau,
            "data_count": len(self.client_subsets[loader_idx]),
            "event_time": event_time,
        })
        self.sim_logger.info(
            f"   📦 버퍼 추가 (v{local_wrapper.version:.1f}, "
            f"τ={tau_ver}, 버퍼: {len(self.gs_buffer)}/{FEDBUFF_K})"
        )

    def _fedbuff_flush(self, temp_model, force_eval=False):
        """Satellite-adapted FedBuff: staleness-weighted buffered averaging.

        위성 환경 적응:
          논문(Nguyen et al., 2022)은 base_i - trained_i (pseudo-gradient SGD)를 사용하지만,
          위성 환경의 구조적 고지연(mean τ≈15)에서는 base_i ≠ global_current로 인해
          stale pseudo-gradient drift가 발생하여 발산함.

          global_current 기준 + s(τ) 정규화 가중합으로 변경하면:
            Δ_avg = Σ (s_i/Σs) × (global_current - trained_i)
            new = global - η_g × Δ_avg
          η_g=1.0일 때: new = Σ (s_i/Σs) × trained_i → convex combination → 수렴 안정성 보장.
        """
        if len(self.gs_buffer) == 0:
            return

        self.aggregation_round += 1
        new_version = round(self.global_model_wrapper.version + 1.0, 1)
        K = len(self.gs_buffer)
        participating_ids = [m["sat_id"] for m in self.gs_buffer]

        self.sim_logger.info(
            f"\n⚡ [FedBuff Round #{self.aggregation_round}] K={K}: {participating_ids}"
        )

        global_sd = self.global_model_wrapper.model_state_dict

        # Satellite-adapted: Δ_avg = Σ (s_i/Σs) × (global_current - trained_i)
        # s(τ) 정규화 → stale 위성 기여 자동 감소, η_g=1.0에서 convex combination
        total_s = sum(m["s_tau"] for m in self.gs_buffer)
        if total_s == 0:
            total_s = float(K)

        delta_avg = OrderedDict()
        for key in global_sd.keys():
            if not self._is_gradient_param(key, global_sd[key]):
                delta_avg[key] = None
                continue
            delta = torch.zeros_like(global_sd[key], dtype=torch.float32)
            for m in self.gs_buffer:
                pseudo_grad = global_sd[key].float() - m["state_dict"][key].float()
                delta += (m["s_tau"] / total_s) * pseudo_grad
            delta_avg[key] = delta

        # 서버 모멘텀: m_t = β·m_{t-1} + Δ_avg
        if self.strategy == "fedbuff":
            beta = FEDBUFF_SERVER_MOMENTUM
        elif self.strategy == "fedspace":
            beta = FEDSPACE_SERVER_MOMENTUM

        if self.server_momentum_state is None:
            self.server_momentum_state = OrderedDict()
            for key in delta_avg:
                if delta_avg[key] is not None:
                    self.server_momentum_state[key] = delta_avg[key].clone()
        else:
            for key in delta_avg:
                if delta_avg[key] is not None and key in self.server_momentum_state:
                    self.server_momentum_state[key] = (
                        beta * self.server_momentum_state[key] + delta_avg[key]
                    )

        # w_{t+1} = w_t - η_g · m_t
        eta_g = FEDBUFF_SERVER_LR
        new_sd = OrderedDict()
        for key in global_sd.keys():
            if not self._is_gradient_param(key, global_sd[key]):
                new_sd[key] = global_sd[key].clone()
            elif key in self.server_momentum_state:
                new_sd[key] = (
                    global_sd[key].float() - eta_g * self.server_momentum_state[key]
                ).to(global_sd[key].dtype).cpu()
            else:
                new_sd[key] = global_sd[key].clone()

        self.sim_logger.info(f"   📐 η_g={eta_g}, β={beta}, K={K}")

        staleness_vals = [m["staleness"] for m in self.gs_buffer]
        sim_time = self.gs_buffer[-1]["event_time"]

        self._update_global_and_evaluate(
            new_sd, new_version, participating_ids, temp_model, force_eval,
            staleness_values=staleness_vals, sim_time=sim_time,
        )

        for m in self.gs_buffer:
            self.metrics.record_gs_contact(m["sat_id"], sim_time, "upload")

        # 참여 위성 동기화
        for m in self.gs_buffer:
            self.satellite_models[m["sat_id"]] = PyTorchModel.from_model(
                self.global_model_net, version=new_version
            )
            self.satellite_last_trained_version[m["sat_id"]] = -1.0

        self.gs_buffer = []

    # ================================================================
    # Strategy 3: FedSpace (So et al., 2022)
    #
    # 핵심: 궤도 예측으로 GS 접촉 밀도를 파악 →
    #       staleness-idleness trade-off를 동적으로 최적화하여 집계 시점 결정.
    # ================================================================

    def _fedspace_predict_upcoming_contacts(self, current_time, window_sec=None) -> int:
        """향후 window_sec 내 예상 GS 접촉 수"""
        if window_sec is None:
            window_sec = FEDSPACE_PREDICT_WINDOW_SEC
        deadline = current_time + timedelta(seconds=window_sec)
        count = 0
        for evt in self.gs_contact_schedule:
            if evt["start_time"] > deadline:
                break
            if evt["start_time"] > current_time:
                count += 1
        return count

    def _fedspace_should_flush(self, current_time, buffer_size, is_last=False) -> bool:
        """FedSpace 동적 flush 판단"""
        if is_last or buffer_size <= 0:
            return is_last and buffer_size > 0

        upcoming = self._fedspace_predict_upcoming_contacts(current_time)
        w = FEDSPACE_STALENESS_WEIGHT

        # 접촉 많으면 threshold 높여서 더 모으고, 적으면 빨리 집계
        dynamic_threshold = max(
            FEDSPACE_MIN_BUFFER,
            int(FEDSPACE_MIN_BUFFER + w * min(upcoming, 15))
        )

        should = buffer_size >= dynamic_threshold
        if should:
            self.sim_logger.info(
                f"   🌍 [FedSpace] 집계 결정: 버퍼={buffer_size} ≥ "
                f"threshold={dynamic_threshold} (향후 접촉 {upcoming}개)"
            )
        return should

    def _fedspace_aggregate(self, temp_model, force_eval=False):
        """FedSpace: FedBuff와 동일한 pseudo-gradient 방식, 시점만 다름"""
        self._fedbuff_flush(temp_model, force_eval)

    # ================================================================
    # Strategy 5: FedPDA (Proposed)
    # Plane-Diversity-Aware Adaptive Buffering
    #
    # FedSpace의 동적 시점 최적화를 기반으로 두 가지 핵심 개선:
    #
    # 1. Dual-condition flush:
    #    - Size 조건: buffer_size >= dynamic_threshold (FedSpace와 동일)
    #    - Diversity 조건: unique_planes >= MIN_DIVERSITY
    #    - Timeout 안전장치: 버퍼 체류 시간 > TIMEOUT_SEC → 강제 flush
    #    - Fallback: buffer_size >= MAX_BUFFER → diversity 무관 강제 flush
    #
    # 2. Diversity-weighted aggregation:
    #    - 표준: w_i = s(τ_i) / Σs  (staleness만 고려)
    #    - FedPDA: w_i = s(τ_i) / c_{p_i}  (plane 내 중복 수로 나눔)
    #    → 과대대표 plane의 기여를 자동 감소, 소수 plane 기여 부스트
    #    → 정규화 후 convex combination으로 수렴 안정성 보장
    #
    # 동기:
    #   Walker-Delta에서 동일 궤도면 위성이 시간 군집하여 GS 접촉
    #   → 버퍼의 60~86%가 단일 plane 지배 (Gap 1 실증)
    #   → per-round Non-IID 강화로 집계 품질 저하
    #   → diversity 조건 + 가중치 보정으로 해결
    # ================================================================

    def _fedpda_collect(self, sat_id, local_wrapper, event_time):
        """FedPDA 버퍼에 학습 완료 모델 수집.
        _fedbuff_collect과 동일하나 별도 버퍼(pda_buffer) 사용."""
        tau_ver, _ = self._compute_staleness(local_wrapper, event_time)
        s_tau = self._staleness_function(tau_ver)
        loader_idx = sat_id % len(self.client_subsets)
        plane_id = self.get_plane_id(sat_id)

        self.pda_buffer.append({
            "sat_id": sat_id,
            "plane_id": plane_id,
            "state_dict": local_wrapper.model_state_dict,
            "base_state_dict": self.satellite_base_state.get(sat_id, {}),
            "base_version": int(local_wrapper.version),
            "staleness": tau_ver,
            "s_tau": s_tau,
            "data_count": len(self.client_subsets[loader_idx]),
            "event_time": event_time,
        })

        unique_planes = len(set(m["plane_id"] for m in self.pda_buffer))
        self.sim_logger.info(
            f"   📦 [PDA] 버퍼 추가 (v{local_wrapper.version:.1f}, "
            f"τ={tau_ver}, P{plane_id}, "
            f"버퍼: {len(self.pda_buffer)}, planes: {unique_planes})"
        )

    def _fedpda_get_buffer_stats(self) -> dict:
        """현재 PDA 버퍼의 다양성 통계"""
        if not self.pda_buffer:
            return {"size": 0, "unique_planes": 0, "plane_counts": {}, "oldest_time": None}
        plane_ids = [m["plane_id"] for m in self.pda_buffer]
        plane_counts = {}
        for p in plane_ids:
            plane_counts[p] = plane_counts.get(p, 0) + 1
        return {
            "size": len(self.pda_buffer),
            "unique_planes": len(plane_counts),
            "plane_counts": plane_counts,
            "oldest_time": self.pda_buffer[0]["event_time"],
        }

    def _fedpda_predict_upcoming_contacts(self, current_time, window_sec=None) -> int:
        """향후 window_sec 내 예상 GS 접촉 수 (FedSpace와 동일 로직)"""
        if window_sec is None:
            window_sec = FEDPDA_PREDICT_WINDOW_SEC
        deadline = current_time + timedelta(seconds=window_sec)
        count = 0
        for evt in self.gs_contact_schedule:
            if evt["start_time"] > deadline:
                break
            if evt["start_time"] > current_time:
                count += 1
        return count

    def _fedpda_should_flush(self, current_time, is_last=False) -> bool:
        """FedPDA dual-condition flush 판단.

        Flush 조건 (OR):
          1. Primary: size >= dynamic_threshold AND unique_planes >= MIN_DIVERSITY
          2. Timeout: 버퍼 내 가장 오래된 모델이 TIMEOUT_SEC 초과 체류
          3. Fallback: size >= MAX_BUFFER (diversity 미충족이어도 강제)
          4. is_last: 시뮬레이션 마지막 이벤트
        """
        stats = self._fedpda_get_buffer_stats()
        if stats["size"] == 0:
            return False
        if is_last:
            return True

        # 동적 size threshold (FedSpace와 동일 로직)
        upcoming = self._fedpda_predict_upcoming_contacts(current_time)
        w = FEDPDA_STALENESS_WEIGHT
        dynamic_threshold = max(
            FEDPDA_MIN_BUFFER,
            int(FEDPDA_MIN_BUFFER + w * min(upcoming, 15))
        )

        size_ok = stats["size"] >= dynamic_threshold
        diversity_ok = stats["unique_planes"] >= FEDPDA_MIN_DIVERSITY

        # Condition 1: Primary (size + diversity 모두 충족)
        if size_ok and diversity_ok:
            self.sim_logger.info(
                f"   🎯 [FedPDA] PRIMARY flush: 버퍼={stats['size']} ≥ {dynamic_threshold}, "
                f"planes={stats['unique_planes']} ≥ {FEDPDA_MIN_DIVERSITY} "
                f"(향후 접촉 {upcoming}개)"
            )
            return True

        # Condition 2: Timeout (다양성 대기 제한 초과)
        if stats["oldest_time"] is not None:
            wait_sec = (current_time - stats["oldest_time"]).total_seconds()
            if wait_sec >= FEDPDA_TIMEOUT_SEC and stats["size"] >= FEDPDA_MIN_BUFFER:
                self.sim_logger.info(
                    f"   ⏰ [FedPDA] TIMEOUT flush: 대기 {wait_sec:.0f}s ≥ {FEDPDA_TIMEOUT_SEC}s, "
                    f"버퍼={stats['size']}, planes={stats['unique_planes']}"
                )
                return True

        # Condition 3: Fallback (버퍼 상한 초과)
        if stats["size"] >= FEDPDA_MAX_BUFFER:
            self.sim_logger.info(
                f"   🔴 [FedPDA] FALLBACK flush: 버퍼={stats['size']} ≥ {FEDPDA_MAX_BUFFER}, "
                f"planes={stats['unique_planes']}"
            )
            return True

        return False

    def _fedpda_flush(self, temp_model, force_eval=False):
        """FedPDA diversity-weighted pseudo-gradient averaging.

        FedBuff/FedSpace 대비 3가지 차별화:
          1. Diversity 가중치: w_i = s(τ_i) / c_{p_i}
             과대대표 plane 기여 감소, 소수 plane 부스트
          2. Global Preservation: η_g < 1.0
             w_new = (1-η_g)·w_global + η_g·Σ nw_i·w_trained_i
             severe Non-IID에서 진동 억제 (기존: η_g=1.0 → 글로벌 완전 교체)
          3. Dual-condition flush: size ≥ threshold AND planes ≥ MIN_DIVERSITY
        """
        if len(self.pda_buffer) == 0:
            return

        self.aggregation_round += 1
        new_version = round(self.global_model_wrapper.version + 1.0, 1)
        K = len(self.pda_buffer)
        participating_ids = [m["sat_id"] for m in self.pda_buffer]

        stats = self._fedpda_get_buffer_stats()
        self.sim_logger.info(
            f"\n⚡ [FedPDA Round #{self.aggregation_round}] K={K}, "
            f"planes={stats['unique_planes']}: {participating_ids}"
        )

        global_sd = self.global_model_wrapper.model_state_dict

        # Diversity-weighted pseudo-gradient averaging
        # w_i = s(τ_i) / c_{p_i} where c_{p_i} = 버퍼 내 plane p의 위성 수
        plane_counts = stats["plane_counts"]
        raw_weights = []
        for m in self.pda_buffer:
            c_p = plane_counts[m["plane_id"]]
            raw_weights.append(m["s_tau"] / c_p)

        total_w = sum(raw_weights)
        if total_w == 0:
            total_w = float(K)
            raw_weights = [1.0] * K

        # 정규화된 가중치
        norm_weights = [w / total_w for w in raw_weights]

        # 로깅: plane별 가중치 합산
        plane_weight_sum = {}
        for m, nw in zip(self.pda_buffer, norm_weights):
            p = m["plane_id"]
            plane_weight_sum[p] = plane_weight_sum.get(p, 0.0) + nw
        self.sim_logger.info(
            f"   📐 [PDA] plane weights: {', '.join(f'P{p}:{w:.3f}' for p, w in sorted(plane_weight_sum.items()))}"
        )

        # ── Diversity-Weighted Pseudo-Gradient with Global Preservation ──
        #
        # 수식:
        #   Δ_avg = Σ nw_i × (w_global - w_trained_i)   (nw_i = diversity weight)
        #   w_new = w_global - η_g × Δ_avg
        #         = (1 - η_g) × w_global + η_g × Σ nw_i × w_trained_i
        #
        # η_g < 1.0 → 글로벌 모델의 (1-η_g)가 보존됨
        # 이는 FedAsync의 mixing ratio를 버퍼 전략에 일반화한 것:
        #   FedAsync:  w = (1-α_eff)·w_g + α_eff·w_local     (단일 위성)
        #   FedPDA:    w = (1-η_g)·w_g + η_g·Σ nw_i·w_i     (diversity-weighted 다수 위성)
        #
        # β=0 (모멘텀 비사용) → severe Non-IID에서 모멘텀 발산 방지

        eta_g = FEDPDA_SERVER_LR
        beta = FEDPDA_SERVER_MOMENTUM  # 0.0 (비사용)

        delta_avg = OrderedDict()
        for key in global_sd.keys():
            if not self._is_gradient_param(key, global_sd[key]):
                delta_avg[key] = None
                continue
            delta = torch.zeros_like(global_sd[key], dtype=torch.float32)
            for m, nw in zip(self.pda_buffer, norm_weights):
                pseudo_grad = global_sd[key].float() - m["state_dict"][key].float()
                delta += nw * pseudo_grad
            delta_avg[key] = delta

        # 서버 모멘텀: m_t = β·m_{t-1} + Δ_avg
        # 서버 모멘텀 (β > 0일 때만 작동, 현재 β=0.0)
        if beta > 0:
            if self.pda_momentum_state is None:
                self.pda_momentum_state = OrderedDict()
                for key in delta_avg:
                    if delta_avg[key] is not None:
                        self.pda_momentum_state[key] = delta_avg[key].clone()
            else:
                for key in delta_avg:
                    if delta_avg[key] is not None and key in self.pda_momentum_state:
                        self.pda_momentum_state[key] = (
                            beta * self.pda_momentum_state[key] + delta_avg[key]
                        )
            effective_delta = self.pda_momentum_state
        else:
            effective_delta = delta_avg

        # w_{t+1} = w_t - η_g · Δ = (1-η_g)·w_t + η_g · Σ nw_i · w_trained_i
        new_sd = OrderedDict()
        for key in global_sd.keys():
            if not self._is_gradient_param(key, global_sd[key]):
                new_sd[key] = global_sd[key].clone()
            elif effective_delta.get(key) is not None:
                new_sd[key] = (
                    global_sd[key].float() - eta_g * effective_delta[key]
                ).to(global_sd[key].dtype).cpu()
            else:
                new_sd[key] = global_sd[key].clone()

        retention_pct = (1.0 - eta_g) * 100

        self.sim_logger.info(
            f"   📐 η_g={eta_g} (global retention={retention_pct:.0f}%), "
            f"β={beta}, K={K}, "
            f"diversity={stats['unique_planes']}/{NUM_PLANES}"
        )

        staleness_vals = [m["staleness"] for m in self.pda_buffer]
        sim_time = self.pda_buffer[-1]["event_time"]

        self._update_global_and_evaluate(
            new_sd, new_version, participating_ids, temp_model, force_eval,
            staleness_values=staleness_vals, sim_time=sim_time,
        )

        for m in self.pda_buffer:
            self.metrics.record_gs_contact(m["sat_id"], sim_time, "upload")

        # 참여 위성 동기화
        for m in self.pda_buffer:
            self.satellite_models[m["sat_id"]] = PyTorchModel.from_model(
                self.global_model_net, version=new_version
            )
            self.satellite_last_trained_version[m["sat_id"]] = -1.0

        self.pda_buffer = []

    # ================================================================
    # Strategy 4: FedOrbit (Jabbarpour et al., 2024)
    #
    # 핵심 구성:
    #   1. Plane-based Clustering: 17개 orbital plane = 17개 클러스터
    #   2. Master Satellite: plane 내 GS 접촉 빈도 최다 위성
    #   3. Intra-Plane ISL Aggregation: plane 내 FedAvg
    #   4. Master → GS Upload: 마스터가 plane 대표 모델을 pseudo-gradient로 전송
    # ================================================================

    def _fedorbit_init_masters(self):
        """각 plane에서 GS 접촉 빈도가 가장 높은 위성을 마스터로 선정"""
        plane_gs_count = defaultdict(lambda: defaultdict(int))

        for sat_id, events in self.check_arr.items():
            plane = self.get_plane_id(sat_id)
            gs_contacts = sum(1 for e in events if e["type"] == "GS_AGGREGATE")
            plane_gs_count[plane][sat_id] = gs_contacts

        for plane_id in range(NUM_PLANES):
            if plane_id in plane_gs_count and plane_gs_count[plane_id]:
                master = max(plane_gs_count[plane_id], key=plane_gs_count[plane_id].get)
                self.plane_masters[plane_id] = master
                self.last_intra_agg_time[plane_id] = self.start_time
                self.sim_logger.info(
                    f"   🛰️ Plane {plane_id}: Master=SAT_{master} "
                    f"(GS접촉 {plane_gs_count[plane_id][master]}회)"
                )

    def _fedorbit_intra_plane_collect(self, sat_id, local_wrapper, event_time):
        """Intra-Plane ISL 버퍼에 수집 (ISL 교환 모사)"""
        plane_id = self.get_plane_id(sat_id)
        loader_idx = sat_id % len(self.client_subsets)

        new_entry = {
            "sat_id": sat_id,
            "state_dict": local_wrapper.model_state_dict,
            "version": local_wrapper.version,
            "data_count": len(self.client_subsets[loader_idx]),
        }

        # 기존 엔트리 교체 (같은 위성 ID가 있으면 최신으로 덮어쓰기)
        buf = self.plane_buffers[plane_id]
        replaced = False
        for idx, entry in enumerate(buf):
            if entry["sat_id"] == sat_id:
                buf[idx] = new_entry
                replaced = True
                break

        if not replaced:
            buf.append(new_entry)

    def _fedorbit_try_intra_aggregate(self, plane_id, event_time, temp_model):
        """Plane 내 ISL 집계: 주기적으로 plane 내 모델들을 FedAvg"""
        buf = self.plane_buffers[plane_id]
        if len(buf) == 0:
            return None

        elapsed = (event_time - self.last_intra_agg_time.get(plane_id, self.start_time)).total_seconds()
        if elapsed < FEDORBIT_INTRA_AGG_INTERVAL_SEC and len(buf) < SATS_PER_PLANE:
            return None

        self.last_intra_agg_time[plane_id] = event_time

        total_data = sum(m["data_count"] for m in buf)
        if total_data == 0:
            total_data = len(buf)

        aggregated = OrderedDict()
        for key in buf[0]["state_dict"].keys():
            if not self._is_gradient_param(key, buf[0]["state_dict"][key]):
                aggregated[key] = buf[0]["state_dict"][key].clone()
                continue
            param = torch.zeros_like(buf[0]["state_dict"][key], dtype=torch.float32)
            for m in buf:
                w = m["data_count"] / total_data
                param += w * m["state_dict"][key].float()
            aggregated[key] = param.to(buf[0]["state_dict"][key].dtype).cpu()

        self.sim_logger.info(
            f"   🔗 [FedOrbit ISL] Plane {plane_id}: "
            f"{len(buf)}개 위성 intra-plane 집계 완료"
        )

        participating = [m["sat_id"] for m in buf]
        staleness_vals = [
            max(0, self.global_model_wrapper.version - int(m.get("version", 0)))
            for m in buf
        ]

        # 버퍼 클리어 (staleness 계산 후!)
        self.plane_buffers[plane_id] = []

        return {
            "state_dict": aggregated,
            "participants": participating,
            "plane_id": plane_id,
            "staleness_values": staleness_vals,
        }

    def _fedorbit_master_upload(self, sat_id, local_wrapper, event_time, temp_model):
        """마스터 위성 GS 접촉: plane ISL 집계 → pseudo-gradient로 글로벌 업데이트"""
        plane_id = self.get_plane_id(sat_id)

        # Intra-plane 집계 강제 실행
        self.last_intra_agg_time[plane_id] = self.start_time
        result = self._fedorbit_try_intra_aggregate(plane_id, event_time, temp_model)

        if result is None:
            return False

        self.aggregation_round += 1
        new_version = round(self.global_model_wrapper.version + 1.0, 1)

        global_sd = self.global_model_wrapper.model_state_dict
        plane_sd = result["state_dict"]

        # pseudo-gradient: Δ = global - plane_aggregated → w_new = w - η·Δ
        eta_g = FEDORBIT_SERVER_LR
        new_sd = OrderedDict()
        for key in global_sd.keys():
            if not self._is_gradient_param(key, global_sd[key]):
                new_sd[key] = global_sd[key].clone()
                continue
            delta = global_sd[key].float() - plane_sd[key].float()
            new_sd[key] = (global_sd[key].float() - eta_g * delta).to(global_sd[key].dtype).cpu()

        self.sim_logger.info(
            f"   🚀 [FedOrbit] Plane {plane_id} Master SAT_{sat_id} → GS Upload "
            f"({len(result['participants'])}개 위성)"
        )

        self._update_global_and_evaluate(
            new_sd, new_version, result["participants"], temp_model,
            staleness_values=result.get("staleness_values", [0]), sim_time=event_time,
            plane_id=plane_id,
        )

        self.metrics.record_gs_contact(sat_id, event_time, "upload")

        # Plane 내 모든 위성 동기화
        for sid in self.get_plane_satellites(plane_id):
            self.satellite_models[sid] = PyTorchModel.from_model(
                self.global_model_net, version=new_version
            )
            self.satellite_last_trained_version[sid] = -1.0
            self.satellite_download_time[sid] = event_time

        return True

    # ================================================================
    # 메인 FL 프로세스
    # ================================================================

    async def manage_fl_process(self):
        self.sim_logger.info(
            f"\n=== 연합 학습 시뮬레이션 시작 [{self.strategy.upper()}] ==="
        )

        # 전체 이벤트 시간순 정렬
        all_events = []
        for sat_id, events in self.check_arr.items():
            for event in events:
                event['sat_id'] = sat_id
                all_events.append(event)
        all_events.sort(key=lambda x: x['start_time'])

        # FedSpace: GS 접촉 스케줄 캐싱
        self.gs_contact_schedule = [e for e in all_events if e['type'] == 'GS_AGGREGATE']

        # 전략별 예상 총 라운드 수 (Cosine LR 스케줄링용)
        gs_event_count = len(self.gs_contact_schedule)
        iot_event_count = len(all_events) - gs_event_count
        if self.strategy == "fedasync":
            self.total_rounds = max(gs_event_count, 1)
        elif self.strategy == "fedbuff":
            self.total_rounds = max(gs_event_count // FEDBUFF_K, 1)
        elif self.strategy in ("fedspace", "fedpda"):
            self.total_rounds = max(gs_event_count // FEDPDA_MIN_BUFFER, 1)
        elif self.strategy == "fedorbit":
            self.total_rounds = max(gs_event_count // max(NUM_PLANES, 1), 1)
        else:
            self.total_rounds = max(gs_event_count, 1)

        self.sim_logger.info(
            f"📅 총 {len(all_events)}개 이벤트 "
            f"(IOT: {iot_event_count}, GS: {gs_event_count}) | "
            f"예상 라운드: {self.total_rounds} | Strategy: {self.strategy}"
        )

        temp_model = create_resnet9(num_classes=self.NUM_CLASSES)

        for i, event in enumerate(all_events):
            sat_id = event['sat_id']
            current_local_wrapper = self.satellite_models[sat_id]
            event_time = event['start_time']

            # ─── IOT_TRAIN (공통) ───
            if event['type'] == 'IOT_TRAIN':
                self.sim_logger.info(
                    f"\n📡 [{event_time.strftime('%m-%d %H:%M')}] "
                    f"SAT_{sat_id} : IoT 학습 ({event['target']})"
                )

                loader_idx = sat_id % len(self.client_subsets)
                dataset = self.client_subsets[loader_idx]

                def seed_worker(worker_id):
                    worker_seed = torch.initial_seed() % 2**32
                    np.random.seed(worker_seed)
                    random.seed(worker_seed)

                train_loader = DataLoader(
                    dataset, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=8, pin_memory=True,
                    worker_init_fn=seed_worker,
                    generator=torch.Generator().manual_seed(SEED)
                )

                # 학습 전 base state 저장 (pseudo-gradient 계산용)
                self.satellite_base_state[sat_id] = {
                    k: v.clone() for k, v in current_local_wrapper.model_state_dict.items()
                }

                current_local_wrapper.to_device(temp_model, device='cpu')
                current_lr = self._get_cosine_lr()

                train_model(
                    model=temp_model,
                    global_state_dict=self.global_model_wrapper.model_state_dict,
                    train_loader=train_loader,
                    epochs=LOCAL_EPOCHS, lr=current_lr,
                    device=self.device, sim_logger=self.sim_logger
                )

                next_version = round(current_local_wrapper.version + 0.1, 1)
                current_local_wrapper = PyTorchModel.from_model(temp_model, version=next_version)
                self.satellite_models[sat_id] = current_local_wrapper
                self.satellite_last_trained_version[sat_id] = next_version

                self.sim_logger.info(
                    f"   ✅ SAT_{sat_id} 학습 완료 (LR: {current_lr:.4f}, v{next_version:.1f})"
                )

                self.metrics.record_train(sat_id, self.get_plane_id(sat_id), event_time)

                # FedOrbit: 학습 완료 시 plane 버퍼에 자동 수집
                if self.strategy == "fedorbit":
                    self._fedorbit_intra_plane_collect(sat_id, current_local_wrapper, event_time)

            # ─── GS_AGGREGATE (전략 분기) ───
            elif event['type'] == 'GS_AGGREGATE':
                self.sim_logger.info(
                    f"\n📡 [{event_time.strftime('%m-%d %H:%M')}] "
                    f"SAT_{sat_id} : 지상국 접속"
                )

                # [공통] 미학습 위성 필터링
                if not self._is_trained_since_global(sat_id):
                    if self.global_model_wrapper.version > current_local_wrapper.version:
                        self.satellite_models[sat_id] = PyTorchModel.from_model(
                            self.global_model_net, version=self.global_model_wrapper.version
                        )
                        self.satellite_download_time[sat_id] = event_time
                        self.metrics.record_gs_contact(sat_id, event_time, "download")
                        self.sim_logger.info(
                            f"   📥 SAT_{sat_id}: 미학습 → v{self.global_model_wrapper.version} 다운로드"
                        )
                    else:
                        self.metrics.record_gs_contact(sat_id, event_time, "skip")
                        self.sim_logger.info(f"   ⏭️ SAT_{sat_id}: 미학습 & 최신 → Skip")

                    # FedOrbit: 마스터인 경우 plane 버퍼가 있으면 업로드 시도
                    if self.strategy == "fedorbit":
                        plane = self.get_plane_id(sat_id)
                        if self.plane_masters.get(plane) == sat_id and len(self.plane_buffers[plane]) > 0:
                            self._fedorbit_master_upload(sat_id, current_local_wrapper, event_time, temp_model)
                    continue

                # [공통] Staleness 초과 → 다운로드만
                if self.global_model_wrapper.version > current_local_wrapper.version + STALENESS_THRESHOLD:
                    self.satellite_models[sat_id] = PyTorchModel.from_model(
                        self.global_model_net, version=self.global_model_wrapper.version
                    )
                    self.satellite_last_trained_version[sat_id] = -1.0
                    self.satellite_download_time[sat_id] = event_time
                    self.metrics.record_gs_contact(sat_id, event_time, "download")
                    self.sim_logger.info(
                        f"   📥 SAT_{sat_id}: Stale → v{self.global_model_wrapper.version} 다운로드"
                    )
                    continue

                # ── FedAsync ──
                if self.strategy == "fedasync":
                    self._fedasync_aggregate(sat_id, current_local_wrapper, temp_model, event_time)

                # ── FedBuff ──
                elif self.strategy == "fedbuff":
                    self._fedbuff_collect(sat_id, current_local_wrapper, event_time)
                    is_last = (i == len(all_events) - 1)
                    if len(self.gs_buffer) >= FEDBUFF_K or is_last:
                        self._fedbuff_flush(temp_model, force_eval=is_last)

                # ── FedSpace ──
                elif self.strategy == "fedspace":
                    self._fedbuff_collect(sat_id, current_local_wrapper, event_time)
                    is_last = (i == len(all_events) - 1)
                    if self._fedspace_should_flush(event_time, len(self.gs_buffer), is_last):
                        self._fedspace_aggregate(temp_model, force_eval=is_last)

                # ── FedPDA (Proposed) ──
                elif self.strategy == "fedpda":
                    self._fedpda_collect(sat_id, current_local_wrapper, event_time)
                    is_last = (i == len(all_events) - 1)
                    if self._fedpda_should_flush(event_time, is_last):
                        self._fedpda_flush(temp_model, force_eval=is_last)

                # ── FedOrbit ──
                elif self.strategy == "fedorbit":
                    plane = self.get_plane_id(sat_id)

                    if self.plane_masters.get(plane) == sat_id:
                        self._fedorbit_master_upload(sat_id, current_local_wrapper, event_time, temp_model)
                    else:
                        # 비마스터: GS 접촉 시 최신 글로벌 다운로드
                        if self.global_model_wrapper.version > current_local_wrapper.version:
                            self.satellite_models[sat_id] = PyTorchModel.from_model(
                                self.global_model_net, version=self.global_model_wrapper.version
                            )
                            self.satellite_download_time[sat_id] = event_time
                            self.sim_logger.info(
                                f"   📥 SAT_{sat_id}: 글로벌 v{self.global_model_wrapper.version} 다운로드"
                            )
                        self.satellite_last_trained_version[sat_id] = -1.0

        # 잔여 버퍼 처리
        if self.strategy in ("fedbuff", "fedspace") and len(self.gs_buffer) > 0:
            self._fedbuff_flush(temp_model, force_eval=True)

        # FedPDA: 잔여 버퍼 처리
        if self.strategy == "fedpda" and len(self.pda_buffer) > 0:
            self._fedpda_flush(temp_model, force_eval=True)

        # FedOrbit: 모든 plane 잔여 버퍼 처리
        if self.strategy == "fedorbit":
            for plane_id in range(NUM_PLANES):
                if len(self.plane_buffers[plane_id]) > 0:
                    self.last_intra_agg_time[plane_id] = self.start_time
                    result = self._fedorbit_try_intra_aggregate(plane_id, self.end_time, temp_model)
                    if result:
                        self.aggregation_round += 1
                        nv = round(self.global_model_wrapper.version + 1.0, 1)
                        global_sd = self.global_model_wrapper.model_state_dict
                        eta_g = FEDORBIT_SERVER_LR
                        new_sd = OrderedDict()
                        for key in global_sd.keys():
                            if not self._is_gradient_param(key, global_sd[key]):
                                new_sd[key] = global_sd[key].clone()
                                continue
                            delta = global_sd[key].float() - result["state_dict"][key].float()
                            new_sd[key] = (global_sd[key].float() - eta_g * delta).to(global_sd[key].dtype).cpu()
                        self._update_global_and_evaluate(
                            new_sd, nv, result["participants"], temp_model, force_eval=True,
                            staleness_values=result["staleness_values"], sim_time=self.end_time,
                            plane_id=plane_id,
                        )

        # ── 메트릭 저장 및 출력 ──
        self.metrics.print_summary(len(self.satellites), logger=self.sim_logger)
        saved = self.metrics.save()
        self.sim_logger.info(f"📁 결과 저장: {saved}")

        self.sim_logger.info(f"\n=== 시뮬레이션 종료 [{self.strategy.upper()}] ===")
        self.sim_logger.info(f"Total Aggregation Rounds: {self.aggregation_round}")
        self.sim_logger.info(f"Final Global Model Accuracy: {self.best_acc:.2f}%")


def parse_tle_epoch(tle_path: str = "constellation.tle") -> datetime:
    """TLE 파일의 첫 번째 위성 epoch을 파싱하여 datetime 반환.
    
    TLE Line 1 형식: 1 NNNNN ... YYDDD.DDDDDDDD ...
    - YY: 2자리 연도 (00-56 → 2000+, 57-99 → 1900+)
    - DDD.DDDDDDDD: 소수점 포함 day-of-year
    """
    with open(tle_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("1 "):
                fields = line.split()
                epoch_str = fields[3]   # "26037.57365555"
                yy = int(epoch_str[:2])
                year = 2000 + yy if yy < 57 else 1900 + yy
                day_frac = float(epoch_str[2:])
                epoch = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_frac - 1)
                return epoch
    raise ValueError(f"TLE epoch을 파싱할 수 없습니다: {tle_path}")


def main():
    try:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
        # 시뮬레이션 시작 시간 결정
        if SIM_START_TIME is not None:
            start_time = SIM_START_TIME
        else:
            # TLE epoch 기반 자동 설정 (SGP4 정확도 + 재현성 보장)
            start_time = parse_tle_epoch("constellation.tle")
        
        duration = timedelta(days=SIM_DURATION_DAYS)
        end_time = start_time + duration

        sim_logger, perf_logger = setup_loggers()
        sim_logger.info(f"시뮬레이션 시간: {start_time.isoformat()} ~ {end_time.isoformat()}")
        sim_logger.info(f"(TLE epoch 기반, {SIM_DURATION_DAYS}일간)")

        sat_manager = Satellite_Manager(start_time, end_time, sim_logger, perf_logger)
        asyncio.run(sat_manager.run())
    except KeyboardInterrupt:
        print("\n시뮬레이션을 종료합니다.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()