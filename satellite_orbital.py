# satellite_orbital.py
# ============================================================
# Orbital Data Center 연합학습 시뮬레이터
# 지상국 없이 궤도 상 Master 위성이 집계 수행
#
# 구조:
#   - 234 Worker: 관측 데이터 수집 → 로컬 학습 → ISL로 Master에 전달
#   - 4 Master (Plane 0,4,8,12): 버퍼 집계 + Master 간 동기화
#   - ISL: 인접 궤도면 상시 연결, hop-by-hop 릴레이
#
# 이벤트 흐름:
#   TRAIN_COMPLETE → MODEL_DELIVERED → MASTER_FLUSH → MASTER_SYNC
# ============================================================

import asyncio
import torch
import numpy as np
import math
import random
import heapq
from datetime import datetime, timedelta, timezone
from collections import defaultdict, OrderedDict, Counter
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from torch.utils.data import DataLoader

from config_orbital import (
    NUM_PLANES, SATS_PER_PLANE, TOTAL_SATS, ORBIT_PERIOD_SEC,
    NUM_MASTERS, MASTER_PLANES, MASTER_SAT_IDS,
    SIM_START_TIME, SIM_DURATION_DAYS,
    OBSERVATION_INTERVAL_SEC, OBSERVATION_JITTER_SEC,
    ISL_HOP_TIME_SEC,
    LOCAL_EPOCHS, FEDPROX_MU, BASE_LR, MIN_LR,
    NUM_CLIENTS, DIRICHLET_ALPHA, BATCH_SIZE, SAMPLES_PER_CLIENT,
    BUFFER_MIN_SIZE, BUFFER_MAX_SIZE, BUFFER_MIN_DIVERSITY, BUFFER_TIMEOUT_SEC,
    SERVER_LR, SERVER_MOMENTUM,
    SYNC_AFTER_FLUSH,
    EVAL_EVERY_N_ROUNDS, STALENESS_THRESHOLD,
    SEED,
)

from ml.data import get_cifar10_loaders
from ml.model import create_resnet9, PyTorchModel
from ml.training import train_model
from ml.metrics import MetricsCollector
from utils.logging_setup import setup_loggers, KST

# ================================================================
# 유틸리티
# ================================================================

def get_plane_id(sat_id: int) -> int:
    return sat_id // SATS_PER_PLANE

def get_position_in_plane(sat_id: int) -> int:
    return sat_id % SATS_PER_PLANE

def find_nearest_master(src_plane: int) -> Tuple[int, int]:
    """src_plane에서 가장 가까운 Master plane과 inter-plane 홉 수를 반환"""
    best_plane, best_hops = MASTER_PLANES[0], NUM_PLANES
    for mp in MASTER_PLANES:
        hops = min(abs(src_plane - mp), NUM_PLANES - abs(src_plane - mp))
        if hops < best_hops:
            best_hops = hops
            best_plane = mp
    return best_plane, best_hops

def compute_delivery_delay(src_sat_id: int, master_sat_id: int) -> float:
    """Worker → Master ISL 전달 지연 (초)"""
    src_plane = get_plane_id(src_sat_id)
    master_plane = get_plane_id(master_sat_id)

    # Inter-plane 홉 (인접 면 릴레이)
    inter_hops = min(abs(src_plane - master_plane),
                     NUM_PLANES - abs(src_plane - master_plane))

    # Intra-plane 홉 (Master 면 도착 후 Master 위성까지)
    # 릴레이는 src와 비슷한 위치의 위성에 도착한다고 가정
    src_idx = get_position_in_plane(src_sat_id)
    master_idx = get_position_in_plane(master_sat_id)
    intra_hops = min(abs(src_idx - master_idx),
                     SATS_PER_PLANE - abs(src_idx - master_idx))

    total_hops = inter_hops + intra_hops
    return total_hops * ISL_HOP_TIME_SEC

def compute_sync_delay() -> float:
    """
    Master 간 ring 동기화 지연 (초).
    N=1이면 동기화 대상이 없으므로 0을 반환 (Master flush가 곧 글로벌 업데이트).
    """
    if NUM_MASTERS <= 1:
        return 0.0
    rounds = math.ceil(NUM_MASTERS / 2)
    spacing = NUM_PLANES / NUM_MASTERS
    return rounds * spacing * ISL_HOP_TIME_SEC

# ================================================================
# 이벤트 타입
# ================================================================
EVT_TRAIN_COMPLETE = "TRAIN_COMPLETE"
EVT_MODEL_DELIVERED = "MODEL_DELIVERED"
EVT_MASTER_SYNC = "MASTER_SYNC"

# ================================================================
# Orbital FL Manager
# ================================================================

class OrbitalFLManager:

    def __init__(self, sim_logger, perf_logger):
        self.start_time = SIM_START_TIME
        self.end_time = SIM_START_TIME + timedelta(days=SIM_DURATION_DAYS)
        self.sim_logger = sim_logger
        self.perf_logger = perf_logger

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sim_logger.info(f"Device: {self.device}")

        # Master/Worker 할당
        self.master_sat_ids = set(MASTER_SAT_IDS)
        self.worker_sat_ids = set(range(TOTAL_SATS)) - self.master_sat_ids
        self.sat_to_master = {}  # sat_id → 담당 master_sat_id
        for sat_id in range(TOTAL_SATS):
            plane = get_plane_id(sat_id)
            nearest_plane, _ = find_nearest_master(plane)
            self.sat_to_master[sat_id] = nearest_plane * SATS_PER_PLANE

        # 모델 상태
        self.satellite_models: Dict[int, PyTorchModel] = {}
        self.satellite_last_trained_version: Dict[int, float] = {}
        self.satellite_base_state: Dict[int, OrderedDict] = {}

        # 글로벌 모델
        self.sim_logger.info("CIFAR-10 데이터셋 로드 중...")
        self.avg_data_count, self.client_subsets, self.val_loader, _ = get_cifar10_loaders(
            num_clients=NUM_CLIENTS,
            dirichlet_alpha=DIRICHLET_ALPHA,
            data_root='./data',
            samples_per_client=SAMPLES_PER_CLIENT
        )
        self.sim_logger.info(f"데이터셋 로드 완료. 위성당 데이터: {self.avg_data_count:.0f}장")

        self.global_model_net = create_resnet9(num_classes=10)
        self.global_model_net.to('cpu')
        self.global_model_wrapper = PyTorchModel.from_model(self.global_model_net, version=0.0)
        self.best_acc = 0.0

        # Master별 버퍼
        self.master_buffers: Dict[int, List[dict]] = {m: [] for m in MASTER_SAT_IDS}
        # Master별 로컬 집계 모델 (동기화 전)
        self.master_local_models: Dict[int, Optional[OrderedDict]] = {m: None for m in MASTER_SAT_IDS}
        self.master_local_versions: Dict[int, float] = {m: 0.0 for m in MASTER_SAT_IDS}
        self.master_contribution_count: Dict[int, int] = {m: 0 for m in MASTER_SAT_IDS}

        # 집계 카운터
        self.aggregation_round = 0
        self.total_rounds = 0
        self.sync_delay = compute_sync_delay()

        # 메트릭
        self.metrics = MetricsCollector(
            strategy="orbital_fl",
            num_planes=NUM_PLANES,
            sats_per_plane=SATS_PER_PLANE,
            sim_start_time=self.start_time,
        )

        # 통계
        self.stats = {
            "total_trains": 0,
            "total_deliveries": 0,
            "total_flushes": 0,
            "total_syncs": 0,
            "delivery_delays": [],
            "flush_sizes": [],
            "planes_per_flush": [],
        }

        self._init_satellites()

    def _init_satellites(self):
        """모든 위성에 초기 글로벌 모델 배포"""
        for sat_id in range(TOTAL_SATS):
            self.satellite_models[sat_id] = PyTorchModel.from_model(
                self.global_model_net, version=0.0
            )
            self.satellite_last_trained_version[sat_id] = -1.0

        self.sim_logger.info(
            f"위성 초기화 완료: {len(self.worker_sat_ids)} Workers + "
            f"{len(self.master_sat_ids)} Masters"
        )
        self.sim_logger.info(
            f"Master 배치: {', '.join(f'SAT_{m}(P{get_plane_id(m)})' for m in sorted(self.master_sat_ids))}"
        )
        for m_id in sorted(self.master_sat_ids):
            assigned = [s for s, m in self.sat_to_master.items()
                        if m == m_id and s not in self.master_sat_ids]
            planes = sorted(set(get_plane_id(s) for s in assigned))
            self.sim_logger.info(
                f"  Master SAT_{m_id}(P{get_plane_id(m_id)}): "
                f"담당 {len(assigned)} Workers, 면 {planes}"
            )

    # ================================================================
    # 이벤트 생성
    # ================================================================

    def _generate_training_events(self) -> List[Tuple[datetime, dict]]:
        """Worker별 주기적 학습 이벤트 생성 (관측 데이터 수집 주기 기반)"""
        random.seed(SEED)
        events = []
        sim_duration_sec = SIM_DURATION_DAYS * 86400

        for sat_id in sorted(self.worker_sat_ids):
            # 초기 오프셋: 위성마다 랜덤하게 분산
            offset = random.uniform(0, OBSERVATION_INTERVAL_SEC)
            t = offset

            while t < sim_duration_sec:
                event_time = self.start_time + timedelta(seconds=t)
                if event_time < self.end_time:
                    events.append((event_time, {
                        "type": EVT_TRAIN_COMPLETE,
                        "sat_id": sat_id,
                    }))
                # 다음 학습까지 간격 (평균 ± 지터)
                interval = OBSERVATION_INTERVAL_SEC + random.uniform(
                    -OBSERVATION_JITTER_SEC, OBSERVATION_JITTER_SEC
                )
                t += max(interval, 1800)  # 최소 30분 간격

        events.sort(key=lambda x: x[0])
        self.sim_logger.info(f"학습 이벤트 생성: {len(events)}개 (7일간)")

        # 총 라운드 추정 (LR 스케줄링용)
        # Worker당 평균 학습 횟수 × avg_workers_per_flush
        avg_trains_per_worker = sim_duration_sec / OBSERVATION_INTERVAL_SEC
        avg_workers_per_flush = (BUFFER_MIN_SIZE + BUFFER_MAX_SIZE) / 2
        self.total_rounds = max(1, int(
            len(self.worker_sat_ids) * avg_trains_per_worker / avg_workers_per_flush
        ))

        return events

    # ================================================================
    # 학습
    # ================================================================

    def _do_local_training(self, sat_id: int, event_time: datetime, temp_model):
        """Worker 로컬 학습 수행"""
        current_wrapper = self.satellite_models[sat_id]

        # 학습 전 글로벌 모델 다운로드 (ISL 상시 연결이므로 즉시)
        if self.global_model_wrapper.version > current_wrapper.version:
            self.satellite_models[sat_id] = PyTorchModel.from_model(
                self.global_model_net, version=self.global_model_wrapper.version
            )
            current_wrapper = self.satellite_models[sat_id]

        loader_idx = sat_id % len(self.client_subsets)
        dataset = self.client_subsets[loader_idx]

        def seed_worker(worker_id):
            np.random.seed(SEED + worker_id)

        train_loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=8, pin_memory=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(SEED)
        )

        # base_state 저장 (pseudo-gradient 계산용)
        self.satellite_base_state[sat_id] = {
            k: v.clone() for k, v in current_wrapper.model_state_dict.items()
        }

        current_wrapper.to_device(temp_model, device='cpu')
        current_lr = self._get_cosine_lr()

        train_model(
            model=temp_model,
            global_state_dict=self.global_model_wrapper.model_state_dict,
            train_loader=train_loader,
            epochs=LOCAL_EPOCHS, lr=current_lr,
            device=self.device, sim_logger=self.sim_logger
        )

        next_version = round(current_wrapper.version + 0.1, 1)
        new_wrapper = PyTorchModel.from_model(temp_model, version=next_version)
        self.satellite_models[sat_id] = new_wrapper
        self.satellite_last_trained_version[sat_id] = next_version

        sim_hours = (event_time - self.start_time).total_seconds() / 3600
        self.sim_logger.info(
            f"   ✅ SAT_{sat_id}(P{get_plane_id(sat_id)}) 학습 완료 "
            f"(LR:{current_lr:.4f}, v{next_version:.1f}, {sim_hours:.1f}h)"
        )
        self.metrics.record_train(sat_id, get_plane_id(sat_id), event_time)
        self.stats["total_trains"] += 1

        return new_wrapper

    # ================================================================
    # Master 버퍼 관리 및 집계
    # ================================================================

    def _deliver_to_master(self, sat_id: int, wrapper: PyTorchModel,
                           event_time: datetime, delivery_time: datetime):
        """Worker 모델을 Master 버퍼에 추가"""
        master_id = self.sat_to_master[sat_id]
        plane_id = get_plane_id(sat_id)

        tau_ver = max(0, self.global_model_wrapper.version - int(wrapper.version))
        s_tau = (1.0 + tau_ver) ** (-0.5)

        if tau_ver > STALENESS_THRESHOLD:
            self.sim_logger.info(
                f"   ⚠️ SAT_{sat_id} 모델 폐기 (staleness={tau_ver} > {STALENESS_THRESHOLD})"
            )
            return False

        loader_idx = sat_id % len(self.client_subsets)
        entry = {
            "sat_id": sat_id,
            "plane_id": plane_id,
            "state_dict": wrapper.model_state_dict,
            "base_state_dict": self.satellite_base_state.get(sat_id, {}),
            "base_version": int(wrapper.version),
            "staleness": tau_ver,
            "s_tau": s_tau,
            "data_count": len(self.client_subsets[loader_idx]),
            "event_time": event_time,
            "delivery_time": delivery_time,
        }
        self.master_buffers[master_id].append(entry)
        self.stats["total_deliveries"] += 1

        buf_size = len(self.master_buffers[master_id])
        self.sim_logger.info(
            f"   📦 SAT_{sat_id}(P{plane_id}) → Master SAT_{master_id}: "
            f"버퍼 {buf_size}, τ={tau_ver}"
        )
        return True

    def _should_flush(self, master_id: int, current_time: datetime, is_last: bool = False) -> bool:
        """Master 버퍼 flush 조건 판단 (FedPDA 기반)"""
        buffer = self.master_buffers[master_id]
        buf_size = len(buffer)

        if buf_size == 0:
            return False
        if is_last and buf_size > 0:
            return True

        # 강제 flush: 버퍼 상한
        if buf_size >= BUFFER_MAX_SIZE:
            return True

        # 면 다양성 확인
        unique_planes = len(set(e["plane_id"] for e in buffer))

        # PRIMARY: 크기 + 다양성
        if buf_size >= BUFFER_MIN_SIZE and unique_planes >= BUFFER_MIN_DIVERSITY:
            return True

        # TIMEOUT: 오래된 항목
        if buffer:
            oldest_age = (current_time - buffer[0]["delivery_time"]).total_seconds()
            if oldest_age >= BUFFER_TIMEOUT_SEC and buf_size >= BUFFER_MIN_SIZE:
                return True

        return False

    def _flush_master(self, master_id: int, temp_model, current_time: datetime,
                      force_eval: bool = False):
        """
        Tier 1: Master 로컬 집계 (FedPDA pseudo-gradient 방식).
        결과를 master_local_models에만 저장하고 글로벌 모델은 건드리지 않는다.
        글로벌 업데이트는 _sync_masters에서 수행된다.
        """
        buffer = self.master_buffers[master_id]
        if not buffer:
            return

        K = len(buffer)
        participating_ids = [e["sat_id"] for e in buffer]
        plane_counts = Counter(e["plane_id"] for e in buffer)
        unique_planes = len(plane_counts)

        self.sim_logger.info(
            f"\n⚡ [Tier 1 Flush - Master SAT_{master_id}] "
            f"K={K}, planes={unique_planes}: {participating_ids}"
        )

        global_sd = self.global_model_wrapper.model_state_dict

        # Staleness 기반 가중치 (plane diversity 제거)
        raw_weights = [e["s_tau"] for e in buffer]
        total_w = sum(raw_weights) or float(K)
        norm_weights = [w / total_w for w in raw_weights]

        # Pseudo-gradient 계산
        eta_g = SERVER_LR
        delta_avg = OrderedDict()
        for key in global_sd.keys():
            if not global_sd[key].is_floating_point():
                delta_avg[key] = None
                continue
            delta = torch.zeros_like(global_sd[key], dtype=torch.float32)
            for e, nw in zip(buffer, norm_weights):
                pseudo_grad = global_sd[key].float() - e["state_dict"][key].float()
                delta += nw * pseudo_grad
            delta_avg[key] = delta

        # Master 로컬 집계 결과: w_local = w_global - η_g × Δ
        new_sd = OrderedDict()
        for key in global_sd.keys():
            if not global_sd[key].is_floating_point():
                new_sd[key] = global_sd[key].clone()
            elif delta_avg.get(key) is not None:
                new_sd[key] = (
                    global_sd[key].float() - eta_g * delta_avg[key]
                ).to(global_sd[key].dtype).cpu()
            else:
                new_sd[key] = global_sd[key].clone()

        self.sim_logger.info(
            f"   📐 η_g={eta_g} (retention={((1-eta_g)*100):.0f}%), K={K} "
            f"→ Master 로컬 모델만 저장 (글로벌 미변경)"
        )

        # Master 로컬 모델 저장 (Sync 대기 중)
        self.master_local_models[master_id] = new_sd
        self.master_contribution_count[master_id] = K

        # 통계
        self.stats["total_flushes"] += 1
        self.stats["flush_sizes"].append(K)
        self.stats["planes_per_flush"].append(unique_planes)

        # 참여한 Worker의 학습 플래그 리셋 (다음 라운드 글로벌 모델 대기)
        for e in buffer:
            self.satellite_last_trained_version[e["sat_id"]] = -1.0
        self.master_buffers[master_id] = []

    # ================================================================
    # Master 간 동기화 (Tier 2)
    # ================================================================

    def _sync_masters(self, temp_model, current_time: datetime, force_eval: bool = False):
        """
        Tier 2: Master 간 글로벌 모델 동기화 + 글로벌 업데이트.
          - 활성 Master 1개: 해당 Master 로컬 모델을 그대로 글로벌로 승격
          - 활성 Master ≥2개: 기여 수 가중 평균으로 글로벌 모델 생성
        실제 경로: 인접 면 Worker 릴레이 (Master 직접 ISL 불가, 동기화 지연은 이벤트 스케줄링에서 반영).
        """
        active_masters = {
            m: sd for m, sd in self.master_local_models.items() if sd is not None
        }

        if len(active_masters) == 0:
            self.sim_logger.info("   🔄 동기화할 활성 Master 없음 (이미 Sync 됨)")
            return

        self.aggregation_round += 1

        if len(active_masters) == 1:
            # 단일 Master → 그대로 글로벌로 승격 (가중 평균 불필요)
            m_id, synced_sd = next(iter(active_masters.items()))
            K = self.master_contribution_count[m_id]
            participants = [m_id]
            self.sim_logger.info(
                f"\n🔄 [Tier 2 Sync Round #{self.aggregation_round}] "
                f"Master SAT_{m_id} 단독 승격 (K={K})"
            )
        else:
            # 다중 Master → 기여 수 가중 평균
            total_contributions = sum(
                self.master_contribution_count[m] for m in active_masters
            ) or len(active_masters)

            global_sd = self.global_model_wrapper.model_state_dict
            synced_sd = OrderedDict()
            for key in global_sd.keys():
                if not global_sd[key].is_floating_point():
                    synced_sd[key] = global_sd[key].clone()
                    continue
                weighted_sum = torch.zeros_like(global_sd[key], dtype=torch.float32)
                for m_id, m_sd in active_masters.items():
                    w = self.master_contribution_count[m_id] / total_contributions
                    weighted_sum += w * m_sd[key].float()
                synced_sd[key] = weighted_sum.to(global_sd[key].dtype).cpu()

            participants = list(active_masters.keys())
            weights_str = ", ".join(
                f"SAT_{m}:{self.master_contribution_count[m]}"
                for m in active_masters
            )
            self.sim_logger.info(
                f"\n🔄 [Tier 2 Sync Round #{self.aggregation_round}] "
                f"{len(active_masters)}개 Master 가중 평균: {weights_str}"
            )

        # 글로벌 모델 업데이트 + 평가
        new_version = round(self.global_model_wrapper.version + 1.0, 1)
        self._update_global_and_evaluate(
            synced_sd, new_version, participants, temp_model,
            force_eval=force_eval, current_time=current_time
        )

        # Master 로컬 모델 초기화
        for m in MASTER_SAT_IDS:
            self.master_local_models[m] = None
            self.master_contribution_count[m] = 0

        self.stats["total_syncs"] += 1
        self.sim_logger.info(
            f"   ✅ 글로벌 v{new_version} 확정 ({len(active_masters)} Masters 반영)"
        )

    # ================================================================
    # 글로벌 모델 평가
    # ================================================================

    def _get_cosine_lr(self) -> float:
        progress = min(self.aggregation_round / max(self.total_rounds, 1), 1.0)
        return MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1 + math.cos(math.pi * progress))

    def _update_global_and_evaluate(self, new_state_dict, new_version,
                                     participating_ids, temp_model,
                                     force_eval=False, current_time=None):
        """글로벌 모델 업데이트 및 평가"""
        self.global_model_net.load_state_dict(new_state_dict)
        g_acc, g_loss = None, None

        if force_eval or (self.aggregation_round % EVAL_EVERY_N_ROUNDS == 0):
            temp_model.load_state_dict(new_state_dict)
            g_acc, g_loss = self._evaluate(temp_model, current_time, new_version)

        self.global_model_wrapper = PyTorchModel.from_model(
            self.global_model_net, version=new_version
        )

        if g_acc is not None:
            self.metrics.record_aggregation(
                round_num=self.aggregation_round,
                sim_time=current_time,
                accuracy=g_acc,
                loss=g_loss,
                participating_ids=participating_ids,
                staleness_values=[],
            )

            if g_acc > self.best_acc:
                self.best_acc = g_acc
                self.sim_logger.info(f"   🏆 새 최고 정확도: {g_acc:.2f}%")

    def _evaluate(self, model, current_time, version):
        """모델 평가"""
        model.to(self.device)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        model.to('cpu')

        sim_hours = (current_time - self.start_time).total_seconds() / 3600 if current_time else 0
        self.sim_logger.info(
            f"   📊 평가 v{version:.1f} ({sim_hours:.1f}h): "
            f"Acc={acc:.2f}%, Loss={avg_loss:.4f}"
        )
        self.perf_logger.info(
            f"{datetime.now(KST).isoformat()},global,0,{version:.2f},"
            f"orbital_fl,{acc:.4f},{avg_loss:.6f},0.0000"
        )
        return acc, avg_loss

    # ================================================================
    # 메인 시뮬레이션 루프
    # ================================================================

    async def run(self):
        self.sim_logger.info("=" * 60)
        self.sim_logger.info("Orbital Data Center FL 시뮬레이션 시작")
        self.sim_logger.info(f"  기간: {SIM_DURATION_DAYS}일")
        self.sim_logger.info(f"  Workers: {len(self.worker_sat_ids)}, Masters: {len(self.master_sat_ids)}")
        self.sim_logger.info(f"  Master 면: {MASTER_PLANES}")
        self.sim_logger.info(f"  ISL 홉 시간: {ISL_HOP_TIME_SEC}초")
        self.sim_logger.info(f"  Master 간 동기화 지연: {self.sync_delay:.1f}초")
        self.sim_logger.info(f"  η_g={SERVER_LR}, β={SERVER_MOMENTUM}")
        self.sim_logger.info("=" * 60)

        # 학습 이벤트 생성
        train_events = self._generate_training_events()

        # 이벤트 큐: (time, seq_num, event_dict)
        event_queue = []
        for seq, (t, evt) in enumerate(train_events):
            heapq.heappush(event_queue, (t, seq, evt))

        seq_counter = len(train_events)
        temp_model = create_resnet9(num_classes=10)
        temp_model.to('cpu')
        processed = 0

        while event_queue:
            event_time, _, event = heapq.heappop(event_queue)

            if event_time >= self.end_time:
                break

            # ── TRAIN_COMPLETE ──
            if event["type"] == EVT_TRAIN_COMPLETE:
                sat_id = event["sat_id"]
                sim_hours = (event_time - self.start_time).total_seconds() / 3600

                self.sim_logger.info(
                    f"\n🛰️ [{event_time.strftime('%m-%d %H:%M')}] ({sim_hours:.1f}h) "
                    f"SAT_{sat_id}(P{get_plane_id(sat_id)}): 관측 데이터 학습"
                )

                # 로컬 학습
                wrapper = self._do_local_training(sat_id, event_time, temp_model)

                # ISL 전달 스케줄링
                master_id = self.sat_to_master[sat_id]
                delay = compute_delivery_delay(sat_id, master_id)
                delivery_time = event_time + timedelta(seconds=delay)
                self.stats["delivery_delays"].append(delay)

                total_hops = round(delay / ISL_HOP_TIME_SEC)
                self.sim_logger.info(
                    f"   📡 → Master SAT_{master_id}(P{get_plane_id(master_id)}): "
                    f"{total_hops}홉, {delay:.1f}초"
                )

                # MODEL_DELIVERED 이벤트 추가
                seq_counter += 1
                heapq.heappush(event_queue, (delivery_time, seq_counter, {
                    "type": EVT_MODEL_DELIVERED,
                    "sat_id": sat_id,
                    "master_id": master_id,
                    "wrapper": wrapper,
                    "train_time": event_time,
                }))

            # ── MODEL_DELIVERED ──
            elif event["type"] == EVT_MODEL_DELIVERED:
                sat_id = event["sat_id"]
                master_id = event["master_id"]

                success = self._deliver_to_master(
                    sat_id, event["wrapper"], event["train_time"], event_time
                )

                if success and self._should_flush(master_id, event_time):
                    self._flush_master(master_id, temp_model, event_time)

                    # Master 간 동기화 스케줄링
                    if SYNC_AFTER_FLUSH:
                        sync_time = event_time + timedelta(seconds=self.sync_delay)
                        seq_counter += 1
                        heapq.heappush(event_queue, (sync_time, seq_counter, {
                            "type": EVT_MASTER_SYNC,
                            "trigger_master": master_id,
                        }))

            # ── MASTER_SYNC ──
            elif event["type"] == EVT_MASTER_SYNC:
                self._sync_masters(temp_model, event_time)

            processed += 1
            if processed % 500 == 0:
                sim_hours = (event_time - self.start_time).total_seconds() / 3600
                self.sim_logger.info(
                    f"\n--- 진행: {processed}개 이벤트, {sim_hours:.1f}h, "
                    f"Round #{self.aggregation_round}, Best Acc: {self.best_acc:.2f}% ---"
                )

        # 잔여 버퍼 flush
        for m_id in MASTER_SAT_IDS:
            if self.master_buffers[m_id]:
                self._flush_master(m_id, temp_model, self.end_time, force_eval=True)

        # 최종 동기화
        if any(sd is not None for sd in self.master_local_models.values()):
            self._sync_masters(temp_model, self.end_time, force_eval=True)

        self._print_summary()

    # ================================================================
    # 결과 출력
    # ================================================================

    def _print_summary(self):
        """시뮬레이션 결과 요약"""
        self.sim_logger.info("\n" + "=" * 60)
        self.sim_logger.info("Orbital FL 시뮬레이션 결과 요약")
        self.sim_logger.info("=" * 60)
        self.sim_logger.info(f"  총 학습 이벤트: {self.stats['total_trains']}")
        self.sim_logger.info(f"  총 전달 이벤트: {self.stats['total_deliveries']}")
        self.sim_logger.info(f"  총 Master flush: {self.stats['total_flushes']}")
        self.sim_logger.info(f"  총 Master sync: {self.stats['total_syncs']}")
        self.sim_logger.info(f"  총 집계 라운드: {self.aggregation_round}")
        self.sim_logger.info(f"  최고 정확도: {self.best_acc:.2f}%")

        if self.stats["delivery_delays"]:
            delays = np.array(self.stats["delivery_delays"])
            self.sim_logger.info(f"\n  [ISL 전달 지연]")
            self.sim_logger.info(f"    평균: {delays.mean():.1f}초")
            self.sim_logger.info(f"    p95:  {np.percentile(delays, 95):.1f}초")
            self.sim_logger.info(f"    max:  {delays.max():.1f}초")

        if self.stats["flush_sizes"]:
            sizes = np.array(self.stats["flush_sizes"])
            planes = np.array(self.stats["planes_per_flush"])
            self.sim_logger.info(f"\n  [Master 집계]")
            self.sim_logger.info(f"    평균 버퍼 크기: {sizes.mean():.1f}")
            self.sim_logger.info(f"    평균 면 다양성: {planes.mean():.1f}")

        self.sim_logger.info(f"\n  [Master 간 동기화]")
        self.sim_logger.info(f"    동기화 지연: {self.sync_delay:.1f}초")
        self.sim_logger.info(f"    경로: ring relay (인접 면 경유, {math.ceil(NUM_MASTERS/2)}라운드)")

        # 메트릭 저장 (NUM_MASTERS별로 분리)
        self.metrics.print_summary(TOTAL_SATS, logger=self.sim_logger)
        results_dir = Path(f"results/orbital_fl_M{NUM_MASTERS}")
        results_dir.mkdir(parents=True, exist_ok=True)
        self.metrics.save(str(results_dir))
        self.sim_logger.info(f"\n  결과 저장: {results_dir}/")
        self.sim_logger.info("=" * 60)


# ================================================================
# 엔트리포인트
# ================================================================

async def main():
    log_dir = Path(f"logs/orbital_fl_M{NUM_MASTERS}")
    log_dir.mkdir(parents=True, exist_ok=True)
    sim_logger, perf_logger = setup_loggers(
        sim_log_path=str(log_dir / "simulation.log"),
        perf_log_path=str(log_dir / "performance.csv"),
    )

    manager = OrbitalFLManager(sim_logger, perf_logger)
    await manager.run()


if __name__ == "__main__":
    asyncio.run(main())
