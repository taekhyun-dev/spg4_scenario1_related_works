# Orbital Data Center FL 시뮬레이터 설계 문서

## 1. 개요

### 1.1 목적

지상국(GS) 없이 **궤도 상에서만** 연합학습을 수행하는 Orbital Data Center 시뮬레이터.
기존 GS 기반 FedPDA+ISL 시뮬레이터(`satellite_fedpda_isl.py`)를 기반으로,
GS 의존성을 완전히 제거하고 Master 위성이 집계를 수행하는 구조로 재설계했다.

### 1.2 기존 시스템과의 차이

| 항목 | 기존 (FedPDA+ISL) | 새 구조 (Orbital FL) |
|------|-------------------|---------------------|
| 집계 주체 | 지상국 (서울) | **Master 위성 4개** |
| 학습 트리거 | IoT 사이트 flyover | **관측 데이터 수집 주기** |
| 모델 전달 | GS 접촉 + ISL 릴레이 (간헐적) | **ISL only (상시 연결)** |
| 전달 지연 | 확률적 (GS 접촉 대기) | **확정적 (홉 수 × 7.1초)** |
| 집계 구조 | 단일 GS | **2-tier (Master 로컬 + Master 간 동기화)** |
| 집계 알고리즘 | FedPDA pseudo-gradient | **동일 (plane-diversity-aware)** |
| 코드 | `satellite_fedpda_isl.py` + `config_fedpda.py` | `satellite_orbital.py` + `config_orbital.py` |

### 1.3 파일 구성

```
config_orbital.py          ← 설정 파라미터
satellite_orbital.py       ← 시뮬레이터 본체
ml/                        ← 기존 ML 모듈 (변경 없음)
  model.py                    ResNet-9, PyTorchModel
  data.py                     CIFAR-10 Dirichlet 샘플링
  training.py                 FedProx 로컬 학습
  aggregation.py              weighted_update, fed_avg
  metrics.py                  MetricsCollector
utils/                     ← 기존 유틸 (logging만 사용)
  logging_setup.py            setup_loggers, KST
```

---

## 2. 콘스텔레이션 및 Master 배치

### 2.1 Walker-Delta 콘스텔레이션

기존 시뮬레이션과 동일한 콘스텔레이션을 사용한다.

| 파라미터 | 값 |
|----------|-----|
| 총 위성 수 | 238 (17 planes × 14 sats) |
| 궤도 고도 | 570 km |
| 경사각 | 80° (Plane 0), 70° (Plane 1~16) |
| Walker F | 1 |
| 궤도 주기 | 95.92 min (~5,760 sec) |

### 2.2 Master 배치 (Option A: 같은 궤도면, 동일 고도)

`master_placement_analysis.md`의 정량 분석 결과에 따라 4개 Master를 선택했다.

**배치 공식**:
```python
MASTER_PLANES = [int(i * 17 / 4) for i in range(4)]  # → [0, 4, 8, 12]
MASTER_SAT_IDS = [p * 14 for p in MASTER_PLANES]      # → [0, 56, 112, 168]
```

각 면의 첫 번째 위성(index 0)을 Master로 지정한다.

### 2.3 Worker → Master 할당

각 Worker는 **가장 가까운 Master plane**에 할당된다.
17개 궤도면의 원형 배치에서 시계/반시계 최단 거리를 기준으로 한다.

```python
def find_nearest_master(src_plane):
    for mp in MASTER_PLANES:
        hops = min(|src_plane - mp|, 17 - |src_plane - mp|)
    return min(hops)
```

| Master (Plane) | 담당 면 | Worker 수 |
|----------------|---------|-----------|
| SAT_0 (P0) | P0, P1, P2, P15, P16 | 69 (5면 × 14 - 1 Master) |
| SAT_56 (P4) | P3, P4, P5, P6 | 55 (4면 × 14 - 1 Master) |
| SAT_112 (P8) | P7, P8, P9, P10 | 55 |
| SAT_168 (P12) | P11, P12, P13, P14 | 55 |
| **합계** | 17면 | **234 Workers** |

---

## 3. ISL 통신 모델

### 3.1 설계 전제

- **인접 궤도면 간 ISL은 항상 가능** (GS 접촉 대기 불필요)
- ISL 홉 시간은 SGP4 실측 기반 **7.1초** (모델전송 2.08s + 오버헤드 5s + 전파지연 ~0.01s)
- Worker → Master 전달 지연은 **확정적** (홉 수에 비례)

### 3.2 전달 지연 계산

```python
def compute_delivery_delay(src_sat_id, master_sat_id):
    # 1) Inter-plane 홉: 원형 토폴로지에서 최단 경로
    inter_hops = min(|src_plane - master_plane|,
                     17 - |src_plane - master_plane|)

    # 2) Intra-plane 홉: Master 면 도착 후 Master 위성까지
    #    릴레이는 src와 비슷한 위치의 위성에 도착
    intra_hops = min(|src_idx - master_idx|,
                     14 - |src_idx - master_idx|)

    return (inter_hops + intra_hops) × 7.1초
```

실측 결과 (234 Workers):
```
평균 전달 지연: 32.9초
p95 전달 지연:  56.8초
최대 전달 지연: 63.9초
```

### 3.3 Master 간 동기화 지연

Master 간 직접 ISL은 불가능하다 (인접 면만 ISL 가능).
중간 궤도면의 Worker 위성을 릴레이로 경유하여 동기화한다.

```python
def compute_sync_delay():
    rounds = ceil(4 / 2)       # = 2라운드 (ring 양방향 전파)
    spacing = 17 / 4           # = 4.25면 (Master 간 간격)
    return 2 × 4.25 × 7.1초   # = 60.4초
```

**총 E2E = avg전달(32.9초) + 동기화(60.3초) = 93.2초**

---

## 4. 이벤트 기반 시뮬레이션 아키텍처

### 4.1 이벤트 타입

기존 시뮬레이터의 `IOT_TRAIN` / `GS_AGGREGATE` 대신
3가지 이벤트로 전체 흐름을 구성한다.

| 이벤트 | 발생 조건 | 처리 내용 |
|--------|-----------|-----------|
| `TRAIN_COMPLETE` | Worker의 관측 데이터 수집 주기 도래 | 로컬 학습 → ISL 전달 스케줄링 |
| `MODEL_DELIVERED` | ISL 전달 지연 경과 | Master 버퍼에 추가 → flush 조건 확인 |
| `MASTER_SYNC` | Master flush 후 동기화 지연 경과 | Master 간 모델 가중 평균 |

### 4.2 이벤트 흐름

```
[TRAIN_COMPLETE]
  Worker 로컬 학습 (FedProx, ResNet-9, 5 epochs)
      ↓
  ISL 전달 지연 계산 (홉 수 × 7.1초)
      ↓
  → MODEL_DELIVERED 이벤트 스케줄링 (현재시각 + 전달지연)
      ↓
[MODEL_DELIVERED]
  Master 버퍼에 모델 추가
  Staleness 확인 (τ > 5.0이면 폐기)
      ↓
  Flush 조건 확인:
    PRIMARY: 버퍼 ≥ 3 AND 면 다양성 ≥ 2
    TIMEOUT: 최오래된 항목 ≥ 1시간 AND 버퍼 ≥ 3
    OVERFLOW: 버퍼 ≥ 15
      ↓
  [조건 충족 시] Master 로컬 집계 (pseudo-gradient)
      ↓
  → MASTER_SYNC 이벤트 스케줄링 (현재시각 + 60.4초)
      ↓
[MASTER_SYNC]
  활성 Master 모델 가중 평균
  글로벌 모델 업데이트 + 평가
  Master 로컬 모델 초기화
```

### 4.3 이벤트 큐 구현

Python `heapq`를 사용한 우선순위 큐로 시간순 처리한다.
이벤트 처리 중 새 이벤트(MODEL_DELIVERED, MASTER_SYNC)가 동적으로 추가된다.

```python
event_queue = []  # (datetime, sequence_number, event_dict)
heapq.heappush(event_queue, (time, seq, event))
```

`sequence_number`는 같은 시각의 이벤트 간 정렬을 보장한다.

---

## 5. 학습 이벤트 생성

### 5.1 기존 시스템: IoT Flyover 기반

기존 시뮬레이터에서는 SGP4 궤도 전파 + Skyfield로 IoT 사이트(Amazon, Great Barrier Reef, Abisko)
상공 통과 시점을 계산하여 학습 이벤트를 생성했다.

### 5.2 새 시스템: 관측 주기 기반

Orbital Data Center에서는 위성이 스스로 데이터를 생성한다(SSA, remote sensing, 지상 관측).
데이터 수집 주기를 기반으로 학습 이벤트를 생성한다.

```python
# 각 Worker별로 독립적 스케줄
for sat_id in worker_sat_ids:
    offset = random(0, INTERVAL)       # 초기 분산
    t = offset
    while t < 7일:
        events.append((start + t초, TRAIN_COMPLETE))
        interval = 14400 + random(-3600, +3600)  # 4시간 ± 1시간
        t += max(interval, 1800)                  # 최소 30분 간격
```

| 파라미터 | 값 | 의미 |
|----------|-----|------|
| `OBSERVATION_INTERVAL_SEC` | 14,400초 (4시간) | 평균 학습 간격 |
| `OBSERVATION_JITTER_SEC` | 3,600초 (±1시간) | 랜덤 지터 |
| 최소 간격 | 1,800초 (30분) | 지터로 인한 과도한 밀집 방지 |
| 초기 오프셋 | random(0, 4시간) | Worker 간 시간 분산 |

7일 시뮬레이션에서 Worker당 약 10~14회 학습 → 총 약 2,300~3,300 학습 이벤트.

---

## 6. 로컬 학습

기존 FedPDA 시뮬레이터의 학습 파이프라인을 그대로 사용한다.

### 6.1 학습 전 글로벌 모델 다운로드

기존 시스템에서는 GS 접촉 시에만 글로벌 모델을 다운로드했다.
새 시스템에서는 ISL이 상시 연결이므로, **학습 시작 시 즉시** 최신 글로벌 모델을 다운로드한다.

```python
if global_version > local_version:
    satellite_models[sat_id] = PyTorchModel.from_model(global_model, global_version)
```

이는 Worker가 항상 최신 글로벌 모델을 기반으로 학습함을 의미하며,
기존 시스템 대비 staleness가 크게 줄어드는 구조적 이점이다.

### 6.2 학습 파라미터

| 파라미터 | 값 | 비고 |
|----------|-----|------|
| 모델 | ResNet-9 | 10 classes (CIFAR-10) |
| 알고리즘 | FedProx | μ = 0.01 |
| 에포크 | 5 | 로컬 에포크 수 |
| 옵티마이저 | SGD | momentum=0.9, weight_decay=1e-4 |
| 학습률 | Cosine annealing | BASE=0.01 → MIN=0.001 |
| 배치 크기 | 128 | |
| 데이터 | CIFAR-10, Dirichlet α=0.5 | 위성당 2,000장 (with replacement) |

### 6.3 base_state 저장

학습 전 모델 상태를 `satellite_base_state[sat_id]`에 저장한다.
이는 Master에서 pseudo-gradient `Δ = w_global - w_trained`를 계산할 때 사용된다.

---

## 7. Master 로컬 집계 (Tier 1)

### 7.1 버퍼 구조

각 Master는 독립적인 버퍼(`master_buffers[master_id]`)를 가진다.
버퍼 항목은 기존 FedPDA의 `pda_buffer` 항목과 동일한 구조이다.

```python
entry = {
    "sat_id": int,           # 학습 Worker
    "plane_id": int,         # Worker의 궤도면
    "state_dict": OrderedDict,  # 학습된 모델 가중치
    "base_state_dict": OrderedDict,  # 학습 전 모델
    "base_version": int,     # 글로벌 모델 버전 (학습 시점)
    "staleness": float,      # τ = global_ver - base_ver
    "s_tau": float,          # (1 + τ)^(-0.5)
    "data_count": int,       # 학습 데이터 수
    "event_time": datetime,  # 학습 완료 시각
    "delivery_time": datetime,  # Master 도착 시각
}
```

### 7.2 Flush 조건

기존 FedPDA의 3단계 flush 조건을 동일하게 적용한다.

```
PRIMARY (크기 + 다양성):
  버퍼 크기 ≥ 3 AND 고유 궤도면 수 ≥ 2

TIMEOUT (시간 기반):
  최오래된 항목 나이 ≥ 3,600초 AND 버퍼 크기 ≥ 3

OVERFLOW (강제):
  버퍼 크기 ≥ 15
```

기존 FedPDA에서 TIMEOUT이 1,800초였던 것을 3,600초로 늘렸다.
이는 GS 접촉 없이 ISL로 모델이 도착하므로, 도착 빈도가 더 균등해져
다양성 확보에 더 많은 시간을 허용할 수 있기 때문이다.

### 7.3 집계 알고리즘 (Pseudo-gradient + Plane-diversity 가중치)

기존 FedPDA의 집계 알고리즘을 그대로 사용한다.

**1단계: Plane-diversity-aware 가중치 계산**
```
c_p(i) = 면 plane_id의 버퍼 내 항목 수
raw_weight(i) = s_tau(i) / c_p(i)
norm_weight(i) = raw_weight(i) / Σraw_weight
```

같은 면에 여러 Worker가 있으면 면 내 기여가 균등 분배되어 면 간 균형이 유지된다.

**2단계: Pseudo-gradient 계산**
```
Δ_avg = Σ norm_weight(i) × (w_global - w_trained(i))
```

**3단계: 글로벌 모델 업데이트**
```
w_new = w_global - η_g × Δ_avg
      = (1 - η_g) × w_global + η_g × w_aggregated
      = 0.3 × w_global + 0.7 × w_aggregated    (η_g = 0.7)
```

글로벌 모델의 30%가 보존되어 non-IID 환경에서의 진동을 억제한다.

---

## 8. Master 간 동기화 (Tier 2)

### 8.1 동기화 트리거

`SYNC_AFTER_FLUSH = True`: Master가 로컬 flush를 수행할 때마다 동기화를 트리거한다.
실제 동기화는 flush 시점 + 동기화 지연(60.4초) 후에 `MASTER_SYNC` 이벤트로 발생한다.

### 8.2 동기화 경로

Master 간 직접 ISL은 불가능하다.
인접 면 Worker 위성을 경유하는 릴레이로 모델을 교환한다.

```
Master0(P0) → [P1→P2→P3] → Master1(P4) → [P5→P6→P7] → Master2(P8)
                                                                ↓
Master3(P12) ← [P11←P10←P9] ← ─────────────────────────────────┘
```

Ring topology 양방향 전파: ceil(4/2) = 2 라운드.
1 라운드 = 4.25면 × 7.1초 = 30.2초.
총 동기화 지연 = 2 × 30.2 = **60.4초**.

### 8.3 동기화 알고리즘

활성 Master(flush를 수행한 Master)들의 로컬 집계 모델을 **기여 수 가중 평균**으로 합산한다.

```python
for key in global_sd:
    synced[key] = Σ (master_contribution_count[m] / total) × master_model[m][key]
```

Master A가 10개 모델로 집계하고 Master B가 5개로 집계했다면,
A의 가중치는 10/15, B는 5/15로 기여 비례 평균을 낸다.

동기화 후 모든 Master의 로컬 모델은 초기화되고,
새 글로벌 모델이 버전 +1.0으로 업데이트된다.

---

## 9. 기존 FedPDA+ISL과의 주요 코드 차이

### 9.1 제거된 구성 요소

| 기존 코드 | 역할 | 제거 사유 |
|-----------|------|-----------|
| `check_iot_comm()` | IoT flyover 계산 | 관측 주기 기반으로 대체 |
| `check_gs_comm()` | GS 접촉 스케줄 계산 | GS 제거 |
| `propagate_orbit()` | SGP4 궤도 전파 | 학습 트리거가 flyover 기반이 아님 |
| `load_constellation()` | TLE 로드 | 동일 사유 |
| `gs_buffer` | GS 집계 버퍼 | Master별 버퍼로 대체 |
| `isl_relay_buffer` | ISL 릴레이 대기 버퍼 | 상시 연결이므로 불필요 |
| `_fedpda_isl_*` 메서드 | GS 접촉 기반 릴레이 | 확정적 ISL 전달로 대체 |
| FedAsync/FedBuff/FedSpace/FedOrbit | 비교 전략 | Orbital FL 전용 단일 전략 |

### 9.2 새로 추가된 구성 요소

| 새 코드 | 역할 |
|---------|------|
| `find_nearest_master()` | Worker → 가장 가까운 Master 찾기 |
| `compute_delivery_delay()` | ISL 전달 지연 계산 (확정적) |
| `compute_sync_delay()` | Master 간 ring 동기화 지연 |
| `_generate_training_events()` | 관측 주기 기반 학습 이벤트 생성 |
| `master_buffers` | Master별 독립 버퍼 (기존: 단일 pda_buffer) |
| `master_local_models` | Master 로컬 집계 결과 (동기화 전) |
| `_flush_master()` | Master 로컬 집계 |
| `_sync_masters()` | Master 간 글로벌 동기화 (Tier 2) |
| `heapq` 기반 이벤트 큐 | 동적 이벤트 추가 지원 |

### 9.3 유지된 구성 요소

| 기존 코드 | 유지 사유 |
|-----------|-----------|
| `ml/model.py` (ResNet-9, PyTorchModel) | 동일 모델 사용 |
| `ml/data.py` (CIFAR-10 Dirichlet) | 동일 데이터 분배 방식 |
| `ml/training.py` (FedProx) | 동일 로컬 학습 알고리즘 |
| `ml/metrics.py` (MetricsCollector) | 동일 평가 지표 |
| Staleness 함수 `s(τ) = (1+τ)^(-0.5)` | 동일 가중치 함수 |
| Plane-diversity-aware 가중치 | FedPDA 핵심 알고리즘 |
| Pseudo-gradient 집계 + η_g=0.7 | FedPDA 핵심 알고리즘 |

---

## 10. 설정 파라미터 요약

### `config_orbital.py` 전체 파라미터

| 카테고리 | 파라미터 | 값 | 의미 |
|----------|----------|-----|------|
| **콘스텔레이션** | NUM_PLANES | 17 | 궤도면 수 |
| | SATS_PER_PLANE | 14 | 면당 위성 수 |
| | TOTAL_SATS | 238 | 총 위성 수 |
| **Master** | NUM_MASTERS | 4 | Master 위성 수 |
| | MASTER_PLANES | [0, 4, 8, 12] | Master 배치 면 |
| | MASTER_SAT_IDS | [0, 56, 112, 168] | Master 위성 ID |
| **시뮬레이션** | SIM_START_TIME | 2026-02-18 00:00 UTC | 시작 시각 |
| | SIM_DURATION_DAYS | 7 | 시뮬레이션 기간 |
| **관측/학습** | OBSERVATION_INTERVAL_SEC | 14,400 (4시간) | 학습 평균 간격 |
| | OBSERVATION_JITTER_SEC | 3,600 (±1시간) | 랜덤 지터 |
| **ISL** | ISL_HOP_TIME_SEC | 7.1 | 1홉 총 시간 (실측) |
| | ISL_BANDWIDTH_MBPS | 100 | ISL 대역폭 |
| | MODEL_SIZE_MB | 26 | ResNet-9 모델 크기 |
| **로컬 학습** | LOCAL_EPOCHS | 5 | 로컬 에포크 |
| | FEDPROX_MU | 0.01 | FedProx 정규화 |
| | BASE_LR / MIN_LR | 0.01 / 0.001 | 학습률 범위 |
| **데이터** | DIRICHLET_ALPHA | 0.5 | Non-IID 강도 |
| | SAMPLES_PER_CLIENT | 2,000 | 위성당 데이터 |
| **집계** | BUFFER_MIN_SIZE | 3 | 최소 flush 크기 |
| | BUFFER_MAX_SIZE | 15 | 강제 flush 상한 |
| | BUFFER_MIN_DIVERSITY | 2 | 최소 면 다양성 |
| | BUFFER_TIMEOUT_SEC | 3,600 | 타임아웃 (1시간) |
| | SERVER_LR | 0.7 | η_g (글로벌 30% 보존) |
| | SERVER_MOMENTUM | 0.0 | β (모멘텀 비활성화) |
| **동기화** | SYNC_AFTER_FLUSH | True | flush 후 즉시 동기화 |
| **평가** | EVAL_EVERY_N_ROUNDS | 5 | 평가 주기 |
| | STALENESS_THRESHOLD | 5.0 | 모델 폐기 기준 |

---

## 11. 실행 방법

```bash
pyenv shell kepler_propagation
python satellite_orbital.py
```

### 출력 경로

```
logs/orbital_fl/
  simulation.log             ← 시뮬레이션 전체 로그
  performance.csv            ← 라운드별 정확도/손실

results/orbital_fl/
  accuracy_history.json      ← 정확도 추이
  metrics_summary.json       ← 요약 통계
```

---

## 12. 향후 확장 가능성

| 항목 | 현재 | 확장 방향 |
|------|------|-----------|
| 학습 데이터 | CIFAR-10 (proxy) | SSA/remote sensing 실제 데이터 |
| 학습 트리거 | 고정 주기 | 궤도 전파 기반 관측 윈도우 |
| Master 수 | 4개 (고정) | 동적 Master 선출 |
| 동기화 방식 | 가중 평균 | Gossip protocol, All-reduce |
| 모델 | ResNet-9 | remote sensing 전용 모델 |
| Fault tolerance | 미구현 | Master 장애 감지 + 재할당 |
