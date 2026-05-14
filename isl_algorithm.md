# ISL 통신 알고리즘 정리

## 1. 개요

본 프로젝트의 Orbital Data Center FL 구조(`satellite_orbital.py`)에서
ISL(Inter-Satellite Link)은 **모든 모델 전달의 유일한 채널**이다.
지상국이 없으므로 Worker→Master 학습 결과 전달, Master→Master 글로벌 동기화가
모두 ISL을 통해 이루어진다.

### 1.1 핵심 설계 원칙

1. **인접 궤도면 간 ISL은 항상 가능** (상시 연결 가정)
2. **ISL 전달 지연은 확정적** (홉 수에만 의존)
3. **Master 간 직접 ISL은 불가능** (항상 인접 면 릴레이)
4. **확률적 접촉 대기 없음** (GS 기반 시스템과 근본적 차이)

### 1.2 기존 FedPDA+ISL과의 차이

두 시스템 모두 인접 궤도면 간 ISL이 상시 가능하다고 가정하지만,
**문제 구조 자체가 다르기 때문에** 알고리즘이 근본적으로 다르다.

| 항목 | 기존 FedPDA+ISL (GS 기반) | 본 프로젝트 (Orbital) |
|------|--------------------------|---------------------|
| ISL 가용성 | 인접 면 상시 가능 | 인접 면 상시 가능 (동일) |
| 릴레이 목적지 | **GS 접촉이 가장 빠른 위성** (동적) | **담당 Master 위성** (정적) |
| 경로 결정 방식 | Greedy hop-by-hop | 정적 최단 경로 |
| 핵심 비교 | "내 GS 대기 시간" vs "ISL 홉 + 다른 위성 GS 대기" | "Master까지 최단 홉 시간" |
| 이득 판정 | `gain > 0`이면 릴레이 사용 ¹ | 해당 개념 없음 (무조건 최단 경로) |
| 전달 지연 | 확률적 (GS 접촉 대기 포함) | 확정적 (홉 × 7.1초) |
| 릴레이 실패 | 가능 (gain ≤ 0인 경우) | 없음 |

¹ 기존 코드의 `FEDPDA_ISL_MIN_GAIN_SEC = 300`은 수학적 근거가 없는 보수적 필터로
판단되어 **0으로 수정됨**. relay_wait에 이미 ISL 비용이 반영되어 있으므로
gain > 0이면 릴레이가 항상 이득이다.

### 1.3 두 알고리즘이 근본적으로 다른 이유

| 전제 | FedPDA+ISL | Orbital FL |
|------|-----------|-----------|
| 목적지가 동적으로 바뀌는가 | ✅ GS 접촉 스케줄에 따라 | ❌ Master 고정 |
| 홉 수를 늘려서 이득이 가능한가 | ✅ 다른 위성이 더 빨리 GS 접촉 가능 | ❌ 최단 경로가 항상 최적 |
| 릴레이를 안 쓰는 선택지가 있는가 | ✅ 직접 GS 대기 가능 | ❌ ISL이 유일한 채널 |

→ Orbital FL에서는 **선택의 여지가 없으므로** Greedy 탐색이 불필요하다.

---

## 2. ISL 홉 시간 모델

### 2.1 1홉 시간 구성

```
t_hop = t_transfer + t_overhead + t_propagation
```

| 구성 요소 | 산출 근거 | 값 |
|-----------|-----------|-----|
| **모델 전송 시간** `t_transfer` | 26MB × 8 / 100Mbps | **2.08초** |
| **프로토콜 오버헤드** `t_overhead` | 핸드셰이크 + ACK | 5.00초 (가정) |
| **전파 지연** `t_propagation` | 거리 / 광속 | ~0.01초 |
| **합계** | | **≈ 7.1초** |

`config_orbital.py`에서 `ISL_HOP_TIME_SEC = 7.1`로 설정.

### 2.2 Intra-plane vs Inter-plane 홉 시간

SGP4 실측 결과 두 경우 모두 7.1초로 거의 동일하다.

| | 위성 간 거리 | 전파 지연 | 1홉 총 시간 |
|--|--|--|--|
| Intra-plane (같은 면) | 3,089 km | 10.3 ms | 7.09초 |
| Inter-plane (인접 면) | 1,096 km | 3.7 ms | 7.08초 |

전파 지연이 전체의 0.05~0.15%에 불과하여 무시 가능하다.
따라서 전달 시간은 **순수하게 홉 수에만 비례**한다.

---

## 3. 토폴로지

### 3.1 위성 연결 그래프

**Intra-plane**: 같은 면 내 14개 위성이 ring topology를 형성한다.

```
각 면: sat0 ↔ sat1 ↔ sat2 ↔ ... ↔ sat13 ↔ sat0
```

**Inter-plane**: 각 위성은 인접 면의 위성과 연결된다 (원형 17면).

```
P0 ↔ P1 ↔ P2 ↔ ... ↔ P16 ↔ P0
```

### 3.2 Master 위성 배치

17개 면 중 4개 면(Plane 0, 4, 8, 12)의 **첫 번째 위성(index 0)**이 Master이다.

```python
MASTER_SAT_IDS = [0, 56, 112, 168]
MASTER_PLANES  = [0, 4, 8, 12]
```

### 3.3 Worker → Master 담당 할당

각 Worker는 **가장 가까운 Master plane**에 정적으로 할당된다.

```python
def find_nearest_master(src_plane):
    best_plane, best_hops = MASTER_PLANES[0], NUM_PLANES
    for mp in MASTER_PLANES:
        hops = min(|src_plane - mp|, 17 - |src_plane - mp|)
        if hops < best_hops:
            best_hops = hops
            best_plane = mp
    return best_plane, best_hops
```

| Master | 담당 면 | Worker 수 |
|--------|---------|-----------|
| SAT_0 (P0) | P15, P16, P0, P1, P2 | 69 |
| SAT_56 (P4) | P3, P4, P5, P6 | 55 |
| SAT_112 (P8) | P7, P8, P9, P10 | 55 |
| SAT_168 (P12) | P11, P12, P13, P14 | 55 |

---

## 4. Tier 1: Worker → Master 전달 (`compute_delivery_delay`)

### 4.1 경로 구조

Worker가 학습을 완료하면 담당 Master로 모델을 전달한다.
경로는 **2단계**로 분해된다:

```
Worker(P_src, idx_src)
       │
       ├─ [Inter-plane 릴레이] ── 인접 면 홉-바이-홉 전파
       │                           (src 면 → ... → Master 면)
       ↓
Master 면의 "idx_src와 같은 위치" 위성 도착
       │
       ├─ [Intra-plane 릴레이] ── 같은 면 내 ring topology 전파
       │                           (idx_src → idx_master)
       ↓
Master 위성 도착
```

### 4.2 홉 수 계산

#### (1) Inter-plane 홉

원형 17면에서 시계/반시계 최단 거리:

```python
inter_hops = min(|src_plane - master_plane|,
                 17 - |src_plane - master_plane|)
```

| 상황 | 홉 수 |
|------|-------|
| Worker가 Master 면에 있음 | 0 |
| 인접 면 | 1 |
| Master 4개 균등 배치 시 최대값 | 2~3 (담당 범위 중앙) |

#### (2) Intra-plane 홉

Inter-plane 릴레이 도착 시 **src와 같은 index 위치의 위성**에 도착한다고 가정한다
(Walker F=1의 미세 위상 오프셋 무시).

그 위성에서 Master까지 ring topology로 전달:

```python
intra_hops = min(|src_idx - master_idx|,
                 14 - |src_idx - master_idx|)
```

Master는 index 0이므로:
```python
intra_hops = min(src_idx, 14 - src_idx)
```

| src_idx | intra_hops |
|---------|-----------|
| 0 (Master와 같은 위치) | 0 |
| 1 또는 13 | 1 |
| 7 (정반대편) | 7 (최대) |

#### (3) 총 전달 시간

```python
def compute_delivery_delay(src_sat_id, master_sat_id):
    inter_hops = ...
    intra_hops = ...
    total_hops = inter_hops + intra_hops
    return total_hops * 7.1초
```

### 4.3 구체적 예시

**예 1: SAT_50 (Plane 3, idx 8) → Master SAT_56 (Plane 4, idx 0)**

```
Inter: |3-4| = 1, 17-1 = 16  →  min = 1홉
Intra: |8-0| = 8, 14-8 = 6   →  min = 6홉

total = 1 + 6 = 7홉 × 7.1초 = 49.7초
```

**예 2: SAT_30 (Plane 2, idx 2) → Master SAT_0 (Plane 0, idx 0)**

```
Inter: |2-0| = 2, 17-2 = 15  →  min = 2홉
Intra: |2-0| = 2, 14-2 = 12  →  min = 2홉

total = 2 + 2 = 4홉 × 7.1초 = 28.4초
```

**예 3: SAT_168의 담당 Worker 중 가장 먼 SAT_210 (Plane 15, idx 0) → Master SAT_168 (Plane 12, idx 0)**

```
Inter: |15-12| = 3, 17-3 = 14  →  min = 3홉
Intra: |0-0| = 0                →  0홉

total = 3홉 × 7.1초 = 21.3초
```

### 4.4 전체 Worker 전달 지연 분포 (234 Workers 기준)

```
평균 전달 지연: 32.9초
p95 전달 지연:  56.8초
최대 전달 지연: 63.9초
```

---

## 5. Tier 2: Master 간 동기화 (`compute_sync_delay`)

### 5.1 설계 제약

**N ≥ 2인 경우, Master 간 직접 ISL은 불가능**하다:
- Master들은 서로 다른 궤도면에 있음 (예: N=4 → P0, P4, P8, P12)
- 본 프로젝트는 **인접 면 간 ISL만 가능**으로 가정
- 예: Master P0 ↔ Master P4는 3면 떨어져 있어 직접 링크 불가

**N = 1인 경우**, 동기화 자체가 불필요하다 (Master가 1개뿐):
- `compute_sync_delay()`가 0을 반환
- Master flush 결과가 즉시 글로벌 모델로 승격
- `_sync_masters()`가 호출되지만 단일 Master 승격 처리

### 5.2 릴레이 경로

중간 궤도면의 Worker 위성이 **릴레이 노드**로 동작한다.

```
Master_0(P0) → [P1] → [P2] → [P3] → Master_1(P4)
                ↑      ↑      ↑
          Worker 위성들이 ISL 중계
```

각 중간 면에서 "Master와 같은 위치(index 0)의 Worker"가 자동으로 릴레이 역할을 수행한다.

### 5.3 Ring All-Reduce 동기화

4개 Master는 ring topology(`M0 ↔ M1 ↔ M2 ↔ M3 ↔ M0`)를 형성하며
**양방향 병렬 전파**로 동기화한다.

#### 라운드 수

```python
rounds = ceil(N_masters / 2) = ceil(4/2) = 2
```

이유: ring 양방향으로 동시에 전파하면 최원거리 Master까지 `N/2` 라운드 필요.

#### 1라운드당 홉 수

Master 간 면 간격:
```python
spacing = NUM_PLANES / NUM_MASTERS = 17 / 4 = 4.25
```

1라운드에 인접 Master로 이동 = 4.25개 인접 면 릴레이 필요.

#### 총 동기화 시간

```python
def compute_sync_delay():
    if NUM_MASTERS <= 1:
        return 0.0  # N=1 특례: 동기화 대상이 없으므로 0
    rounds = math.ceil(NUM_MASTERS / 2)       # 2
    spacing = NUM_PLANES / NUM_MASTERS         # 4.25
    return rounds * spacing * ISL_HOP_TIME_SEC # 2 × 4.25 × 7.1 = 60.3초
```

| NUM_MASTERS | rounds | spacing | sync_delay |
|-------------|--------|---------|-----------|
| **1** | — | — | **0초** (특례) |
| 2 | 1 | 8.5 | 60.4초 |
| **4** | 2 | 4.25 | **60.3초** (현재 기본값) |
| 6 | 3 | 2.83 | 60.4초 |
| 17 | 9 | 1.0 | 63.9초 |

흥미로운 점: Master 수와 무관하게 sync_delay가 거의 일정하다.
이는 `(N/2) × (17/N) = 17/2 = 8.5`로 N이 상쇄되기 때문이다.

### 5.4 동기화 타임라인

```
t=0        Master A flush (Tier 1)
           → master_local_models[A] 저장
           → MASTER_SYNC 이벤트 예약 (t=60.3)

t=0.5s     Master B flush (Tier 1)
           → master_local_models[B] 저장
           → MASTER_SYNC 이벤트 예약 (t=60.8)

t=60.3s    MASTER_SYNC 발동
           → 활성: A, B → 가중 평균 → 글로벌 v+1.0
           → master_local_models 모두 초기화

t=60.8s    MASTER_SYNC 발동
           → 활성: 없음 (이미 처리됨) → skip
```

---

## 6. 전체 E2E 지연

```
Worker 학습 완료
    ↓
[Tier 1] Worker → Master ISL 전달
    ↓
Master 버퍼 집계 (flush 조건 충족 시)
    ↓
[Tier 2] Master 간 ring 동기화 (sync_delay 후)
    ↓
글로벌 모델 v+1.0 확정
```

### 6.1 NUM_MASTERS별 E2E 비교

| 항목 | NUM_MASTERS = 1 | NUM_MASTERS = 4 |
|------|-----------------|-----------------|
| Worker 수 | 237 | 234 |
| 최대 inter-plane 홉 | 8홉 | 3홉 |
| avg 전달 지연 | **55.2초** | 32.9초 |
| p95 전달 지연 | 92.3초 | 56.8초 |
| max 전달 지연 | **106.5초** | 63.9초 |
| 동기화 지연 | 0초 (불필요) | 60.3초 |
| **총 E2E (avg)** | **55.2초** | **93.2초** |
| 통신 부하 분산 | ✗ (1개 위성에 집중) | ✓ |
| SPOF | ✓ (단일 장애점) | ✗ |

**관찰**:
- N=1이 평균 E2E는 빠르지만 (55.2 < 93.2초), max 전달은 더 오래 걸린다 (106.5 > 63.9초).
- N=1은 동기화 비용이 없는 대신 worst-case worker가 더 멀리 떨어져 있다.
- 정확도 비교는 실제 시뮬레이션 실행으로 확인 필요.

### 6.2 결과 디렉토리 분리

NUM_MASTERS별로 출력 경로가 자동 분리되어 비교 실험 가능:

```
results/orbital_fl_M1/    ← NUM_MASTERS=1 결과
results/orbital_fl_M4/    ← NUM_MASTERS=4 결과
logs/orbital_fl_M1/
logs/orbital_fl_M4/
```

---

## 7. 구현 코드 위치

| 기능 | 함수 / 상수 | 파일 |
|------|-------------|------|
| ISL 홉 시간 상수 | `ISL_HOP_TIME_SEC = 7.1` | `config_orbital.py` |
| Master 수 | `NUM_MASTERS` (현재 4) | `config_orbital.py` |
| Master 배치 | `MASTER_PLANES`, `MASTER_SAT_IDS` (자동 계산) | `config_orbital.py` |
| 가장 가까운 Master 탐색 | `find_nearest_master()` | `satellite_orbital.py` |
| Worker → Master 전달 지연 | `compute_delivery_delay()` | `satellite_orbital.py` |
| Master 간 동기화 지연 (N=1 특례 포함) | `compute_sync_delay()` | `satellite_orbital.py` |
| 학습 이벤트 생성 | `_generate_training_events()` | `satellite_orbital.py` |
| 이벤트 큐 (heapq) | `run()` 메서드 내 | `satellite_orbital.py` |
| Master 버퍼 집계 (Tier 1) | `_flush_master()` | `satellite_orbital.py` |
| Master 간 동기화 (Tier 2) | `_sync_masters()` | `satellite_orbital.py` |
| 결과 출력 경로 | `results/orbital_fl_M{N}/` | `_print_summary()` |
| 로그 출력 경로 | `logs/orbital_fl_M{N}/` | `main()` |

---

## 8. 이벤트 기반 ISL 처리 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│ TRAIN_COMPLETE (Worker 학습 완료)                                │
│   ┌─────────────────────────────────────┐                      │
│   │ 1. _do_local_training()             │                      │
│   │ 2. master_id = sat_to_master[s]     │                      │
│   │ 3. delay = compute_delivery_delay() │                      │
│   │ 4. MODEL_DELIVERED 이벤트 예약       │                      │
│   │    (현재 시각 + delay)              │                      │
│   └─────────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
                       ↓ delay 후
┌─────────────────────────────────────────────────────────────────┐
│ MODEL_DELIVERED (Master 버퍼 도착)                               │
│   ┌─────────────────────────────────────┐                      │
│   │ 1. _deliver_to_master()             │                      │
│   │    → staleness 체크                 │                      │
│   │    → buffer에 추가                   │                      │
│   │ 2. _should_flush() 판단             │                      │
│   │ 3. [충족 시] _flush_master()         │                      │
│   │    → Master 로컬 모델 저장           │                      │
│   │ 4. MASTER_SYNC 이벤트 예약           │                      │
│   │    (현재 시각 + sync_delay)         │                      │
│   └─────────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
                       ↓ sync_delay(60.3s) 후
┌─────────────────────────────────────────────────────────────────┐
│ MASTER_SYNC (Master 간 동기화)                                   │
│   ┌─────────────────────────────────────┐                      │
│   │ 1. 활성 Master 수집                  │                      │
│   │ 2. [단일] 그대로 승격                │                      │
│   │    [다중] 기여 수 가중 평균          │                      │
│   │ 3. 글로벌 모델 업데이트 + 평가        │                      │
│   │ 4. master_local_models 초기화        │                      │
│   └─────────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. 제약과 단순화 가정

### 9.1 단순화한 부분

1. **Walker F=1의 위상 오프셋 무시**
   - 인접 면 릴레이 시 "같은 index 위성"에 도착한다고 가정
   - 실제로는 F=1로 ~1.51°씩 어긋나 있음
   - 영향: 수십 ms 수준의 전파 지연 차이, 무시 가능

2. **ISL 링크 장애 없음**
   - 하드웨어 장애, 빔 차폐 등 고려 안 함
   - 실제 LEO에서는 극지 통과 시 ISL 끊김 가능

3. **대역폭 경합 없음**
   - 동시에 여러 모델이 같은 링크를 사용해도 지연 증가 없음
   - 실제로는 큐잉 지연 발생 가능

4. **동기화 경로가 정확히 ring topology**
   - 실제로는 Master 간 최단 경로가 ring이 아닐 수도 있음
   - 현재 Master 배치(P0/4/8/12)에서는 ring이 최적

### 9.2 확장 가능성

| 항목 | 현재 구현 | 확장 방향 |
|------|----------|-----------|
| 경로 결정 | 정적 최단 경로 | 동적 부하 분산 |
| 장애 처리 | 없음 | 링크 실패 감지 + 재라우팅 |
| 대역폭 | 상수 가정 | 링크별 가용 대역폭 추적 |
| Master 동기화 | Ring all-reduce | Gossip, hierarchical |
| 담당 할당 | 정적 (plane 기반) | 동적 부하 기반 재할당 |

---

## 10. 요약

본 프로젝트의 ISL 알고리즘은 **확정적·단순한 정적 라우팅** 기반이다:

1. **홉 시간 상수** 7.1초 (SGP4 실측)
2. **Worker → Master**: `(inter_hops + intra_hops) × 7.1초`
3. **Master → Master**: `⌈N/2⌉ × (NUM_PLANES/N) × 7.1초` (N=1이면 0)
4. **이벤트 기반 스케줄링**: TRAIN_COMPLETE → MODEL_DELIVERED → MASTER_SYNC

### 기존 FedPDA+ISL과의 본질적 차이

두 시스템 모두 인접 면 ISL 상시 가능을 가정하지만, **문제 구조가 다르다**:
- FedPDA+ISL: GS 접촉이 동적 목적지 → Greedy 탐색 필요
- Orbital FL: Master가 정적 목적지 → 최단 경로가 항상 최적

따라서 Orbital FL은 Greedy hop-by-hop 알고리즘을 **사용하지 않는다**.

### 핵심 수치

| 시나리오 | 평균 E2E |
|---------|---------|
| NUM_MASTERS = 1 | **55.2초** (sync 불필요) |
| NUM_MASTERS = 4 | **93.2초** (sync 60.3초 포함) |

NUM_MASTERS는 `config_orbital.py`에서 변경 가능하며, 결과는 자동으로
`results/orbital_fl_M{N}/` 경로에 분리되어 저장된다.
