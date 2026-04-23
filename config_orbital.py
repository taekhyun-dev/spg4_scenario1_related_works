# config_orbital.py
# ============================================================
# Orbital Data Center 연합학습 설정
# 지상국 없음 — Master 위성이 궤도 상에서 집계
# 570km Walker-Delta: 17 planes × 14 sats = 238 satellites
#   - 4 Master (Plane 0, 4, 8, 12)
#   - 234 Worker
# ============================================================
from datetime import datetime, timezone

# === Constellation ===
NUM_PLANES = 17
SATS_PER_PLANE = 14
TOTAL_SATS = NUM_PLANES * SATS_PER_PLANE      # 238
ORBIT_ALTITUDE_KM = 570
ORBIT_PERIOD_SEC = 5760                         # ~96분

# === Master 위성 배치 (Option A: 같은 궤도면, 동일 고도) ===
NUM_MASTERS = 4
# 균등 배치: int(i * 17/4) → Plane 0, 4, 8, 12
MASTER_PLANES = [int(i * NUM_PLANES / NUM_MASTERS) for i in range(NUM_MASTERS)]
# 각 면의 첫 번째 위성을 Master로 지정 (sat_id = plane * SATS_PER_PLANE)
MASTER_SAT_IDS = [p * SATS_PER_PLANE for p in MASTER_PLANES]

# === Simulation Time ===
SIM_START_TIME = datetime(2026, 2, 18, 0, 0, 0, tzinfo=timezone.utc)
SIM_DURATION_DAYS = 7

# === 관측/학습 스케줄 ===
# 학습 트리거 근거: 궤도 주기 × 재방문 궤도 수
# - 570km LEO 궤도 주기 ≈ 96분
# - 1 궤도당 지구 자전으로 ground track이 서쪽으로 ~22.5° 이동
# - REVISIT_ORBITS = 3 궤도 동안 약 67.5°의 경도 범위를 커버
#   → 단일 위성이 지역적 데이터 다양성 확보 가능
# - 단일 위성의 동일 지역 완전 재방문은 ~15궤도(1일) 소요되므로
#   3궤도는 부분 재방문 + 새로운 관측 영역을 균형 있게 포함
REVISIT_ORBITS = 3                                          # 재방문 궤도 수 (학습 주기)
OBSERVATION_INTERVAL_SEC = ORBIT_PERIOD_SEC * REVISIT_ORBITS  # 17,280초 (4.8시간)
OBSERVATION_JITTER_ORBITS = 1                               # ±1 궤도 랜덤 지터
OBSERVATION_JITTER_SEC = ORBIT_PERIOD_SEC * OBSERVATION_JITTER_ORBITS  # ±96분

# === ISL 통신 ===
ISL_HOP_TIME_SEC = 7.1                          # 1홉 총 시간 (SGP4 실측 기반)
                                                # = 모델전송 2.08s + 오버헤드 5s + 전파지연 ~0.01s
ISL_BANDWIDTH_MBPS = 100                        # ISL 대역폭
MODEL_SIZE_MB = 26                              # ResNet-9 모델 크기

# === 로컬 학습 ===
LOCAL_EPOCHS = 5
FEDPROX_MU = 0.01
BASE_LR = 0.01
MIN_LR = 0.001

# === 데이터 ===
NUM_CLIENTS = TOTAL_SATS                        # 238 (Worker+Master 모두 데이터 할당, Master는 학습 안함)
DIRICHLET_ALPHA = 0.5                           # Non-IID 강도
BATCH_SIZE = 128
SAMPLES_PER_CLIENT = 2000

# === 집계 (FedPDA 기반) ===
# Master 로컬 집계 파라미터
BUFFER_MIN_SIZE = 3                             # flush 최소 버퍼
BUFFER_MAX_SIZE = 15                            # 강제 flush 상한
BUFFER_MIN_DIVERSITY = 1                        # flush 최소 궤도면 다양성 (1=다양성 제약 해제)
BUFFER_TIMEOUT_SEC = 3600                       # 다양성 대기 timeout (1시간)
SERVER_LR = 0.5                                 # η_g: 글로벌 50% 보존 + 로컬 50% 반영
SERVER_MOMENTUM = 0.0                           # β=0: 모멘텀 비활성화

# === Master 간 동기화 ===
# Tier 2: Master 간 ring all-reduce
SYNC_AFTER_FLUSH = True                         # Master flush 후 즉시 동기화
# 동기화 지연 = ceil(NUM_MASTERS/2) × (NUM_PLANES/NUM_MASTERS) × ISL_HOP_TIME_SEC
# = 2 × 4.25 × 7.1 ≈ 60.4초 (자동 계산)

# === 평가 ===
EVAL_EVERY_N_ROUNDS = 5
STALENESS_THRESHOLD = 5.0

# === 시드 ===
SEED = 42
