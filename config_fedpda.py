# config.py
# ============================================================
# LEO 위성 비동기 연합학습 비교 실험 설정
# 570km Walker-Delta: 17 planes × 14 sats = 238 satellites
#
# 전략 선택: AGGREGATION_STRATEGY 변경
#   "fedasync"  : FedAsync  (Xie et al., 2019)
#   "fedbuff"   : FedBuff   (Nguyen et al., 2022)
#   "fedspace"  : FedSpace  (So et al., 2022)
#   "fedorbit"  : FedOrbit  (Jabbarpour et al., 2024)
#   "fedpda"    : FedPDA    (Proposed — Plane-Diversity-Aware)
# ============================================================

from datetime import datetime, timedelta, timezone

# === 전략 선택 ===
AGGREGATION_STRATEGY = "fedpda"

# === Constellation ===
NUM_PLANES = 17
SATS_PER_PLANE = 14
ORBIT_ALTITUDE_KM = 570
ORBIT_PERIOD_SEC = 5760        # ~96분 (570km LEO)

# === Simulation Time ===
SIM_START_TIME = datetime(2026, 2, 18, 0, 0, 0, tzinfo=timezone.utc)
SIM_DURATION_DAYS = 7

# === FedAsync (Xie et al., 2019) ===
FEDASYNC_ALPHA_MAX = 0.3
FEDASYNC_STALENESS_FUNC = "poly"  # "poly" | "hinge" | "const"

# === FedBuff (Nguyen et al., 2022) ===
FEDBUFF_K = 10
FEDBUFF_SERVER_LR = 1.0
FEDBUFF_SERVER_MOMENTUM = 0.9

# === FedSpace (So et al., 2022) ===
FEDSPACE_PREDICT_WINDOW_SEC = 600
FEDSPACE_MIN_BUFFER = 3
FEDSPACE_STALENESS_WEIGHT = 0.7
FEDSPACE_SERVER_MOMENTUM = 0.0

# === FedOrbit (Jabbarpour et al., 2024) ===
FEDORBIT_INTRA_AGG_INTERVAL_SEC = 1800
FEDORBIT_SERVER_LR = 1.0

# === FedPDA (Proposed) ===
# Plane-Diversity-Aware Adaptive Buffering
#
# 핵심: FedSpace의 동적 시점 최적화 + 궤도면 다양성 인식
#
# Flush 조건 (dual-condition):
#   Primary:   buffer_size >= dynamic_threshold AND unique_planes >= MIN_DIVERSITY
#   Timeout:   oldest_in_buffer 이후 TIMEOUT_SEC 초과 시 강제 flush (staleness 방지)
#   Fallback:  buffer_size >= MAX_BUFFER 시 diversity 무관 강제 flush
#
# 가중치 (diversity-weighted):
#   w_i = s(tau_i) / plane_count(plane_i)  -> 과대대표 plane 패널티
#   정규화 후 convex combination
#
FEDPDA_PREDICT_WINDOW_SEC = 600      # 접촉 예측 윈도우
FEDPDA_MIN_BUFFER = 3                # 최소 버퍼 크기
FEDPDA_STALENESS_WEIGHT = 0.7        # 동적 threshold 가중치
FEDPDA_MIN_DIVERSITY = 2             # 최소 고유 plane 수
FEDPDA_MAX_BUFFER = 15               # 다양성 미충족 시 강제 flush 상한
FEDPDA_TIMEOUT_SEC = 1800            # 다양성 대기 제한 (30분)
FEDPDA_SERVER_MOMENTUM = 0.0         # 서버 모멘텀 beta

# === 공통: 로컬 학습 ===
LOCAL_EPOCHS = 5
FEDPROX_MU = 0.01
BASE_LR = 0.01
MIN_LR = 0.001

# === 공통: 통신 ===
IOT_FLYOVER_THRESHOLD_DEG = 30.0
GS_FLYOVER_THRESHOLD_DEG = 10.0

# === 공통: 데이터 ===
NUM_CLIENTS = 238
DIRICHLET_ALPHA = 0.5
BATCH_SIZE = 128
SAMPLES_PER_CLIENT = 2000

# === 공통: 평가/필터링 ===
EVAL_EVERY_N_ROUNDS = 5
STALENESS_THRESHOLD = 5.0
