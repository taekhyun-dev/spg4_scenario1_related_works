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
# ============================================================

# === 전략 선택 ===
AGGREGATION_STRATEGY = "fedspace"

# === Constellation ===
NUM_PLANES = 17
SATS_PER_PLANE = 14
ORBIT_ALTITUDE_KM = 570
ORBIT_PERIOD_SEC = 5760        # ~96분 (570km LEO)

# === FedAsync (Xie et al., 2019) ===
FEDASYNC_ALPHA_MAX = 0.3
FEDASYNC_STALENESS_FUNC = "poly"  # "poly" | "hinge" | "const"

# === FedBuff (Nguyen et al., 2022) ===
FEDBUFF_K = 10                 # 버퍼 크기 (논문 K=10 최적)
FEDBUFF_SERVER_LR = 1.0        # 서버 학습률 η_g
FEDBUFF_SERVER_MOMENTUM = 0.9  # 서버 모멘텀 β
# β=0.0: 순수 pseudo-gradient averaging (η_g=1.0 → FedAvg 등가)
# β=0.9: 서버 모멘텀 활성화 시 η_g=0.1 권장 (실효 스텝 ≈ η_g/(1-β) = 1.0)

# === FedSpace (So et al., 2022) ===
FEDSPACE_PREDICT_WINDOW_SEC = 600
FEDSPACE_MIN_BUFFER = 3
FEDSPACE_STALENESS_WEIGHT = 0.7
FEDSPACE_SERVER_MOMENTUM = 0.0

# === FedOrbit (Jabbarpour et al., 2024) ===
FEDORBIT_INTRA_AGG_INTERVAL_SEC = 1800  # plane 내 ISL 집계 주기
FEDORBIT_SERVER_LR = 1.0

# === 공통: 로컬 학습 ===
LOCAL_EPOCHS = 5
FEDPROX_MU = 0.01
BASE_LR = 0.01
MIN_LR = 0.001

# === 공통: 통신 ===
IOT_FLYOVER_THRESHOLD_DEG = 30.0
GS_FLYOVER_THRESHOLD_DEG = 10.0

# === 공통: 데이터 ===
NUM_CLIENTS = 238               # 위성 수와 동일
DIRICHLET_ALPHA = 0.5
BATCH_SIZE = 128
SAMPLES_PER_CLIENT = 2000       # 위성당 학습 데이터 수 (독립 샘플링, 중복 허용)

# === 공통: 평가/필터링 ===
EVAL_EVERY_N_ROUNDS = 5
STALENESS_THRESHOLD = 5.0

# === 시뮬레이션 시간 ===
# 기존 실험과 동일한 통신 스케줄 재현을 위해 시작 시간 고정
# (원본 실행: 2026-02-18 02:05:34 UTC, Strategy: FEDASYNC)
from datetime import datetime, timezone
SIM_START_TIME = datetime(2026, 2, 18, 2, 5, 34, tzinfo=timezone.utc)
SIM_DURATION_DAYS = 14