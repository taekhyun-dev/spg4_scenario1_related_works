# config_fedpda.py
# ============================================================
# LEO 위성 비동기 연합학습 비교 실험 설정
# 570km Walker-Delta: 17 planes × 14 sats = 238 satellites
#
# 전략 선택: AGGREGATION_STRATEGY 변경
#   "fedasync"  : FedAsync  (Xie et al., 2019)
#   "fedbuff"   : FedBuff   (Nguyen et al., 2022)
#   "fedspace"  : FedSpace  (So et al., 2022)
#   "fedorbit"  : FedOrbit  (Jabbarpour et al., 2024)
#   "fedpda"    : FedPDA    (Proposed)
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
FEDSPACE_SERVER_MOMENTUM = 0.0  # β=0: severe Non-IID 안정성

# === FedOrbit (Jabbarpour et al., 2024) ===
FEDORBIT_INTRA_AGG_INTERVAL_SEC = 1800  # plane 내 ISL 집계 주기
FEDORBIT_SERVER_LR = 1.0

# === FedPDA (Proposed) ===
FEDPDA_PREDICT_WINDOW_SEC = 600     # 궤도 예측 윈도우 (FedSpace 기반)
FEDPDA_MIN_BUFFER = 3               # 최소 버퍼 크기
FEDPDA_STALENESS_WEIGHT = 0.7       # 동적 threshold의 staleness 가중치
FEDPDA_MIN_DIVERSITY = 2            # flush 최소 궤도면 다양성
FEDPDA_MAX_BUFFER = 15              # 강제 flush 버퍼 상한
FEDPDA_TIMEOUT_SEC = 1800           # 다양성 대기 timeout (s)
FEDPDA_SERVER_MOMENTUM = 0.0        # β=0: 모멘텀 발산 방지
FEDPDA_SERVER_LR = 0.7              # η_g=0.7: 글로벌 30% 보존 + 로컬 70% 반영

# === FedPDA ISL Extension ===
# ISL 비교 실험: True/False 전환으로 ISL 유무 비교
FEDPDA_ISL_ENABLED = True           # ISL 활성화 여부
FEDPDA_ISL_HOP_TIME_SEC = 30        # 궤도면 간 ISL 1홉 전송 시간 (초)
                                    # ResNet-9 (~26MB), LEO ISL ~100Mbps 기준
                                    # 핸드셰이크+확인응답 포함 30초
FEDPDA_ISL_MAX_HOPS = 3             # 최대 릴레이 홉 수
FEDPDA_ISL_MIN_GAIN_SEC = 300       # ISL 릴레이 최소 시간 이득 (5분)
                                    # 이득이 이보다 작으면 직접 GS 접촉 대기

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
DIRICHLET_ALPHA = 0.5           # 0.5 = moderate, 0.1 = severe Non-IID
BATCH_SIZE = 128
SAMPLES_PER_CLIENT = 2000       # 위성당 학습 데이터 수

# === 공통: 평가/필터링 ===
EVAL_EVERY_N_ROUNDS = 5
STALENESS_THRESHOLD = 5.0
