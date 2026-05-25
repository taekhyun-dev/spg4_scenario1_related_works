# 프로젝트 핸드오버 문서

이 문서는 다른 환경의 Claude Code(또는 새 세션)가 프로젝트 상태를
빠르게 파악할 수 있도록 작성된 요약 문서입니다.

작성일: 2026-05-17

---

## 1. 프로젝트 개요

### 연구 주제
**LEO 위성 비동기 연합학습**: 570km Walker-Delta 콘스텔레이션
(17 planes × 14 sats = 238 위성)에서 5가지 FL 전략을 비교하고,
GS(지상국) 없이 궤도 상에서만 학습하는 **Orbital Data Center** 구조로 확장.

### 데이터/모델
- 데이터: CIFAR-10 (기본) 또는 EuroSAT (보완 실험)
- 모델: ResNet-9 (~4.7M params, 26MB)
- 학습: FedProx (μ=0.01), 5 epochs, cosine LR (0.01 → 0.001)
- 분산: Dirichlet α 비IID 샘플링, 위성당 2000장(CIFAR) / 1500장(EuroSAT)

### 비교 대상 5가지 FL 전략
| 전략 | 출처 | 특징 |
|------|-----|------|
| FedAsync | Xie et al., 2019 | 1:1 즉시 비동기 |
| FedBuff | Nguyen et al., 2022 | K-버퍼 pseudo-gradient |
| FedSpace | So et al., 2022 | 궤도 인식 동적 스케줄 |
| FedOrbit | Jabbarpour et al., 2024 | Plane 클러스터 + 마스터 |
| **FedPDA** | **본 연구 (Proposed)** | **Plane-Diversity-Aware + ISL** |

---

## 2. 디렉토리 구조

```
sgp4_scenario1_related_works/
├── config_fedpda.py              # 5전략 시뮬레이터 설정
├── config_orbital.py             # Orbital FL 설정 (Master 위성 기반)
├── satellite_fedpda.py           # 5전략 시뮬레이터 (ISL 미사용)
├── satellite_fedpda_isl.py       # 5전략 시뮬레이터 (FedPDA에 ISL 추가)
├── satellite_orbital.py          # Orbital FL 시뮬레이터 (지상국 없음)
│
├── ml/
│   ├── model.py                  # ResNet-9, MobileNetV3, PyTorchModel
│   ├── data.py                   # CIFAR-10 + EuroSAT Dirichlet 로더
│   ├── training.py               # FedProx 로컬 학습
│   ├── aggregation.py            # weighted_update, fed_avg
│   └── metrics.py                # MetricsCollector
│
├── utils/
│   ├── skyfield_utils.py         # SGP4 궤도 전파
│   └── logging_setup.py          # 로거 (suffix/log_dir 인자 지원)
│
├── run_all_strategies.py         # 5전략 순차 실행 (env var 방식)
├── run_parallel_sweep.py         # 병렬 sweep (satellite_fedpda_isl.py 호출)
├── run_fedpda_sweep.py           # 병렬 sweep (satellite_fedpda.py 호출)
├── run_orbital_sweep.py          # 병렬 sweep (satellite_orbital.py 호출)
│
├── analysis_master_placement.py  # Master 배치 정량 분석 스크립트
├── master_placement_analysis.md  # Master 배치 분석 결과 문서
├── satellite_orbital_design.md   # Orbital FL 설계 문서
├── isl_algorithm.md              # ISL 통신 알고리즘 문서
│
├── constellation.tle             # Walker-Delta TLE 파일
├── results/                      # 실험 결과 (전략/SEED/α 별 분리)
└── logs/                         # 시뮬레이션 로그 + sweep 진행 로그
```

---

## 3. 환경변수 (Env Var) 인터페이스

**중요**: config 파일을 직접 수정하지 않고 환경변수로 모든 변수를 제어할 수 있게 리팩토링됨.
이는 병렬 실행 시 config 파일 충돌(race condition)을 막기 위한 설계.

| 환경변수 | 적용 시뮬레이터 | 기본값 | 의미 |
|---------|---------------|--------|------|
| `ORBITAL_FL_STRATEGY` | fedpda_isl, fedpda | "fedpda" | 5전략 중 선택 (fedasync/fedbuff/fedspace/fedorbit/fedpda) |
| `ORBITAL_FL_SEED` | 모두 | 42 | 시드 (random/numpy/torch 모두 적용) |
| `ORBITAL_FL_ALPHA` | 모두 | 0.5 | Dirichlet α 비IID 강도 |
| `ORBITAL_FL_ETA_G` | fedpda, orbital | 0.37 | FedPDA SERVER_LR (η_g) |
| `ORBITAL_FL_DATASET` | orbital | "cifar10" | "cifar10" 또는 "eurosat" |

---

## 4. 시뮬레이터 3종 비교

| 항목 | `satellite_fedpda.py` | `satellite_fedpda_isl.py` | `satellite_orbital.py` |
|------|----------------------|--------------------------|------------------------|
| 지상국 | O (서울) | O (서울) | **X (없음)** |
| ISL 릴레이 | X | O (fedpda 한정) | **O (필수, 상시 연결)** |
| 학습 트리거 | IoT flyover | IoT flyover | **궤도 주기 (4.8시간)** |
| 집계 노드 | GS | GS | **Master 위성 4개** |
| 5 전략 지원 | O | O | **X (orbital FL 전용)** |
| 데이터셋 | CIFAR-10 | CIFAR-10 | **CIFAR-10 또는 EuroSAT** |

### 4.1 `satellite_orbital.py` 핵심 설계

- **Master 4개** (Plane 0, 4, 8, 12의 sat_id=0 위치): `MASTER_SAT_IDS = [0, 56, 112, 168]`
- **이벤트 흐름**: `TRAIN_COMPLETE` → `MODEL_DELIVERED` → `MASTER_SYNC`
- **2-tier 집계**:
  - Tier 1: Master 로컬 집계 (FedPDA pseudo-gradient)
  - Tier 2: Master 간 동기화 (가중 평균, 60.3초 지연)
- **글로벌 업데이트 시점**: Tier 2 sync에서만 (Tier 1은 master_local_models에만 저장)
- **ISL 홉 시간**: 7.1초/홉 (모델 26MB / 100Mbps + 5초 오버헤드)

---

## 5. 결과 디렉토리 명명 규칙

### 5.1 태그 포맷

```python
# config 파일에서 자동 계산되는 태그
ALPHA_TAG   = f"A{int(α * 10):02d}"     # 0.1→A01, 0.5→A05, 1.0→A10
ETA_TAG     = f"E{int(η_g * 10):02d}"   # 0.3→E03, 0.5→E05, 0.7→E07
DATASET_TAG = {"cifar10":"C10", "eurosat":"ES"}[DATASET]
```

⚠ α=0.01은 A00이 되어 0.0과 구분되지 않음 (현재 알려진 한계).
필요 시 `f"A{int(α*100):03d}"`로 변경 가능 (A001, A010, A050, A100).

### 5.2 시뮬레이터별 결과 경로

| 시뮬레이터 | 전략 | 결과 경로 |
|----------|-----|---------|
| `satellite_fedpda.py` | 모든 전략 | `results/{strategy}_S{S}_{ALPHA_TAG}[_{ETA_TAG}]/` (fedpda만 ETA_TAG 포함) |
| `satellite_fedpda_isl.py` | fedpda + ISL=True | `results/fedpda_isl_S{S}_{ALPHA_TAG}_{ETA_TAG}/` |
| `satellite_fedpda_isl.py` | 그 외 4전략 | `results/{strategy}_S{S}_{ALPHA_TAG}/` |
| `satellite_orbital.py` | (단일 전략) | `results/orbital_fl_{DATASET_TAG}_M{N}_S{S}_{ALPHA_TAG}_{ETA_TAG}/` |

### 5.3 로그 경로

각 시뮬레이션의 내부 로그는 결과 경로와 같은 폴더명을 `logs/` 아래에 사용:
```
logs/orbital_fl_C10_M4_S42_A05_E07/
  ├── simulation_{ts}_S42_A05_E07.log
  └── performance_{ts}_S42_A05_E07.csv
```

Sweep 진행 로그(subprocess stdout)는 별도:
```
logs/sweep_parallel/{tag}.log       # run_parallel_sweep.py
logs/sweep_fedpda_plain/{tag}.log   # run_fedpda_sweep.py
logs/sweep_orbital/{tag}.log        # run_orbital_sweep.py
```

---

## 6. Sweep 스크립트 사용법

### 6.1 run_parallel_sweep.py (FedPDA+ISL 기준 5전략)

```bash
# 기본: 5전략 × 3시드 × 4α (60 실험)
python run_parallel_sweep.py --jobs 4

# η_g 추가 sweep (fedpda 전용으로 적용됨)
python run_parallel_sweep.py --jobs 4 --eta-gs 0.1 0.3 0.5 0.7 1.0

# 특정 전략만
python run_parallel_sweep.py --strategies fedpda fedbuff

# 일부만
python run_parallel_sweep.py --seeds 42 --alphas 0.1 0.5

# Dry-run
python run_parallel_sweep.py --dry-run
```

### 6.2 run_fedpda_sweep.py (Plain FedPDA, ISL 없음)

```bash
# FedPDA only (기본): 3시드 × 4α = 12 실험
python run_fedpda_sweep.py --jobs 4

# η_g sweep
python run_fedpda_sweep.py --jobs 4 --eta-gs 0.1 0.3 0.5 0.7 1.0
```

### 6.3 run_orbital_sweep.py (Orbital FL)

```bash
# CIFAR-10 (기본)
python run_orbital_sweep.py

# EuroSAT
python run_orbital_sweep.py --datasets eurosat

# 둘 다 비교
python run_orbital_sweep.py --datasets cifar10 eurosat

# 검증용 (1 실험)
python run_orbital_sweep.py --datasets eurosat \
    --seeds 42 --alphas 0.5 --jobs 1
```

### 6.4 자원 가이드 (64GB RAM + RTX 4090 24GB 기준)

| --jobs | RAM | VRAM | 권장 |
|--------|-----|------|------|
| 1 | 10GB | 2GB | 검증용 |
| 3 | 30GB | 6GB | ✅ 안전 |
| **4** | **40GB** | **8GB** | ✅ **추천 (속도)** |
| 5 | 50GB | 10GB | ⚠ 좁음 |
| 6+ | 60GB+ | 12GB+ | ❌ 위험 |

---

## 7. 핵심 알고리즘 결정 사항 (변경 이력)

### 7.1 FedPDA (제안 알고리즘)

**Pseudo-gradient 집계** + **Staleness 가중치**:
```
s(τ) = (1 + τ)^(-0.5)                  # τ = global_ver - base_ver
w_i = s(τ_i) / Σs(τ)                    # 정규화 가중치
Δ = Σ w_i × (w_global - w_trained_i)    # pseudo-gradient
w_new = w_global - η_g × Δ              # η_g = 0.37 (or 0.7, sweep 대상)
```

### 7.2 본 세션에서 변경된 사항

| 항목 | 변경 전 | 변경 후 | 사유 |
|-----|--------|--------|------|
| Plane diversity 가중치 | `s(τ) / c_p` | **`s(τ)`만** | 사용자 결론: 의미 없음 |
| Flush 다양성 조건 | `≥ 2 면` | **`≥ 1 면`** | 다양성 제약 해제 |
| Buffer timeout (orbital) | 1800초 | **3600초** | ISL 균등 도착으로 여유 |
| Tier 1 → 글로벌 업데이트 | 즉시 | **하지 않음** | Tier 2 합의 결과만 글로벌 |
| sync_delay (N=1) | 120.7초 (오류) | **0초** | N=1은 동기화 불필요 |
| FEDPDA_ISL_MIN_GAIN_SEC | 300 | **0** | 수학적 근거 없는 보수 필터 제거 |
| 학습 주기 근거 | 4시간 (임의) | **4.8시간 = 3궤도 × 96분** | 궤도역학 기반 |

### 7.3 시드 적용 (재현성)

`satellite_orbital.py`는 `__init__`에서 `set_global_seed(SEED)`로 전역 시드 적용:
```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

`satellite_fedpda*.py`는 `main()`에서 동일하게 적용 (기존부터).

---

## 8. ISL 통신 모델

### 8.1 홉 시간 (모든 시뮬레이터 동일)
```
t_hop = t_transfer + t_overhead + t_propagation
      = 2.08s (26MB/100Mbps) + 5s (가정) + ~0.01s (전파)
      = 7.1초
```

### 8.2 Worker → Master 전달 (Orbital FL)
```
inter_hops = min(|src_plane - master_plane|, 17 - |...|)
intra_hops = min(|src_idx - master_idx|, 14 - |...|)
delivery = (inter + intra) × 7.1초
```
평균 32.9초, 최대 63.9초 (Master 4개 기준).

### 8.3 Master 간 동기화 지연
```python
def compute_sync_delay():
    if NUM_MASTERS <= 1:
        return 0.0  # N=1 특례
    rounds = math.ceil(NUM_MASTERS / 2)
    spacing = NUM_PLANES / NUM_MASTERS
    return rounds * spacing * ISL_HOP_TIME_SEC  # = 60.3s (N=4)
```

⚠ 현재는 "parallel ring" 가정의 낙관적 추정값. 표준 ring all-reduce 사용 시 181초.
설계 문서(`isl_algorithm.md`)에 정확한 알고리즘 명시 필요.

---

## 9. EuroSAT 추가 사항 (보완 실험용)

### 9.1 데이터셋
- 27,000장 × 64×64 RGB × 10 classes (균형)
- Sentinel-2 위성 관측 데이터
- 원본 64×64를 **32×32로 다운샘플링**하여 ResNet-9 호환 유지

### 9.2 `ml/data.py`의 `get_eurosat_loaders()`
CIFAR-10 로더와 동일한 시그니처:
```python
get_eurosat_loaders(
    num_clients, dirichlet_alpha=0.5,
    data_root='./data', batch_size_val=256, num_workers=8,
    samples_per_client=1500,   # EuroSAT 풀 크기 고려
    image_size=32,             # ResNet-9 호환
    val_ratio=0.2,             # 수동 train/val 분할
    split_seed=42,             # 분할 시드 (실험 시드와 별개)
)
```

### 9.3 EuroSAT 사용
`satellite_orbital.py`만 지원. `ORBITAL_FL_DATASET=eurosat`로 활성화.

```bash
# 다운로드 (1회)
python -c "from torchvision.datasets import EuroSAT; EuroSAT(root='./data', download=True)"

# 실행
ORBITAL_FL_DATASET=eurosat python satellite_orbital.py
```

ResNet-9 + 32×32 EuroSAT 예상 정확도: 82~88%.

---

## 10. 관련 문서

| 파일 | 내용 |
|------|------|
| `master_placement_analysis.md` | Master 위성 배치 정량 분석 (Option A/B, 4개 권장 근거) |
| `satellite_orbital_design.md` | Orbital FL 시뮬레이터 전체 설계 |
| `isl_algorithm.md` | ISL 통신 알고리즘 상세 (Tier 1/2, 홉 계산) |
| `analysis_master_placement.py` | SGP4 실측 기반 Master 배치 분석 코드 |

---

## 11. 환경 (실행 환경)

### Python
- **사용 환경**: `kepler_propagation` (pyenv)
  - Python 3.10.2
  - torch 2.8.0+cu128
  - torchvision 0.23.0+cu128

```bash
pyenv activate kepler_propagation
# 또는
pyenv shell kepler_propagation
```

### 사용 불가 환경
- `sgp4` (Python 3.7.9): torchvision import 시 `from torch._six import PY3` 오류

### 하드웨어 권장
- RAM 64GB+ (4 jobs 병렬 기준 40GB 사용)
- GPU VRAM 24GB+ (RTX 4090 권장, 4 jobs 병렬 기준 8GB)

---

## 12. 진행 중 / TODO

### 완료
- ✅ 5전략 비교 시뮬레이터 (`satellite_fedpda*.py`)
- ✅ Orbital FL 시뮬레이터 (`satellite_orbital.py`)
- ✅ Master 배치 정량 분석 (4개 권장)
- ✅ 환경변수 인터페이스 (SEED/ALPHA/ETA_G/STRATEGY/DATASET)
- ✅ 병렬 sweep 스크립트 3종
- ✅ EuroSAT 데이터로더 추가
- ✅ Plane diversity 제거, Tier 1/2 분리, sync_delay 수정

### 진행 중
- 🟡 EuroSAT 데이터 다운로드 (4090 PC에서 수행 예정)
- 🟡 EuroSAT sweep 실행 (`run_orbital_sweep.py --datasets eurosat`)

### 잠재적 TODO
- ⚪ `ALPHA_TAG` 포맷 개선 (A00 → A001로 명확화)
- ⚪ Master 간 동기화 알고리즘 명시 (Super Master vs Ring All-Reduce)
- ⚪ ISL 5초 overhead의 근거 보강
- ⚪ 결과 집계/시각화 스크립트 (eta_sweep_analysis 확장)

---

## 13. 빠른 시작 가이드

### 단일 실험 예시

```bash
pyenv shell kepler_propagation

# Orbital FL with CIFAR-10
ORBITAL_FL_SEED=42 ORBITAL_FL_ALPHA=0.5 python satellite_orbital.py

# Orbital FL with EuroSAT
ORBITAL_FL_DATASET=eurosat ORBITAL_FL_SEED=42 ORBITAL_FL_ALPHA=0.5 \
    python satellite_orbital.py

# FedPDA+ISL with specific η_g
ORBITAL_FL_STRATEGY=fedpda ORBITAL_FL_SEED=42 ORBITAL_FL_ALPHA=0.5 \
    ORBITAL_FL_ETA_G=0.5 python satellite_fedpda_isl.py
```

### Sweep 예시 (백그라운드)

```bash
# FedPDA+ISL 60 실험 (3시드 × 4α × 5η_g)
nohup python run_parallel_sweep.py --jobs 4 \
    --strategies fedpda --eta-gs 0.1 0.3 0.5 0.7 1.0 \
    > sweep.log 2>&1 &

# 진행 확인
tail -f sweep.log
ls -lt logs/sweep_parallel/
```

### 결과 위치
```bash
ls results/                       # 모든 실험 결과
ls results/orbital_fl_*/          # Orbital FL 결과
ls results/fedpda_isl_*/          # FedPDA+ISL 결과
```

---

## 14. 주의사항

1. **CLAUDE.md는 .gitignore에 포함됨** (Claude Code 전용 문서)
2. 실험 결과 폴더 (`results/`, `logs/`)는 `.gitignore`되어 있음
3. `constellation.tle`은 생성 스크립트 별도 (기존 코드 참조)
4. Pre-existing `results/orbital_fl_M4_S42/` 같은 경로는 이전 명명 규칙
   - 새 규칙: `results/orbital_fl_{DTAG}_M{N}_S{S}_{ATAG}_{ETAG}/`
   - 기존 결과 보존하려면 폴더 rename 권장
5. **시드 42 첫 실험은 numpy/torch 시드 미적용 상태**였음 (현재는 적용됨)
   - 정확한 재현성 위해서는 SEED=42 재실행 필요

