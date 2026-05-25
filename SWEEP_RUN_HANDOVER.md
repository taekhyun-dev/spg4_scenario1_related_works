# Sweep 실행 핸드오버 문서

다른 환경/세션의 Claude Code가 **현재 진행 중인 η_g sweep**을 이어받을 수 있도록
작성된 문서. 기존 `PROJECT_HANDOVER.md`(프로젝트 전반)와 별개로,
**이번 세션의 sweep 실행 작업**에 집중.

작성일: 2026-05-25
환경: A5000 × 2 (각 24GB) + 40 cores + **Pod RAM 한도 125.8 GB**

---

## 1. 이 sweep의 핵심 목적

**이론값 η_g = 0.37 검증**: FedPDA의 server learning rate(η_g)를 수식으로 도출한 값이
0.37인데, 이것이 **실측 best η_g와 일치하는지** 비교하는 것이 목적.

- 기존 결과: η_g=0.37, SEED=42, α∈{0.1, 0.5, 1.0} 은 이미 **gdrive에 저장됨** (FP32, 동일 코드 버전)
- 이번 sweep: 다른 η_g 값들을 SEED=42 동일 조건으로 돌려 곡선을 그리고 0.37과 비교
- **그래프 계획**: x축=η_g, y축=accuracy, α별 곡선 + 0.37은 gdrive 결과를 별도 marker(★)로 표시

---

## 2. 현재 실행 중인 Sweep 구성 (총 39 실험)

| Sweep | 스크립트 | 전략 | seeds | α | η_g | 실험 수 | jobs |
|-------|---------|------|-------|---|------|---------|------|
| **A** | run_parallel_sweep.py | fedorbit | 42,123,7777 | 0.1,0.5,1.0 | — | 9 | 9 |
| **B** | run_parallel_sweep.py | fedpda(+ISL) | **42** | 0.1,0.5,1.0 | 0.1,0.3,0.5,0.7,1.0 | 15 | 11 |
| **C** | run_fedpda_sweep.py | fedpda(plain) | **42** | 0.1,0.5,1.0 | 0.1,0.3,0.5,0.7,1.0 | 15 | 10 |

**총 30 jobs 동시 실행** (GPU 라운드로빈 분배).

### 왜 fedpda/fedpda+ISL은 SEED=42만?
- η_g sweep은 하이퍼파라미터 탐색 + 이론값 검증 목적 → 단일 seed로 충분
- 0.37 기존 결과가 SEED=42라서 **동일 seed로 비교해야 fair**
- fedorbit만 3 seeds (baseline 비교용으로 다른 전략처럼 seed sweep 유지)

### 실행 명령 (재시작 시 그대로 사용)
```bash
# A: fedorbit
SWEEP_GPU_IDS="0,1" nohup python -u run_parallel_sweep.py --jobs 9 \
  --strategies fedorbit --seeds 42 123 7777 --alphas 0.1 0.5 1.0 \
  > logs/sweep_fedorbit_master.log 2>&1 &

# B: fedpda + ISL
SWEEP_GPU_IDS="1,0" nohup python -u run_parallel_sweep.py --jobs 11 \
  --strategies fedpda --seeds 42 --alphas 0.1 0.5 1.0 \
  --eta-gs 0.1 0.3 0.5 0.7 1.0 \
  > logs/sweep_fedpda_isl_master.log 2>&1 &

# C: fedpda plain (ISL 미사용)
SWEEP_GPU_IDS="0,1" nohup python -u run_fedpda_sweep.py --jobs 10 \
  --strategies fedpda --seeds 42 --alphas 0.1 0.5 1.0 \
  --eta-gs 0.1 0.3 0.5 0.7 1.0 \
  > logs/sweep_fedpda_plain_master.log 2>&1 &
```

---

## 3. 이번 세션에서 변경한 코드 (★ 중요)

> 다른 환경에서 동일 코드 버전을 받았다면 이 변경들이 이미 적용되어 있어야 함.
> git diff로 확인할 것.

### 3.1 GPU 라운드로빈 분배 — `run_parallel_sweep.py`, `run_fedpda_sweep.py`
- `run_one(job)`에 `job_idx` 추가, `SWEEP_GPU_IDS` 환경변수(기본 "0,1")로 GPU 분배
- 각 sub-job env에 `CUDA_VISIBLE_DEVICES = gpu_ids[job_idx % len(gpu_ids)]`
- **thread 제한 env**도 함께 주입: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`
- `jobs = [(*j, idx) for idx, j in enumerate(raw_jobs)]` 형태로 인덱스 부여

### 3.2 AMP (자동 혼합정밀) — `ml/training.py`
- `train_model()`에 `autocast('cuda')` + `GradScaler('cuda')` 적용
- forward/loss/prox_term을 autocast 안에서 계산, `scaler.scale(loss).backward()`
- gradient clipping은 `scaler.unscale_(optimizer)` 후 적용
- **FedProx 항(prox_term)은 그대로 보존** (μ=0.01)
- `evaluate_model()`의 `autocast()` → `autocast('cuda')` (deprecation 수정)

### 3.3 GPU 최적화 플래그 — `satellite_fedpda.py`, `satellite_fedpda_isl.py` `main()`
```python
torch.backends.cudnn.deterministic = False   # 기존 True
torch.backends.cudnn.benchmark = True         # 기존 False
torch.set_float32_matmul_precision('high')    # 신규 (TF32 활성화)
```

### 3.4 DataLoader 워커 감소 — `satellite_fedpda.py:~1085`, `satellite_fedpda_isl.py:~1058`
- 학습 train_loader의 `num_workers=8 → 2` (학습마다 8 worker spawn 비용 제거)

### 3.5 변경 안 한 것 (의도적)
- **BATCH_SIZE = 128 유지** (config_fedpda.py): batch 변경 시 학습 동역학이 달라져
  gdrive 0.37 결과와 fair comparison이 깨지므로 **절대 바꾸지 말 것**.
  (세션 중 256으로 올렸다가 fair comparison 문제로 되돌림)
- LOCAL_EPOCHS=5, SIM_DURATION_DAYS=7 (학술 설정 유지)

---

## 4. 환경 제약 — ★ Pod RAM 한도 (반드시 숙지)

- 이 환경은 **Kubernetes Pod, cgroup memory 한도 = 125.8 GB**
- `free -h`가 보여주는 251 GB는 **호스트 전체이지 컨테이너 가용량이 아님**
- 확인: `cat /sys/fs/cgroup/memory/memory.usage_in_bytes` (cgroup v1) 또는 `memory.current` (v2)
- **39 jobs에서 131 GB 사용 → OOM kill 5개 발생** (2026-05-24)
- **30 jobs가 안전 상한** (RAM ~116 GB, 92%). 더 늘리지 말 것.
- 한 sub-job ≈ main 3.4 GB + DataLoader workers. jobs × 단위 RAM이 100 GB 넘지 않게.

### Python 환경
- `/home/jovyan/.venv/torch2.5.1-py3.12-cuda12.4` (torch 2.5.1+cu124, CUDA 2 GPU)
- 추가 설치 필요했던 패키지: `skyfield==1.53`, `sgp4==2.25`, `torchmetrics`
- `constellation.tle`이 한때 누락 → 현재는 존재 (714줄, 238 위성). 없으면 모든 실험 즉시 실패.

---

## 5. 현재 진행 상황 (2026-05-25 18:56 기준)

- **시작**: 2026-05-24 15:02
- **경과**: 약 28시간
- **시뮬 진행**: `02-20` 부근 = 약 64h / 168h (**38%**)
- **속도**: 시뮬/실시간 ≈ **2.29×** (AMP+tuning 효과)
- **완료**: 0/39 (한 실험 ~3일 소요, 첫 완료까지 ~2일 남음)
- **ETA**: 전체 약 **4.5~5일** (5/29경)
- RAM 116 GB(92%), GPU 각 8.5 GB / OOM 없음

### 성능 튜닝 히스토리 (ETA 변화)
| 단계 | ETA |
|------|-----|
| 초기 8+8 jobs FP32 | ~19일 |
| 30 jobs FP32 | ~13일 |
| + thread limit + workers↓ | ~11일 |
| **+ AMP + cuDNN benchmark + TF32 (현재 30 jobs)** | **~5일** |

---

## 6. 모니터링 / 진행 확인 방법

```bash
# 완료/실패 카운트
for f in fedorbit_master fedpda_isl_master fedpda_plain_master; do
  echo "$f: ✅$(grep -cE '\[[0-9]+/[0-9]+\] ✅' logs/sweep_${f}.log) \
❌$(grep -cE '\[[0-9]+/[0-9]+\] ❌' logs/sweep_${f}.log)"
done

# Pod RAM (★ OOM 감시)
cat /sys/fs/cgroup/memory/memory.usage_in_bytes | awk '{printf "%.1f GB / 125.8\n", $1/1073741824}'

# GPU
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

# 시뮬 진행 (한 실험)
tail -100 logs/sweep_fedpda_plain/fedpda_S42_A05_E05.log | grep -oE "📡 \[02-[0-9]+ [0-9:]+\]" | tail -1

# OOM 발생 여부
dmesg -T | grep -iE "oom|kill" | tail -5

# 활성 main jobs (30이어야 정상)
ps -ef | awk '$NF ~ /satellite_fedpda(_isl)?\.py$/' | wc -l
```

---

## 7. 결과 위치 & 다음 단계

### 결과 디렉토리 (완료 시 생성)
- fedorbit: `results/fedorbit_S{42|123|7777}_A{01|05|10}/`
- fedpda plain: `results/fedpda_S42_A{01|05|10}_E{01|03|05|07|10}/`
- fedpda+ISL: `results/fedpda_isl_S42_A{01|05|10}_E{01|03|05|07|10}/`

⚠️ **ETA_TAG 인코딩 주의**: `E{int(η_g*10):02d}` → η_g=0.37이면 E03 (0.3과 충돌!).
   gdrive의 0.37 결과를 받아올 때 폴더명을 E03과 겹치지 않게 별도 이름 부여 권장
   (예: `results/fedpda_S42_A05_E037_theory/`).

### 완료 후 할 일
1. gdrive에서 η_g=0.37 결과 (SEED=42, α=0.1/0.5/1.0) 다운로드
2. sweep 결과(5 η_g점) + 0.37 marker로 비교 그래프 작성 (plain, ISL 각각)
3. 실측 best η_g 식별 → 이론값 0.37과 일치 여부 판단
4. 주의: 새 sweep은 AMP(FP16), gdrive 0.37은 FP32 → 절대 accuracy는 약간 차이날 수 있음.
   곡선 모양/best 위치 비교는 유효하나 0.37 점은 "참고 marker"로 다룰 것.

---

## 8. 문제 발생 시 대응

| 증상 | 원인 | 대응 |
|------|------|------|
| 모든 실험 즉시(초 단위) 실패 | constellation.tle 누락 또는 import 에러 | 파일 존재 확인, skyfield/sgp4/torchmetrics 설치 확인 |
| OOM kill (dmesg) | Pod RAM 125.8 GB 초과 | jobs 줄이기 (30→25). free -h 말고 cgroup 확인 |
| 통신 계산이 분 단위로 느림 | thread oversubscription | SWEEP_GPU_IDS sweep script에 thread env 있는지 확인 |
| epoch 시간 과도 (>10s) | AMP 미적용 또는 jobs 과다 | ml/training.py AMP 확인, jobs 수 점검 |

재시작 절차: 기존 프로세스 kill → logs/results 정리 → §2.3 명령 재실행
```bash
ps -ef | grep -E "run_parallel|run_fedpda|satellite_fedpda" | grep -v grep \
  | awk '{print $2}' | xargs -r kill -9
```
