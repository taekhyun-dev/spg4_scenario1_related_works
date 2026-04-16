# FedPDA 논문 프레이밍 재구성 세션 정리

## 1. 세션 배경

이전 세션까지: diversity weight 무효성 수학적 증명, "궤도면 접촉 군집" 가설 실측 반증(연속 접촉 0.02%), FedPDA+ISL 코드 통합 완료, η_g sweep 결과 확보.

본 세션: Option C(diversity weight 제거, Global Preservation 중심 재구성) 확정, Lemma 1 완전 증명 작성, 교수 보고용 정리.

---

## 2. 핵심 결정사항

### 2.1 논문 프레이밍 전면 재구성

**기존:**
```
FedPDA = diversity weight + Global Preservation + dual-condition flush
```

**수정:**
```
FedPDA = satellite-adapted pseudo-gradient 분석
       + Global Preservation (η_g < 1.0)
       + ISL 확장
       + 수렴 바운드 증명
```

### 2.2 재구성 근거

1. **Diversity weight 무효성**: 독립 Dirichlet에서 분산 80% 증가 (Monte Carlo 10,000회 검증)
2. **궤도면 군집 반증**: GS 연속 접촉 5,880회 중 1회(0.02%)
3. **η_g가 실질적 개선 원인**: sweep에서 명확한 U자형 확인

---

## 3. 핵심 개념: satellite-adapted pseudo-gradient

### 3.1 원 FedBuff vs 위성적응

```
원 FedBuff:    Δ = w_base - w_trained      (η_g = 학습률)
위성적응:      Δ = w_global - w_trained    (η_g = 혼합 비율)
```

**의미 변화**: 같은 기호 η_g지만 수학적 역할이 완전히 다름.
- 원 FedBuff: gradient 방향의 step size (학습률)
- 위성적응: 글로벌-로컬 혼합 비율, (1-η_g) = 글로벌 보존률

### 3.2 w_global을 base로 사용하는 이유

stale gradient drift 회피. τ≈5인 환경에서 w_base(v3)의 gradient를 현재 글로벌(v8)에 적용하면 loss landscape 괴리로 발산 가능. satellite-adapted는 convex combination이므로 항상 안정.

### 3.3 η_g=1.0의 구조적 문제

satellite-adapted에 대입하면:
```
w_{t+1} = w_t - 1.0 × Σ α_i × (w_t - w_i) = Σ α_i × w_i
```
→ 이전 글로벌 모델이 수식에서 완전 소거 (보존률 0%)

FedBuff/FedSpace/FedOrbit이 모두 η_g=1.0 사용 → 위성 환경에서 구조적 문제 발생.

---

## 4. 실험 데이터

### 4.1 η_g Sweep (α=0.1, ISL 없음, 7일)

| η_g | 보존률 | 최고 | 최종 | std | 70% 도달 |
|-----|-------|------|------|-----|---------|
| 1.0 | 0% | 75.31% | 67.82% | 5.21% | 91.9h |
| 0.7 | 30% | 77.16% | 70.61% | 3.23% | 84.9h |
| **0.5** | **50%** | **77.44%** | **70.97%** | 2.17% | **84.9h** |
| 0.3 | 70% | 76.47% | 70.99% | 1.52% | 94.3h |
| 0.1 | 90% | 70.12% | 67.68% | 1.33% | 165.7h |

**발견**: α=0.1 최적은 η_g=0.5 (기존 논문 0.7 수정 필요). 명확한 U자형 트레이드오프.

### 4.2 FedPDA+ISL 결과

- α=0.5, η_g=0.7: peak 84.59%, final 81.27%, std 0.68%, 70%→19.5h, 80%→50.3h
- α=0.1, η_g=0.5: peak 81.49%, final 69.33%, std 2.29%, 70%→36.0h, 80%→119.4h
- GS 접촉 5,880 동일 (ISL은 궤도 역학 불변, upload/download/skip 합 10,438)
- 평균 τ: 3.5 → 2.63 (25% 감소), 총 라운드 366 → 707 (1.93배)

---

## 5. 수렴 증명 로드맵

### 5.1 가정 (Assumption 1-4)
L-smoothness, bounded local variance, bounded heterogeneity (σ_G²), bounded staleness (τ_max).

### 5.2 업데이트 방향 분해
```
u_t = η_l·E·∇F(w_t) + d̄ + η_l·E·ē^G + η_l·E·ē^τ
```
(유효 gradient + global drift + Non-IID 오차 + staleness 오차)

### 5.3 Lemma 1 (1라운드 감소량, 완전 증명 완료)

```
F(w_{t+1}) ≤ F(w_t)
  - (η_g·η_l·E/2)‖∇F‖²              (I: 유효 하강)
  + 2L·η_g²·η_l²·E²·‖∇F‖²            (II: gradient 제곱)
  + 2η_g·η_l·E·σ_G² + 2L·η_g²·η_l²·E²·σ_G²   (III: Non-IID)
  + 2η_g·η_l·E·L²·D̄_t² + ...          (IV: staleness)
  + (η_g/η_l·E)·D̄_t² + 2L·η_g²·D̄_t²  (V: drift)
```

### 5.4 η_g의 차수 구조 (핵심 결과)

D̄_t² ≤ η_g²·U²·τ̄² 대입 시:

| 항 | η_g 차수 | η_g↓ 효과 |
|---|---|---|
| (I) 유효 하강 | η_g¹ | 선형 손해 |
| (IV) staleness | **η_g³** | 세제곱 이득 |
| (V) drift | **η_g³** | 세제곱 이득 |

→ 위성 환경(높은 τ)에서 η_g<1이 바운드 개선. U자형 트레이드오프의 수학적 근거.

### 5.5 Theorem 1 & Corollary (미완)

- T라운드 텔레스코핑 → 수렴 바운드
- 최적 η_g* = f(τ, σ_G²) 도출
- η_g=1.0이 바운드 악화 증명

### 5.6 주의점
η_l·E=0.05 << 1에서 Step 2의 부등식이 느슨해짐. 초과분을 drift 항에 흡수시키는 방식으로 처리. 최종 논문에서 정교화 필요.

---

## 6. 관련 연구와의 차별화

| 논문 | 보존 방식 | 차별점 |
|---|---|---|
| FedAsync (2019) | 87.8% (staleness-adaptive) | 1:1, 효율 10% |
| FedASMU (AAAI 2024) | 동적 조절 | 직접 혼합, 1:1 |
| FedMeld (2024.12) | 지역 간 mixing ratio | 탈중앙, 수평 혼합 |
| **FedPDA** | **η_g<1 via pseudo-gradient** | **버퍼 집계에서 η_g 의미 변화 분석** |

**인용 위치**:
- 2.4.2절(위성 토폴로지): FedMeld, FedSN
- 2.2절(비동기 FL): FedAsync, FedASMU
- 4장(η_g 분석): 세 논문 모두 차별화 기술

---

## 7. 논문 수정 지점

### 서론
- "궤도면 군집" 제거 → "구조적으로 높은 staleness"
- 연구 목표 3가지 재작성 (satellite-adapted + η_g 중심)

### 2.3절 (LEO 환경)
- "간헐적 통신, 구조적 staleness, 비대칭 데이터 수집"으로 특성 변경
- 2.3.1: Walker-Delta는 "결정론적 예측 가능성"만 기술

### 3.4.3절
- satellite-adapted 설명 강화
- η_g=1.0 "안정성" 표현 제거

### 4장 전면 재작성
- 4.1-4.4 수렴 분석
- 4.5 FedPDA 알고리즘 (η_g 중심)
- 4.6 ISL 확장

---

## 8. SCI 어필 판단

**수렴 증명 추가 시 기여 구조**:
1. 5전략 동일 조건 비교 (실험)
2. satellite-adapted η_g 의미 변화 규명 (분석)
3. 수렴 바운드 + 최적 η_g* 도출 (이론)
4. η_g sweep + ISL 검증 (실증)

수렴 증명 없이는 "파라미터 튜닝" 비판 회피 어려움. 증명 완성 후 SCI 투고 가능.

---

## 9. Orbital Data Center (향후 연구)

지상국 없이 위성군집 내 완결 FL. 마스터 위성(GPU 탑재)이 집계. 데이터: SSA/원격탐사/지상관측. 6장 향후 연구로 포함.

**설계 고려사항**:
- 마스터 위성 수/배치
- 연산량 추정 (모델 크기 × 집계 빈도)
- ISL 통신량 (LEO 100Mbps~10Gbps 기준)
- AI 모델 (U-Net 등 EO 분할, 256×256 패치)
- 본 연구의 η_g 이론이 그대로 적용 가능

---

## 10. TODO (Phase별)

### Phase 1: 실험 보완 (진행 중)
- [ ] FedPDA α=0.5, η_g=0.5, 0.3 sweep → η_g*가 σ_G²의 함수임을 실증

### Phase 2: 수렴 증명 완성
- [x] Lemma 1 완전 증명
- [ ] Theorem 1 (T라운드 텔레스코핑)
- [ ] Corollary 1 (최적 η_g*)
- [ ] Corollary 2 (η_g=1.0 한계)
- [ ] η_l·E<1 조건 정교화

### Phase 3: 논문 작성
- [ ] 1-3장 프레이밍 수정
- [ ] 4장 수렴 분석 작성
- [ ] 5장 실험 결과 (이론 검증 포함)
- [ ] 6장 결론 + Orbital Data Center

### Phase 4: Orbital Data Center (졸업 후)
- [ ] EO 데이터 파이프라인 조사
- [ ] AI 모델 선정 + 연산량 추정
- [ ] ISL 통신량 추정
- [ ] 마스터 배치 최적화

---

## 11. 파일 위치

- `/mnt/user-data/outputs/convergence_analysis_draft.tex`: 4장 수렴 분석 초안
- `/mnt/user-data/outputs/lemma1_full_proof.tex`: Lemma 1 완전 증명
- `/mnt/user-data/outputs/satellite_fedpda_isl.py`: ISL 통합 코드
- `/mnt/user-data/outputs/config_fedpda.py`: 설정 (η_g=0.5 권장)
