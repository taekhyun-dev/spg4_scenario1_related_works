#!/bin/bash
# ============================================================
# FedPDA η_g Sweep — α=0.1 우선, 최적값 후 α=0.5 검증
# ============================================================
# 사용법:
#   chmod +x run_eta_sweep.sh
#   ./run_eta_sweep.sh
#
# 전제조건:
#   - satellite_fedpda.py, config_fedpda.py 가 현재 디렉토리에 있어야 함
#   - config_fedpda.py 에 FEDPDA_SERVER_LR, DIRICHLET_ALPHA 변수가 있어야 함
# ============================================================

set -e

SIMULATOR="satellite_fedpda.py"
CONFIG="config_fedpda.py"
CONFIG_BACKUP="config_fedpda.py.backup"
RESULTS_BASE="results/eta_sweep"

# η_g 후보값 (α=0.1 먼저)
ETA_VALUES=(0.5 0.3 0.1)
ALPHA_PHASE1=0.1
ALPHA_PHASE2=0.5

# ── 안전 확인 ──
if [ ! -f "$SIMULATOR" ]; then
    echo "❌ $SIMULATOR 없음. 경로 확인 필요."
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "❌ $CONFIG 없음. 경로 확인 필요."
    exit 1
fi

# 원본 config 백업
cp "$CONFIG" "$CONFIG_BACKUP"
echo "✅ config 백업: $CONFIG_BACKUP"

# ── 함수: config 수정 ──
set_config() {
    local eta=$1
    local alpha=$2
    
    # FEDPDA_SERVER_LR 교체
    sed -i "s/^FEDPDA_SERVER_LR\s*=.*/FEDPDA_SERVER_LR = ${eta}/" "$CONFIG"
    
    # DIRICHLET_ALPHA 교체
    sed -i "s/^DIRICHLET_ALPHA\s*=.*/DIRICHLET_ALPHA = ${alpha}/" "$CONFIG"
    
    echo "  → FEDPDA_SERVER_LR = ${eta}, DIRICHLET_ALPHA = ${alpha}"
}

# ── 함수: 결과 이동 ──
move_results() {
    local eta=$1
    local alpha=$2
    local dest="${RESULTS_BASE}/eta${eta}_alpha${alpha}"
    
    mkdir -p "$dest"
    
    # 기본 출력 경로 (simulator가 results/fedpda_eta_tuning/ 에 저장)
    local src="results/fedpda_eta_tuning"
    if [ -d "$src" ]; then
        cp "$src"/* "$dest/" 2>/dev/null || true
        echo "  → 결과 저장: $dest"
    else
        echo "  ⚠️  $src 없음. 결과 경로 확인 필요."
    fi
    
    # performance CSV, simulation log (최신 파일)
    local latest_perf=$(ls -t performance_*.csv 2>/dev/null | head -1)
    local latest_log=$(ls -t simulation_*.log 2>/dev/null | head -1)
    [ -n "$latest_perf" ] && cp "$latest_perf" "$dest/"
    [ -n "$latest_log" ] && cp "$latest_log" "$dest/"
}

# ============================================================
# Phase 1: α=0.1에서 η_g sweep
# ============================================================
echo ""
echo "════════════════════════════════════════════════"
echo " Phase 1: α=${ALPHA_PHASE1} — η_g sweep"
echo "════════════════════════════════════════════════"

for eta in "${ETA_VALUES[@]}"; do
    echo ""
    echo "── η_g=${eta}, α=${ALPHA_PHASE1} ──"
    
    # 이미 결과 있으면 스킵
    dest="${RESULTS_BASE}/eta${eta}_alpha${ALPHA_PHASE1}"
    if [ -f "${dest}/fedpda_metrics.json" ]; then
        echo "  ⏭️  이미 완료. 스킵."
        continue
    fi
    
    set_config "$eta" "$ALPHA_PHASE1"
    
    echo "  🚀 시뮬레이션 시작: $(date '+%Y-%m-%d %H:%M:%S')"
    python3 "$SIMULATOR" --strategy fedpda
    echo "  ✅ 시뮬레이션 완료: $(date '+%Y-%m-%d %H:%M:%S')"
    
    move_results "$eta" "$ALPHA_PHASE1"
done

# ============================================================
# Phase 1 분석 → 최적 η_g 선택
# ============================================================
echo ""
echo "════════════════════════════════════════════════"
echo " Phase 1 분석"
echo "════════════════════════════════════════════════"

python3 << 'PYEOF'
import json, os, sys
import numpy as np

base = "results/eta_sweep"
alpha = "0.1"

# 기존 결과 포함
results = {
    1.0: {"best": 75.31, "final": 67.82, "late_std": 5.21, "late_mean": 68.45},
    0.7: {"best": 77.16, "final": 70.61, "late_std": 3.23, "late_mean": 72.61},
}

for eta_str in ["0.5", "0.3", "0.1"]:
    path = f"{base}/eta{eta_str}_alpha{alpha}/fedpda_metrics.json"
    if not os.path.exists(path):
        print(f"  ⚠️  η_g={eta_str} 결과 없음: {path}")
        continue
    with open(path) as f:
        m = json.load(f)
    acc = [h['accuracy'] for h in m['accuracy_history']]
    n = len(acc)
    late = acc[n*2//3:]
    results[float(eta_str)] = {
        "best": m['best_accuracy'],
        "final": m['final_accuracy'],
        "late_std": round(np.std(late), 2),
        "late_mean": round(np.mean(late), 2),
    }

print(f"\n{'η_g':>5} {'보존률':>6} {'최고':>7} {'최종':>7} {'후반mean':>9} {'후반std':>8}")
print("-" * 50)
for eta in sorted(results.keys(), reverse=True):
    r = results[eta]
    pres = f"{(1-eta)*100:.0f}%"
    print(f"{eta:5.1f} {pres:>6} {r['best']:7.2f} {r['final']:7.2f} {r['late_mean']:9.2f} {r['late_std']:8.2f}")

# 최적 η_g 추천 (최종 정확도 기준)
best_eta = max(results.keys(), key=lambda e: results[e]['final'])
print(f"\n🏆 최종 정확도 기준 최적 η_g = {best_eta}")
print(f"   → Phase 2에서 α=0.5로 검증 필요")

# 최적값 파일로 저장
with open(f"{base}/best_eta.txt", "w") as f:
    f.write(str(best_eta))
PYEOF

# ============================================================
# Phase 2: 최적 η_g를 α=0.5에서 검증
# ============================================================
BEST_ETA=$(cat "${RESULTS_BASE}/best_eta.txt" 2>/dev/null || echo "")

if [ -z "$BEST_ETA" ]; then
    echo "⚠️  Phase 1 분석 실패. Phase 2 수동 진행 필요."
else
    # 이미 0.7로 α=0.5 실험은 완료했으므로, 다른 값이면 실행
    if [ "$BEST_ETA" != "0.7" ]; then
        echo ""
        echo "════════════════════════════════════════════════"
        echo " Phase 2: α=${ALPHA_PHASE2} — η_g=${BEST_ETA} 검증"
        echo "════════════════════════════════════════════════"
        
        dest="${RESULTS_BASE}/eta${BEST_ETA}_alpha${ALPHA_PHASE2}"
        if [ -f "${dest}/fedpda_metrics.json" ]; then
            echo "  ⏭️  이미 완료."
        else
            set_config "$BEST_ETA" "$ALPHA_PHASE2"
            echo "  🚀 시뮬레이션 시작: $(date '+%Y-%m-%d %H:%M:%S')"
            python3 "$SIMULATOR" --strategy fedpda
            echo "  ✅ 시뮬레이션 완료: $(date '+%Y-%m-%d %H:%M:%S')"
            move_results "$BEST_ETA" "$ALPHA_PHASE2"
        fi
    else
        echo ""
        echo "✅ 최적 η_g=0.7 — α=0.5 검증은 이미 완료."
    fi
fi

# ── config 복원 ──
cp "$CONFIG_BACKUP" "$CONFIG"
echo ""
echo "✅ config 원본 복원 완료."
echo "🎯 전체 sweep 완료. 분석: python3 analyze_eta_sweep.py"
