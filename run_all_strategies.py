#!/usr/bin/env python3
# run_all_strategies.py
# ============================================================
# 4개 비동기 FL 전략을 순차 실행하는 러너
# config.py의 AGGREGATION_STRATEGY를 자동으로 전환하며 실행
#
# 사용법:
#   python run_all_strategies.py              # 4개 전부
#   python run_all_strategies.py fedbuff fedorbit  # 지정 전략만
# ============================================================

import subprocess
import sys
import time
import re
from datetime import datetime, timedelta
from pathlib import Path

STRATEGIES = ["fedasync", "fedbuff", "fedspace", "fedorbit"]
CONFIG_PATH = Path("config.py")
SATELLITE_SCRIPT = "satellite.py"  # 실행할 메인 스크립트 경로


def set_strategy(strategy: str):
    """config.py의 AGGREGATION_STRATEGY 값을 교체"""
    text = CONFIG_PATH.read_text()
    text = re.sub(
        r'^AGGREGATION_STRATEGY\s*=\s*".*?"',
        f'AGGREGATION_STRATEGY = "{strategy}"',
        text,
        flags=re.MULTILINE
    )
    CONFIG_PATH.write_text(text)


def format_elapsed(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def run_strategy(strategy: str) -> dict:
    """단일 전략 실행 후 결과 반환"""
    set_strategy(strategy)

    print(f"\n{'='*60}")
    print(f"  [{strategy.upper()}] 시작 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, SATELLITE_SCRIPT],
        capture_output=False,  # 실시간 출력
    )
    elapsed = time.time() - t0

    status = "✅ 성공" if result.returncode == 0 else f"❌ 실패 (code={result.returncode})"
    print(f"\n  [{strategy.upper()}] {status} — 소요: {format_elapsed(elapsed)}\n")

    return {
        "strategy": strategy,
        "returncode": result.returncode,
        "elapsed_sec": elapsed,
    }


def main():
    # 실행할 전략 결정
    if len(sys.argv) > 1:
        targets = [s.lower() for s in sys.argv[1:]]
        invalid = [s for s in targets if s not in STRATEGIES]
        if invalid:
            print(f"오류: 알 수 없는 전략 {invalid}")
            print(f"사용 가능: {STRATEGIES}")
            sys.exit(1)
    else:
        targets = STRATEGIES

    print(f"\n{'#'*60}")
    print(f"  비동기 FL 전략 비교 실험")
    print(f"  전략: {', '.join(s.upper() for s in targets)}")
    print(f"  시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    results = []
    total_t0 = time.time()

    for strategy in targets:
        r = run_strategy(strategy)
        results.append(r)

        # 실패 시 계속할지 중단할지
        if r["returncode"] != 0:
            print(f"⚠️  {strategy.upper()} 실패. 다음 전략으로 계속합니다.\n")

    total_elapsed = time.time() - total_t0

    # 결과 요약
    print(f"\n{'#'*60}")
    print(f"  실험 완료 — 총 소요: {format_elapsed(total_elapsed)}")
    print(f"{'#'*60}")
    print(f"\n  {'전략':<12} {'상태':<8} {'소요 시간':<15}")
    print(f"  {'─'*35}")
    for r in results:
        status = "성공" if r["returncode"] == 0 else "실패"
        print(f"  {r['strategy'].upper():<12} {status:<8} {format_elapsed(r['elapsed_sec']):<15}")
    print()

    # 실패한 전략이 있으면 exit code 1
    if any(r["returncode"] != 0 for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()