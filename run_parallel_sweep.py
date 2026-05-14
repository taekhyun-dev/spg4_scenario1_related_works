#!/usr/bin/env python3
# run_parallel_sweep.py
# ============================================================
# 5개 전략 × N SEED × M α sweep을 동시에 K개씩 병렬 실행
#
# 메모리 64GB + RTX 4090 (24GB VRAM) 환경 기준
#   기본값: K=3 (1프로세스당 ~10GB RAM, ~2GB VRAM)
#
# 사용법:
#   python run_parallel_sweep.py                     # 기본 sweep
#   python run_parallel_sweep.py --jobs 4            # 4개 병렬
#   python run_parallel_sweep.py --seeds 42 123      # 시드 지정
#   python run_parallel_sweep.py --alphas 0.1 0.5    # alpha 지정
#   python run_parallel_sweep.py --strategies fedpda fedbuff
#   python run_parallel_sweep.py --dry-run           # 명령만 출력
# ============================================================

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

STRATEGIES_ALL = ["fedasync", "fedbuff", "fedspace", "fedorbit", "fedpda"]
SATELLITE_SCRIPT = "satellite_fedpda_isl.py"
LOG_DIR_ROOT = Path("logs/sweep_parallel")


def format_elapsed(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def alpha_tag(alpha: float) -> str:
    return f"A{int(alpha * 10):02d}"


def run_one(job):
    """단일 실험 (strategy, seed, alpha) 실행"""
    strategy, seed, alpha = job
    tag = f"{strategy}_S{seed}_{alpha_tag(alpha)}"

    LOG_DIR_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR_ROOT / f"{tag}.log"

    env = {
        **os.environ,
        "ORBITAL_FL_STRATEGY": strategy,
        "ORBITAL_FL_SEED": str(seed),
        "ORBITAL_FL_ALPHA": str(alpha),
    }

    t0 = time.time()
    with open(log_path, "w") as f:
        result = subprocess.run(
            [sys.executable, SATELLITE_SCRIPT],
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - t0

    return {
        "tag": tag,
        "strategy": strategy,
        "seed": seed,
        "alpha": alpha,
        "returncode": result.returncode,
        "elapsed_sec": elapsed,
        "log_path": str(log_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="병렬 sweep 러너 (5 전략 × N 시드 × M α)"
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=3,
        help="동시 실행 프로세스 수 (기본 3, 64GB RAM 기준)"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 123, 7777],
        help="시드 목록 (기본: 42 123 7777)"
    )
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=[0.01, 0.1, 0.5, 1.0],
        help="alpha 목록 (기본: 0.01 0.1 0.5 1.0)"
    )
    parser.add_argument(
        "--strategies", nargs="+", default=STRATEGIES_ALL,
        choices=STRATEGIES_ALL,
        help=f"전략 목록 (기본: 모두)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="명령만 출력하고 실행 안 함"
    )
    args = parser.parse_args()

    # 작업 목록 생성
    jobs = list(product(args.strategies, args.seeds, args.alphas))
    total = len(jobs)

    print(f"\n{'#'*70}")
    print(f"  병렬 Sweep 시작")
    print(f"{'#'*70}")
    print(f"  전략     : {args.strategies}")
    print(f"  시드     : {args.seeds}")
    print(f"  α        : {args.alphas}")
    print(f"  총 실험  : {total}")
    print(f"  병렬     : {args.jobs}개 동시 실행")
    print(f"  로그 경로: {LOG_DIR_ROOT}/")
    print(f"  시작     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")

    if args.dry_run:
        print("[Dry-run] 실행할 명령:")
        for strategy, seed, alpha in jobs:
            print(
                f"  ORBITAL_FL_STRATEGY={strategy} "
                f"ORBITAL_FL_SEED={seed} ORBITAL_FL_ALPHA={alpha} "
                f"python {SATELLITE_SCRIPT}"
            )
        return

    results = []
    total_t0 = time.time()
    completed = 0

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        future_to_job = {executor.submit(run_one, job): job for job in jobs}

        for future in as_completed(future_to_job):
            r = future.result()
            results.append(r)
            completed += 1
            status = "✅" if r["returncode"] == 0 else "❌"
            print(
                f"  [{completed}/{total}] {status} {r['tag']} "
                f"({format_elapsed(r['elapsed_sec'])}) → {r['log_path']}"
            )

    total_elapsed = time.time() - total_t0

    # 결과 요약
    print(f"\n{'#'*70}")
    print(f"  Sweep 완료 — 총 소요: {format_elapsed(total_elapsed)}")
    print(f"{'#'*70}\n")

    successes = sum(1 for r in results if r["returncode"] == 0)
    failures = total - successes
    print(f"  성공: {successes}/{total}, 실패: {failures}")

    if failures > 0:
        print(f"\n  실패한 작업:")
        for r in results:
            if r["returncode"] != 0:
                print(f"    ❌ {r['tag']} → {r['log_path']}")

    # 전략별 평균 시간
    print(f"\n  [전략별 평균 시간]")
    for strategy in args.strategies:
        s_results = [r for r in results if r["strategy"] == strategy]
        if s_results:
            avg = sum(r["elapsed_sec"] for r in s_results) / len(s_results)
            print(f"    {strategy:<12} {format_elapsed(avg):<15} (N={len(s_results)})")

    if failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
