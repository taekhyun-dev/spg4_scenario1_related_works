#!/usr/bin/env python3
# run_fedpda_sweep.py
# ============================================================
# satellite_fedpda.py (ISL 미사용 버전)로 sweep을 병렬 실행
#
# 주요 사용 사례:
#   - Plain FedPDA (ISL 없는 버전) sweep
#   - ISL 효과 분석용 baseline 생성
#
# satellite_fedpda_isl.py 대신 satellite_fedpda.py를 호출하는 것 외에는
# run_parallel_sweep.py와 동일한 인터페이스.
#
# 사용법:
#   python run_fedpda_sweep.py                    # FedPDA 전체 sweep
#   python run_fedpda_sweep.py --jobs 4
#   python run_fedpda_sweep.py --seeds 42 123
#   python run_fedpda_sweep.py --strategies fedpda fedbuff
#   python run_fedpda_sweep.py --dry-run
# ============================================================

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

STRATEGIES_ALL = ["fedasync", "fedbuff", "fedspace", "fedorbit", "fedpda"]
SATELLITE_SCRIPT = "satellite_fedpda.py"   # ★ ISL 없는 버전
LOG_DIR_ROOT = Path("logs/sweep_fedpda_plain")


def format_elapsed(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def alpha_tag(alpha: float) -> str:
    return f"A{int(alpha * 10):02d}"


def eta_tag(eta: float) -> str:
    return f"E{int(eta * 10):02d}"


def run_one(job):
    """단일 실험 (strategy, seed, alpha, eta_g) 실행"""
    strategy, seed, alpha, eta_g = job
    if strategy == "fedpda" and eta_g is not None:
        tag = f"{strategy}_S{seed}_{alpha_tag(alpha)}_{eta_tag(eta_g)}"
    else:
        tag = f"{strategy}_S{seed}_{alpha_tag(alpha)}"

    LOG_DIR_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR_ROOT / f"{tag}.log"

    env = {
        **os.environ,
        "ORBITAL_FL_STRATEGY": strategy,
        "ORBITAL_FL_SEED": str(seed),
        "ORBITAL_FL_ALPHA": str(alpha),
    }
    if strategy == "fedpda" and eta_g is not None:
        env["ORBITAL_FL_ETA_G"] = str(eta_g)

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
        "eta_g": eta_g,
        "returncode": result.returncode,
        "elapsed_sec": elapsed,
        "log_path": str(log_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description=f"Plain FedPDA sweep 러너 ({SATELLITE_SCRIPT} 사용, ISL 미사용)"
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
        "--alphas", nargs="+", type=float, default=[0.1, 0.5, 1.0],
        help="alpha 목록 (기본: 0.1 0.5 1.0)"
    )
    parser.add_argument(
        "--strategies", nargs="+", default=["fedpda"],
        choices=STRATEGIES_ALL,
        help="전략 목록 (기본: fedpda만)"
    )
    parser.add_argument(
        "--eta-gs", nargs="+", type=float, default=None,
        help="η_g 목록 (fedpda 전용). 예: --eta-gs 0.1 0.3 0.5 0.7 1.0"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="명령만 출력하고 실행 안 함"
    )
    args = parser.parse_args()

    # 작업 목록 생성
    jobs = []
    for strategy, seed, alpha in product(args.strategies, args.seeds, args.alphas):
        if strategy == "fedpda" and args.eta_gs:
            for eta in args.eta_gs:
                jobs.append((strategy, seed, alpha, eta))
        else:
            jobs.append((strategy, seed, alpha, None))
    total = len(jobs)

    print(f"\n{'#'*70}")
    print(f"  Plain FedPDA Sweep 시작 ({SATELLITE_SCRIPT}, ISL 미사용)")
    print(f"{'#'*70}")
    print(f"  전략     : {args.strategies}")
    print(f"  시드     : {args.seeds}")
    print(f"  α        : {args.alphas}")
    if args.eta_gs:
        print(f"  η_g      : {args.eta_gs}  (fedpda 전용)")
    print(f"  총 실험  : {total}")
    print(f"  병렬     : {args.jobs}개 동시 실행")
    print(f"  로그 경로: {LOG_DIR_ROOT}/")
    print(f"  결과 경로: results/{{strategy}}_S{{SEED}}_A{{α}}/  (ISL 태그 없음)")
    print(f"  시작     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")

    if args.dry_run:
        print("[Dry-run] 실행할 명령:")
        for strategy, seed, alpha, eta_g in jobs:
            eta_env = f"ORBITAL_FL_ETA_G={eta_g} " if eta_g is not None else ""
            print(
                f"  ORBITAL_FL_STRATEGY={strategy} "
                f"ORBITAL_FL_SEED={seed} ORBITAL_FL_ALPHA={alpha} "
                f"{eta_env}python {SATELLITE_SCRIPT}"
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
