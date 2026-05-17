#!/usr/bin/env python3
# run_orbital_sweep.py
# ============================================================
# satellite_orbital.py (Orbital Data Center FL) sweep을 병렬 실행
#
# 지원하는 변수:
#   - SEED (시드)
#   - ALPHA (Dirichlet 비IID 강도)
#   - ETA_G (FedPDA server LR)
#   - DATASET (cifar10 | eurosat)
#
# 사용법:
#   python run_orbital_sweep.py                          # 기본 sweep
#   python run_orbital_sweep.py --jobs 4
#   python run_orbital_sweep.py --datasets eurosat
#   python run_orbital_sweep.py --datasets cifar10 eurosat  # 양쪽 모두
#   python run_orbital_sweep.py --eta-gs 0.1 0.3 0.5 0.7 1.0
#   python run_orbital_sweep.py --dry-run
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

SATELLITE_SCRIPT = "satellite_orbital.py"
LOG_DIR_ROOT = Path("logs/sweep_orbital")

DATASET_TAG_MAP = {"cifar10": "C10", "eurosat": "ES"}


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
    """단일 실험 (dataset, seed, alpha, eta_g) 실행"""
    dataset, seed, alpha, eta_g = job
    dtag = DATASET_TAG_MAP[dataset]
    tag = f"{dtag}_S{seed}_{alpha_tag(alpha)}"
    if eta_g is not None:
        tag += f"_{eta_tag(eta_g)}"

    LOG_DIR_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR_ROOT / f"{tag}.log"

    env = {
        **os.environ,
        "ORBITAL_FL_DATASET": dataset,
        "ORBITAL_FL_SEED": str(seed),
        "ORBITAL_FL_ALPHA": str(alpha),
    }
    if eta_g is not None:
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
        "dataset": dataset,
        "seed": seed,
        "alpha": alpha,
        "eta_g": eta_g,
        "returncode": result.returncode,
        "elapsed_sec": elapsed,
        "log_path": str(log_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description=f"Orbital FL sweep 러너 ({SATELLITE_SCRIPT})"
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
        "--eta-gs", nargs="+", type=float, default=None,
        help="η_g 목록 (미지정 시 config 기본값 0.37 사용). "
             "예: --eta-gs 0.1 0.3 0.5 0.7 1.0"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["cifar10"],
        choices=["cifar10", "eurosat"],
        help="데이터셋 목록 (기본: cifar10만). 예: --datasets cifar10 eurosat"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="명령만 출력하고 실행 안 함"
    )
    args = parser.parse_args()

    # 작업 목록 생성: (dataset, seed, alpha, eta_g)
    jobs = []
    eta_gs = args.eta_gs if args.eta_gs else [None]
    for dataset, seed, alpha, eta in product(
        args.datasets, args.seeds, args.alphas, eta_gs
    ):
        jobs.append((dataset, seed, alpha, eta))
    total = len(jobs)

    print(f"\n{'#'*70}")
    print(f"  Orbital FL Sweep 시작 ({SATELLITE_SCRIPT})")
    print(f"{'#'*70}")
    print(f"  데이터셋  : {args.datasets}")
    print(f"  시드      : {args.seeds}")
    print(f"  α         : {args.alphas}")
    if args.eta_gs:
        print(f"  η_g       : {args.eta_gs}")
    else:
        print(f"  η_g       : config 기본값")
    print(f"  총 실험   : {total}")
    print(f"  병렬      : {args.jobs}개 동시 실행")
    print(f"  로그 경로 : {LOG_DIR_ROOT}/")
    print(f"  결과 경로 : results/orbital_fl_{{tag}}/")
    print(f"  시작      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")

    if args.dry_run:
        print("[Dry-run] 실행할 명령:")
        for dataset, seed, alpha, eta_g in jobs:
            eta_env = f"ORBITAL_FL_ETA_G={eta_g} " if eta_g is not None else ""
            print(
                f"  ORBITAL_FL_DATASET={dataset} "
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

    # 데이터셋별 평균 시간
    print(f"\n  [데이터셋별 평균 시간]")
    for dataset in args.datasets:
        d_results = [r for r in results if r["dataset"] == dataset]
        if d_results:
            avg = sum(r["elapsed_sec"] for r in d_results) / len(d_results)
            print(f"    {dataset:<10} {format_elapsed(avg):<15} (N={len(d_results)})")

    if failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
