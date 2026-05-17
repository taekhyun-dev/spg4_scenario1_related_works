import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from collections import Counter
import os


def get_cifar10_loaders(num_clients: int, dirichlet_alpha: float = 0.5,
                        data_root: str = './data', batch_size_val: int = 256,
                        num_workers: int = 8, samples_per_client: int = 2000):
    """
    CIFAR-10 데이터셋을 다운로드하고, 각 클라이언트(위성)에게
    Dirichlet 분포 기반 Non-IID 데이터를 **독립 샘플링**합니다.

    기존 방식(split)과의 차이:
      - split: 50,000장을 N등분 → 위성당 ~210장 (N=238)
      - sample: 위성마다 독립적으로 samples_per_client장을 Dirichlet 비율로 샘플링
               → 위성 간 데이터 중복 허용 (위성들이 유사 지역 촬영하는 현실 반영)

    Args:
        num_clients: 클라이언트(위성) 수
        dirichlet_alpha: Non-IID 강도 (작을수록 편향 ↑, 0.5 = moderate)
        data_root: 데이터 저장 경로
        batch_size_val: 검증 배치 크기
        num_workers: DataLoader 워커 수
        samples_per_client: 위성당 학습 데이터 수 (기본 2000)
    """

    # 1. CIFAR-10 전용 정규화 값 (Mean, Std)
    CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR_STD  = (0.2023, 0.1994, 0.2010)

    # 2. 전처리 파이프라인 정의
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    print(f"📥 [Data] CIFAR-10 데이터셋 로드 중... (Root: {data_root})")

    # 3. 데이터셋 다운로드 및 로드
    train_dataset = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test
    )

    # 4. 클래스별 인덱스 사전 구축
    targets = np.array(train_dataset.targets)
    num_classes = 10
    class_indices = {k: np.where(targets == k)[0] for k in range(num_classes)}

    print(
        f"⚖️ [Data] Dirichlet(α={dirichlet_alpha}) 독립 샘플링: "
        f"{num_clients}개 위성 × {samples_per_client}장/위성"
    )

    # 5. 위성마다 독립적으로 Dirichlet 샘플링
    client_subsets = []
    total_data_count = 0

    for i in range(num_clients):
        # (a) Dirichlet 분포로 이 위성의 클래스 비율 생성
        class_probs = np.random.dirichlet(np.repeat(dirichlet_alpha, num_classes))

        # (b) 비율에 따라 클래스별 샘플 수 결정
        class_counts = np.round(class_probs * samples_per_client).astype(int)

        # 반올림 오차 보정: 총합이 samples_per_client와 다를 수 있음
        diff = samples_per_client - class_counts.sum()
        if diff != 0:
            # 가장 비율이 큰 클래스에서 조정
            max_class = np.argmax(class_counts)
            class_counts[max_class] += diff

        # 각 클래스에서 최소 1개는 보장하지 않음 (Non-IID 특성 유지)
        # 단, 음수 방지
        class_counts = np.maximum(class_counts, 0)

        # (c) 클래스별로 중복 허용 랜덤 샘플링
        selected_indices = []
        for k in range(num_classes):
            n_samples = class_counts[k]
            if n_samples == 0:
                continue
            pool = class_indices[k]
            # replace=True: 중복 허용 (위성 간 + 위성 내 클래스 내)
            # 위성 내 중복은 augmentation이 다르므로 실질적으로 다른 샘플
            sampled = np.random.choice(pool, size=n_samples, replace=True)
            selected_indices.extend(sampled)

        np.random.shuffle(selected_indices)
        subset = Subset(train_dataset, selected_indices)
        client_subsets.append(subset)
        total_data_count += len(selected_indices)

    avg_data_count = total_data_count / num_clients

    # 6. Global Validation Loader 생성
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # (디버깅) 분할 결과 요약 출력 (첫 5개 위성)
    print(
        f"📊 샘플링 완료: 위성당 {samples_per_client}장 "
        f"(총 {total_data_count:,}장, 중복 허용)"
    )
    for i in range(min(5, num_clients)):
        indices = [client_subsets[i].indices[j]
                   for j in range(len(client_subsets[i]))]
        labels = [targets[idx] for idx in indices]
        counts = Counter(labels)
        dist_str = ' '.join(f"{k}:{v}" for k, v in sorted(counts.items()))
        print(f"  - SAT_{i}: {len(indices)} samples [{dist_str}]")

    return avg_data_count, client_subsets, val_loader, train_dataset.classes


def get_eurosat_loaders(num_clients: int, dirichlet_alpha: float = 0.5,
                        data_root: str = './data', batch_size_val: int = 256,
                        num_workers: int = 8, samples_per_client: int = 1500,
                        image_size: int = 32, val_ratio: float = 0.2,
                        split_seed: int = 42):
    """
    EuroSAT (Sentinel-2 Earth Observation) 데이터셋 Dirichlet 비IID 샘플링.

    - 원본: 27,000장 × 64×64 RGB × 10 classes (균형 분포)
    - 클래스: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial,
              Pasture, PermanentCrop, Residential, River, SeaLake
    - 64×64 → image_size (기본 32)로 리사이즈하여 ResNet-9 호환
    - torchvision에 train/test split이 없어 수동 분할 (val_ratio = 20%)

    Args:
        num_clients: 클라이언트(위성) 수
        dirichlet_alpha: Non-IID 강도
        data_root: 데이터 저장 경로
        batch_size_val: 검증 배치 크기
        num_workers: DataLoader 워커 수
        samples_per_client: 위성당 학습 데이터 수 (기본 1500, EuroSAT 크기 고려)
        image_size: 리사이즈 목표 크기 (기본 32 = ResNet-9 호환)
        val_ratio: 전체에서 validation으로 분할할 비율
        split_seed: train/val 분할 시드 (실험 시드와 별개, 고정 권장)

    Returns:
        (avg_data_count, client_subsets, val_loader, classes)
        — CIFAR-10 로더와 동일한 인터페이스
    """
    # EuroSAT RGB 통계 (Sentinel-2 RGB 채널, 일반적으로 인용되는 값)
    EUROSAT_MEAN = (0.3444, 0.3803, 0.4078)
    EUROSAT_STD = (0.2037, 0.1366, 0.1148)
    NUM_CLASSES = 10

    # 전처리 파이프라인
    transform_train = transforms.Compose([
        transforms.Resize(image_size),                     # 64 → 32
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(EUROSAT_MEAN, EUROSAT_STD),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(EUROSAT_MEAN, EUROSAT_STD),
    ])

    print(f"📥 [Data] EuroSAT 데이터셋 로드 중... (Root: {data_root})")

    # 전체 데이터셋: train/val 분할용 (transform이 다른 두 인스턴스)
    full_train = datasets.EuroSAT(
        root=data_root, transform=transform_train, download=True
    )
    full_val = datasets.EuroSAT(
        root=data_root, transform=transform_test, download=True
    )

    n_total = len(full_train)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    # 타겟 추출 (torchvision EuroSAT은 ImageFolder 기반)
    if hasattr(full_train, 'targets'):
        all_targets = np.array(full_train.targets)
    elif hasattr(full_train, '_labels'):
        all_targets = np.array(full_train._labels)
    else:
        # 마지막 fallback: samples는 (path, label) 튜플 리스트
        all_targets = np.array([s[1] for s in full_train.samples])

    # train/val 인덱스 분할 (고정 시드 - 모든 실험에서 동일)
    rng = np.random.RandomState(split_seed)
    all_indices = np.arange(n_total)
    rng.shuffle(all_indices)
    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train:].tolist()

    # 클래스별 train 인덱스 사전 (Dirichlet 샘플링용)
    train_targets = all_targets[train_indices]
    class_indices = {
        k: train_indices[train_targets == k] for k in range(NUM_CLASSES)
    }

    # 데이터셋 크기 진단
    print(
        f"⚖️ [Data] Dirichlet(α={dirichlet_alpha}) 독립 샘플링: "
        f"{num_clients}개 위성 × {samples_per_client}장/위성  "
        f"(train pool: {n_train}, val: {n_val})"
    )

    # 위성별 Dirichlet 샘플링 (CIFAR-10 로더와 동일 로직)
    client_subsets = []
    total_data_count = 0

    for i in range(num_clients):
        class_probs = np.random.dirichlet(np.repeat(dirichlet_alpha, NUM_CLASSES))
        class_counts = np.round(class_probs * samples_per_client).astype(int)

        # 반올림 오차 보정
        diff = samples_per_client - class_counts.sum()
        if diff != 0:
            max_class = np.argmax(class_counts)
            class_counts[max_class] += diff
        class_counts = np.maximum(class_counts, 0)

        selected_indices = []
        for k in range(NUM_CLASSES):
            n_samples = class_counts[k]
            if n_samples == 0:
                continue
            pool = class_indices[k]
            if len(pool) == 0:
                continue
            sampled = np.random.choice(pool, size=n_samples, replace=True)
            selected_indices.extend(sampled)

        np.random.shuffle(selected_indices)
        subset = Subset(full_train, selected_indices)
        client_subsets.append(subset)
        total_data_count += len(selected_indices)

    avg_data_count = total_data_count / num_clients

    # Validation loader (test transform 적용된 인스턴스의 val 인덱스 부분)
    val_subset = Subset(full_val, val_indices)
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 클래스 이름
    classes = getattr(full_train, 'classes',
                      [f"class_{i}" for i in range(NUM_CLASSES)])

    print(
        f"📊 EuroSAT 샘플링 완료: 위성당 {samples_per_client}장 "
        f"(총 {total_data_count:,}장, 중복 허용)"
    )
    for i in range(min(5, num_clients)):
        indices = [client_subsets[i].indices[j]
                   for j in range(len(client_subsets[i]))]
        labels = [all_targets[idx] for idx in indices]
        counts = Counter(labels)
        dist_str = ' '.join(f"{k}:{v}" for k, v in sorted(counts.items()))
        print(f"  - SAT_{i}: {len(indices)} samples [{dist_str}]")

    return avg_data_count, client_subsets, val_loader, classes