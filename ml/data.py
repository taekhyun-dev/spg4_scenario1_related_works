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