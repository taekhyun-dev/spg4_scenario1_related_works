# ml/training.py
from torchmetrics import JaccardIndex
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.amp import autocast, GradScaler
from .model import create_mobilenet, create_resnet9
from config import FEDPROX_MU

IMAGENET_CLASSES = 1000
CIFAR_CLASSES = 10

def train_model(model, global_state_dict, train_loader, epochs=1, lr=0.01, device='cuda', sim_logger=None):
    """ResNet9 로컬 학습 (FedProx 포함). AMP(FP16) + cuDNN benchmark로 가속."""
    try:
        loader_length = len(train_loader)
        sim_logger.info(f"✅ [Train] 배치 개수: {loader_length}")
        if loader_length == 0:
            sim_logger.error("⚠️ DataLoader가 비어있습니다.")
            return 0
    except Exception as e:
        sim_logger.error(f"❌ DataLoader 길이 확인 에러: {e}")
        return 0

    model.to(device)
    model.train()

    global_model = create_resnet9(num_classes=CIFAR_CLASSES)
    global_model.load_state_dict(global_state_dict)
    global_model.to(device)
    global_model.eval()
    for param in global_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler('cuda')

    samples_count = 0

    for epoch in range(epochs):
        sim_logger.info(f"              에포크 {epoch+1}/{epochs} 진행 중...")
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                prox_term = 0.0
                for local_param, global_param in zip(model.parameters(), global_model.parameters()):
                    prox_term = prox_term + torch.sum((local_param - global_param.detach()) ** 2)
                total_loss = loss + (FEDPROX_MU / 2) * prox_term

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            samples_count += labels.size(0)

        scheduler.step()

    if sim_logger:
        sim_logger.info(f"             🧠 학습 완료 (Samples: {samples_count})")

    model.to('cpu')
    del global_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return samples_count

def evaluate_model(model_state_dict, data_loader, device):
    """주어진 모델 가중치와 데이터로더로 정확도와 손실을 평가"""
    model = create_resnet9(num_classes=CIFAR_CLASSES)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()

    jaccard = JaccardIndex(task="multiclass", num_classes=IMAGENET_CLASSES).to(device)

    total_loss = 0.0
    correct_1 = 0
    correct_5 = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            total += labels.size(0)

            # --- Top-k Accuracy 계산 ---
            # ImageNet은 Top-5가 중요한 지표임
            _, pred = outputs.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))

            # Top-1
            correct_1 += correct[:1].reshape(-1).float().sum().item()
            # Top-5
            correct_5 += correct[:5].reshape(-1).float().sum().item()

            # mIoU (Top-1 기준)
            jaccard.update(pred[0], labels)
            
    acc1 = 100 * correct_1 / total
    acc5 = 100 * correct_5 / total
    avg_loss = total_loss / len(data_loader)
    miou = jaccard.compute().item() * 100

    model.to('cpu')
    
    return acc1, avg_loss, miou