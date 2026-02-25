# ml/aggregation.py
import torch
import torch.nn as nn
import numpy as np
from typing import List, OrderedDict

"""
Aggregation 함수
이 부분에서 전체 모델의 성능 차이가 발생함.
연구의 핵심 부분
"""
def fed_avg(models_to_average: List[OrderedDict]) -> OrderedDict:
    """
    Federated Averaging 알고리즘을 수행.
    여러 모델의 가중치(state_dict) 리스트를 받아 각 가중치의 평균을 계산하여 반환.
    """
    if not models_to_average: return OrderedDict()
    avg_state_dict = OrderedDict()
    for key in models_to_average[0].keys():
        tensor_ref = models_to_average[0][key]
        if not tensor_ref.is_floating_point():
            # non-float (num_batches_tracked 등): 첫 번째 모델 값 사용
            avg_state_dict[key] = tensor_ref.clone()
            continue
        tensors = [model[key].float() for model in models_to_average]
        avg_tensor = torch.stack(tensors).mean(dim=0)
        avg_state_dict[key] = avg_tensor.to(tensor_ref.dtype)
    return avg_state_dict

def calculate_mixing_weight(local_ver, global_ver, local_acc, global_acc, 
                            local_data_count, avg_data_count):
    """
    레거시 코드의 동적 가중치 계산 로직 이식
    """
    BASE_ALPHA = 0.5  # 기본 반영 비율
    
    # 1. Staleness (버전 차이) 패널티
    staleness = max(0, global_ver - local_ver)
    staleness_factor = 1.0 / (1.0 + staleness * 0.5)

    performance_factor = 1.0

    # 3. Data Volume (데이터 양) 가중치
    if avg_data_count > 0:
        data_ratio = local_data_count / avg_data_count
        data_factor = np.clip(data_ratio, 0.5, 1.5)
    else:
        data_factor = 1.0

    # 최종 Alpha 계산
    final_alpha = BASE_ALPHA * staleness_factor * performance_factor * data_factor

    return min(final_alpha, 0.6), staleness_factor, performance_factor, data_factor

def weighted_update(global_state_dict: OrderedDict, local_state_dict: OrderedDict, alpha: float, device: str = 'cpu') -> OrderedDict:
    """
    기존 글로벌 모델과 로컬 모델을 alpha 비율로 섞는 Weighted Update를 수행합니다.
    
    수식:
        w_new = (1 - alpha) * w_global + alpha * w_local
    
    BatchNorm의 num_batches_tracked(int64) 등 non-float 텐서는
    가중 평균 대상에서 제외하고 글로벌 값을 그대로 유지합니다.
    float 파라미터도 연산 후 원본 dtype으로 복원합니다.
    """
    updated_state_dict = OrderedDict()
    
    for key in global_state_dict.keys():
        global_param = global_state_dict[key]
        
        # non-float (num_batches_tracked 등): 가중 평균하지 않음
        if not global_param.is_floating_point():
            updated_state_dict[key] = global_param.clone().cpu()
            continue
        
        if key in local_state_dict:
            local_param = local_state_dict[key]
            
            # float32로 올려서 연산 후 원본 dtype 복원
            g = global_param.to(device).float()
            l = local_param.to(device).float()
            updated_param = (1.0 - alpha) * g + alpha * l
            
            updated_state_dict[key] = updated_param.to(global_param.dtype).cpu()
        else:
            updated_state_dict[key] = global_param.clone().cpu()
            
    return updated_state_dict