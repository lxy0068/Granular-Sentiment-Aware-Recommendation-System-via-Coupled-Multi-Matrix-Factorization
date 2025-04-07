import autograd.numpy as np
import random
from typing import Dict, List


def minibatch_sparse_cost(U, I, F, W, uiaw_samples, uw_freq_matrix,
                          ui_ratings, uia_sentiments, iaw_frequencies,
                          lambda_reg, lambda_r, lambda_s, lambda_o,
                          neg_rate, lambda_bpr, batch_size, verbose=False):
    """Autograd-compatible loss function with verified operations"""

    # 初始化损失分量
    loss_R = np.zeros(())
    loss_S = np.zeros(())
    loss_O = np.zeros(())
    loss_bpr = np.zeros(())

    # 维度验证
    max_dims = {
        'U': U.shape[0],
        'I': I.shape[0],
        'F': F.shape[0],
        'W': W.shape[0]
    }

    # 有效样本筛选
    valid_samples = []
    for sample in random.sample(uiaw_samples, min(batch_size, len(uiaw_samples))):
        try:
            parts = sample.strip('[]').split(',')
            if len(parts) != 4:
                continue

            u_id, i_id, a_id, w_id = map(int, parts)

            if (u_id < 0 or u_id >= max_dims['U'] or
                    i_id < 0 or i_id >= max_dims['I'] or
                    a_id < 0 or a_id >= max_dims['F'] or
                    w_id < 0 or w_id >= max_dims['W']):
                if verbose:
                    print(f"Invalid sample {sample}")
                continue

            valid_samples.append((u_id, i_id, a_id, w_id))

        except ValueError:
            continue

    # 核心计算
    for u_id, i_id, a_id, w_id in valid_samples:
        # 特征拼接
        item_feature = np.hstack((I[i_id], F[-1]))
        aspect_feature = np.hstack((I[i_id], F[a_id]))

        # 评分预测
        rating_pred = np.dot(U[u_id], item_feature)
        real_rating = ui_ratings.get(f"[{u_id},{i_id}]", 3.0)
        loss_R += (real_rating - rating_pred) ** 2

        # 情感预测
        senti_pred = np.dot(U[u_id], aspect_feature)
        real_senti = uia_sentiments.get(f"[{u_id},{i_id},{a_id}]", 0.0)
        loss_S += (real_senti - senti_pred) ** 2

    # 正则化项（保持计算图完整）
    reg_term = lambda_reg * (
            np.sqrt(np.mean(U ** 2)) +
            np.sqrt(np.mean(I ** 2)) +
            np.sqrt(np.mean(F ** 2))
    )

    # 总损失计算
    total_loss = (
            lambda_r * loss_R +
            lambda_s * loss_S +
            lambda_o * loss_O +
            lambda_bpr * loss_bpr +
            reg_term
    )

    # 安全打印（避免影响计算图）
    if verbose:
        print(f"\nValid samples: {len(valid_samples)}")
        _safe_print_loss(total_loss)

    return total_loss


def _safe_print_loss(loss_tensor):
    """安全提取损失值进行打印"""
    if hasattr(loss_tensor, '_value'):
        # 从ArrayBox中提取数值
        value = loss_tensor._value
    elif isinstance(loss_tensor, np.ndarray):
        # 普通numpy数组
        value = loss_tensor
    else:
        # 其他类型处理
        value = np.asarray(loss_tensor)

    print(f"Current Loss: {float(value):.4f}")