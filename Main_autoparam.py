"""
Hyperparameter Optimization System (Robust Version)
--------------------------------------------------
Implements complete protection against index errors
"""
import datetime
import random
import numpy as np
from Core.Train import minibatch_adagradSGD_train
from Core.DataLoader import dataprocess

# Global system dimensions with type hints
USER_COUNT: int = None
ITEM_COUNT: int = None
FEATURE_COUNT: int = None
WORD_COUNT: int = None

class SafeEvaluator:
    """Dimension-aware prediction handler"""
    def __init__(self, U, I):
        self.max_user = U.shape[0] - 1
        self.max_item = I.shape[0] - 1
        self.pred_matrix = self._create_pred_matrix(U, I)

    def _create_pred_matrix(self, U, I):
        """Safe matrix construction with validation"""
        assert U.shape[1] == I.shape[1], "Embedding dimension mismatch"
        return U @ I.T

    def get_prediction(self, user_id, item_id):
        """Bounds-checked prediction access"""
        if 0 <= user_id <= self.max_user and 0 <= item_id <= self.max_item:
            return self.pred_matrix[user_id, item_id]
        return np.nan


def initialize_system_dimensions(data_package):
    """修正后的维度初始化函数"""
    global USER_COUNT, ITEM_COUNT, FEATURE_COUNT, WORD_COUNT

    try:
        # 获取矩阵数据结构
        uw_freq_matrix = data_package[1]  # 用户-词频矩阵（numpy数组）
        I_matrix = data_package[3]["matrix"]  # 物品矩阵从字典中提取
        F_matrix = data_package[7]["features"]  # 特征矩阵从字典中提取

        # 验证数据结构类型
        assert isinstance(uw_freq_matrix, np.ndarray), "uw_freq_matrix应为numpy数组"
        assert isinstance(I_matrix, np.ndarray), "I_matrix应为numpy数组"
        assert isinstance(F_matrix, np.ndarray), "F_matrix应为numpy数组"

        # 设置全局维度
        USER_COUNT, WORD_COUNT = uw_freq_matrix.shape
        ITEM_COUNT = I_matrix.shape[0]
        FEATURE_COUNT = F_matrix.shape[0]

    except (KeyError, IndexError, AttributeError) as e:
        raise RuntimeError(f"维度初始化失败: {str(e)}")

def filter_test_samples(raw_test_data):
    """Pre-filter test samples to valid index ranges"""
    valid_samples = {}
    invalid_count = 0

    for key, value in raw_test_data.items():
        try:
            # 解析并验证ID
            u, i = map(int, key.strip('[]').split(','))
            if 0 <= u < USER_COUNT and 0 <= i < ITEM_COUNT:
                valid_samples[key] = value
            else:
                invalid_count += 1
        except ValueError:
            invalid_count += 1

    print(f"测试数据过滤: 有效样本={len(valid_samples)}, 无效样本={invalid_count}")
    return valid_samples

def execute_optimization_workflow(data_package):
    """Main workflow with data validation"""
    # 解包原始数据
    (training_samples, uw_matrix, rating_dict, sentiment_dict,
     frequency_dict, test_ratings, _, _, test_sentiments) = data_package

    # 预处理测试数据
    filtered_test_data = filter_test_samples(test_ratings)

    for trial_number in range(1000):
        conduct_full_trial(
            trial_number,
            training_samples,
            uw_matrix,
            rating_dict,
            sentiment_dict,
            frequency_dict,
            filtered_test_data,  # 使用过滤后的数据
            test_sentiments
        )

def conduct_full_trial(trial_number, *data_components):
    """Safe trial execution with parameter validation"""
    print(f"\n=== 执行试验 {trial_number + 1}/1000 ===")

    (samples, uw_data, ratings, sentiments,
     frequencies, test_ratings, test_sentiment) = data_components

    # 试验配置
    trial_config = {
        'model_params': {
            'U_dim': 24, 'I_dim': 12,
            'F_dim': 12, 'W_dim': 24
        },
        'training_params': {
            'num_iter': 6000,
            'lambda_reg': 10 * round(random.random(), 2),
            'lambda_r': 1,
            'lambda_s': round(random.random(), 2),
            'lambda_o': round(random.random(), 2),
            'neg_rate': 0,
            'lambda_bpr': 0,
            'batch_size': 200,
            'lr': 0.1,
            'eps': 1e-8,
            'random_seed': 0
        }
    }

    # 执行训练流程
    model = minibatch_adagradSGD_train(
        uiaw_list=samples,
        uw_freq_mat=uw_data,
        ui_rating_dict=ratings,
        uia_senti_dict=sentiments,
        iaw_freq_dict=frequencies,
        U_dim=24,
        I_dim=12,
        F_dim=12,
        W_dim=24,
        U_num=USER_COUNT,
        I_num=ITEM_COUNT,
        F_total=FEATURE_COUNT + 1,
        W_num=WORD_COUNT,
        num_iter=6000,
        lambda_reg=trial_config['training_params']['lambda_reg'],
        lambda_r=1,
        lambda_s=trial_config['training_params']['lambda_s'],
        lambda_o=trial_config['training_params']['lambda_o'],
        neg_rate=0,
        lambda_bpr=0,
        batch_size=200,
        lr=0.1,
        ui_rating_dic_test=test_ratings,
        uia_senti_dic_test=test_sentiment,
        random_seed=0,
        eps=1e-8
    )

    # 处理试验结果
    process_trial_results(model, test_ratings, trial_config)

def process_trial_results(model, test_data, config):
    """Safe result processing with validation"""
    U, I, F, _ = model
    evaluator = SafeEvaluator(U, I)

    # 构建评估数据集
    evaluation_data = []
    valid_samples = 0
    for composite_key, true_score in test_data.items():
        try:
            u, i = map(int, composite_key.strip('[]').split(','))
            pred_score = evaluator.get_prediction(u, i)
            if not np.isnan(pred_score):
                evaluation_data.append([u, i, true_score, pred_score])
                valid_samples += 1
        except (ValueError, IndexError):
            continue

    print(f"有效评估样本: {valid_samples}/{len(test_data)}")

    # 计算评估指标
    from Metric import metric
    evaluator = metric.Metric()
    if evaluation_data:
        mae_score = evaluator.MAE(evaluation_data)
        rmse_score = evaluator.RMSE(evaluation_data)
    else:
        mae_score = rmse_score = np.nan
        print("警告: 无有效评估数据")

    # 保存结果
    save_trial_record(config, mae_score, rmse_score)

def save_trial_record(config, mae, rmse):
    """Enhanced logging with dimensional context"""
    log_entry = (
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"模型维度: U_dim={config['model_params']['U_dim']}, I_dim={config['model_params']['I_dim']}\n"
        f"正则系数: lambda_reg={config['training_params']['lambda_reg']:.2f}\n"
        f"评估指标: MAE={mae:.4f}, RMSE={rmse:.4f}\n"
        "----------------------------------------\n"
    )

    with open("optimization_log.txt", "a", encoding="utf-8") as f:
        f.write(log_entry)

if __name__ == "__main__":
    try:
        # 初始化数据系统
        complete_dataset = dataprocess()
        initialize_system_dimensions(complete_dataset)

        # 维度一致性验证
        assert ITEM_COUNT == complete_dataset[3].shape[0], "物品矩阵维度不一致"

        # 执行优化流程
        execute_optimization_workflow(complete_dataset)

    except Exception as e:
        print(f"系统运行失败: {str(e)}")
        exit(1)