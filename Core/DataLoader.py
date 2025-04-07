import _thread
import math
import random
import time
import numpy as np
from sklearn.cluster import KMeans
from Core.config import config


def get_index(key):
    """解析字典键中的索引元组

    Args:
        key (str): 格式为 '[i,a,w]' 的字符串

    Returns:
        tuple: 包含 (i_idx, a_idx, w_idx) 的整数元组
    """
    index = key[1:-1].split(',')
    return tuple(map(int, index))


def dataprocess():
    """数据处理与特征工程主函数

    完成以下任务：
    1. 读取原始数据文件
    2. 构建特征矩阵和统计字典
    3. 执行特征标准化和非线性变换
    4. 返回处理后的数据结构

    Returns:
        tuple: 包含以下元素的元组：
            - uiaw_list (list): 训练集四元组记录
            - uw_frequency_mat (ndarray): 用户-词频次矩阵
            - ui_rating_dic (dict): 用户-物品评分字典
            - uia_senti_dic_train (dict): 训练情感得分字典
            - iaw_frequency_dic (dict): 物品-特征-词频次字典
            - ui_rating_dic_test (dict): 测试集评分字典
            - word_dic (dict): 词表映射
            - aspect_dic (dict): 特征映射
            - iaw_frequency_test_dic (dict): 测试集频次字典
            - uia_senti_dic_test (dict): 测试情感得分字典
    """

    # ======================
    # 第一阶段：数据扫描确定维度
    # ======================
    def scan_max_ids(file_path):
        """扫描文件获取最大ID值

        Args:
            file_path (str): 数据文件路径

        Returns:
            tuple: (max_u, max_i, max_a, max_w)
        """
        max_u = max_i = max_a = max_w = 0
        with open(file_path, 'r', encoding='UTF-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    max_u = max(max_u, int(parts[0]))
                    max_i = max(max_i, int(parts[1]))
                    max_a = max(max_a, int(parts[2]))
                    max_w = max(max_w, int(parts[3]))
        return max_u, max_i, max_a, max_w

    # 获取全局最大ID
    train_max = scan_max_ids(f"./Data/{config.dataset_name}/uiawr_id.train")
    test_max = scan_max_ids(f"./Data/{config.dataset_name}/uiawr_id.test")

    # 动态计算维度
    U_num = max(train_max[0], test_max[0]) + 1
    I_num = max(train_max[1], test_max[1]) + 1
    F_num = max(train_max[2], test_max[2]) + 1
    W_num = max(train_max[3], test_max[3]) + 1

    # ======================
    # 第二阶段：数据加载与特征构建
    # ======================
    # 初始化数据结构
    uw_frequency_mat = np.zeros((U_num, W_num), dtype=np.float32)
    ui_rating_dic = {}
    uia_senti_dic_train = {}
    iaw_frequency_dic = {}
    ui_rating_dic_test = {}
    iaw_frequency_test_dic = {}
    uia_senti_dic_test = {}
    uiaw_list = []

    # 加载元数据
    aspect_dic = {}
    with open(f"./Data/{config.dataset_name}/aspect.map", 'r', encoding='UTF-8') as f:
        for line in f:
            k, v = line.strip().split('=', 1)
            aspect_dic[int(k)] = v

    word_dic = {}
    word_senti_dic = {}
    with open(f"./Data/{config.dataset_name}/word.senti.map", 'r', encoding='UTF-8') as f:
        for line in f:
            parts = line.strip().split('=', 2)
            word_dic[parts[0]] = parts[1]
            word_senti_dic[int(parts[0])] = int(parts[2])

    # ======================
    # 第三阶段：训练数据处理
    # ======================
    with open(f"./Data/{config.dataset_name}/uiawr_id.train", 'r', encoding='UTF-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 5:
                continue  # 跳过格式错误行

            u_idx, i_idx, a_idx, w_idx = map(int, parts[:4])
            rating = float(parts[4])

            # 记录原始数据
            uiaw_list.append(f"[{u_idx},{i_idx},{a_idx},{w_idx}]")

            # 构建用户-物品评分矩阵
            ui_key = f"[{u_idx},{i_idx}]"
            ui_rating_dic[ui_key] = rating

            # 构建用户-词频次矩阵
            uw_frequency_mat[u_idx, w_idx] += 1

            # 构建物品-特征-词频次统计
            iaw_key = f"[{i_idx},{a_idx},{w_idx}]"
            iaw_frequency_dic[iaw_key] = iaw_frequency_dic.get(iaw_key, 0) + 1

            # 累计情感得分
            uia_key = f"[{u_idx},{i_idx},{a_idx}]"
            current = uia_senti_dic_train.get(uia_key, 0)
            current += word_senti_dic[w_idx]
            uia_senti_dic_train[uia_key] = current

    # ======================
    # 第四阶段：测试数据处理
    # ======================
    with open(f"./Data/{config.dataset_name}/uiawr_id.test", 'r', encoding='UTF-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 5:
                continue

            u_idx, i_idx, a_idx, w_idx = map(int, parts[:4])
            rating = float(parts[4])

            # 测试集评分记录
            ui_key = f"[{u_idx},{i_idx}]"
            ui_rating_dic_test[ui_key] = rating

            # 测试集频次统计
            iaw_key = f"[{i_idx},{a_idx},{w_idx}]"
            iaw_frequency_test_dic[iaw_key] = iaw_frequency_test_dic.get(iaw_key, 0) + 1

            # 测试集情感得分
            uia_key = f"[{u_idx},{i_idx},{a_idx}]"
            current = uia_senti_dic_test.get(uia_key, 0)
            current += word_senti_dic[w_idx]
            uia_senti_dic_test[uia_key] = current

    # ======================
    # 第五阶段：特征工程
    # ======================
    # 情感得分Sigmoid标准化
    def sigmoid_transform(values_dict):
        """应用Sigmoid标准化到字典值"""
        for key in values_dict:
            raw = values_dict[key]
            values_dict[key] = 1 + 4 / (1 + np.exp(-raw))

    sigmoid_transform(uia_senti_dic_train)
    sigmoid_transform(uia_senti_dic_test)

    # 频次特征双曲正切变换
    def tanh_transform(values_dict):
        """应用缩放后的tanh变换"""
        for key in values_dict:
            x = values_dict[key] / 20.0
            transformed = 5 * (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

            # 处理负面词和低频词
            i, a, w = get_index(key)
            if word_senti_dic[w] < 0 or transformed < 0.5:
                transformed = 0

            values_dict[key] = transformed

    tanh_transform(iaw_frequency_dic)

    return (
        uiaw_list, uw_frequency_mat, ui_rating_dic, uia_senti_dic_train,
        iaw_frequency_dic, ui_rating_dic_test, word_dic, aspect_dic,
        iaw_frequency_test_dic, uia_senti_dic_test
    )