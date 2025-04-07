import numpy as np
import time
from autograd import grad
from Core.Loss import minibatch_sparse_cost
from Core.config import config


def construct_IF_W_vec(I, F, W, if_pair):
    """Constructs feature-word interaction vectors

    Args:
        I (ndarray): Item embedding matrix
        F (ndarray): Feature embedding matrix 
        W (ndarray): Word embedding matrix
        if_pair (tuple): (item_idx, feature_idx) pair

    Returns:
        ndarray: Combined interaction vector
    """
    A = np.hstack((I[if_pair[0]], F[if_pair[1]]))
    return np.einsum("a,ma->m", A, W)


def construct_IF_U_row(U, I, F, u_id, i_id, feature_pairs):
    """Calculates user-specific feature ratings

    Args:
        U (ndarray): User embedding matrix
        I (ndarray): Item embedding matrix
        F (ndarray): Feature embedding matrix
        u_id (int): User index
        i_id (int): Item index
        feature_pairs (list): List of feature indices

    Returns:
        dict: Feature ratings dictionary {feature_idx: rating}
    """
    ratings = {}
    for a in feature_pairs:
        A = np.hstack((I[i_id], F[a]))
        ratings[a] = np.einsum("a,a->", U[u_id], A)
    return ratings


def minibatch_adagradSGD_train(
    # 基础数据参数 (1-5)
    uiaw_list, uw_freq_mat, ui_rating_dict,
    uia_senti_dict, iaw_freq_dict,
    # 模型架构参数 (6-9)
    U_dim, I_dim, F_dim, W_dim,
    # 系统维度参数 (10-13)
    U_num, I_num, F_total, W_num,
    # 训练参数 (14-22)
    num_iter, lambda_reg, lambda_r, lambda_s, lambda_o,
    neg_rate, lambda_bpr, batch_size, lr,
    # 测试参数 (23-24)
    ui_rating_dic_test, uia_senti_dic_test,
    # 可选参数 (25-26)
    random_seed=0, eps=1e-8
):
    """Adaptive gradient descent trainer with automatic differentiation

    Args:
        uiaw_list (list): Training tuples (user, item, aspect, word)
        uw_freq_mat (ndarray): User-word frequency matrix
        ui_rating_dict (dict): User-item ratings
        uia_senti_dict (dict): User-item-aspect sentiment scores
        iaw_freq_dict (dict): Item-aspect-word frequencies
        U_dim (int): User embedding dimension
        I_dim (int): Item embedding dimension
        F_dim (int): Feature embedding dimension
        W_dim (int): Word embedding dimension
        U_num (int): Number of users
        I_num (int): Number of items
        F_total (int): Total features (+1 for special)
        W_num (int): Number of words
        num_iter (int): Training iterations
        lambda_reg (float): Regularization strength
        lambda_r (float): Rating loss weight
        lambda_s (float): Sentiment loss weight
        lambda_o (float): Frequency loss weight
        neg_rate (float): Negative sampling rate
        lambda_bpr (float): BPR loss weight
        batch_size (int): Mini-batch size
        lr (float): Base learning rate
        test_ratings (dict): Test set ratings
        test_senti (dict): Test set sentiment
        seed (int): Random seed
        eps (float): Numerical stability term

    Returns:
        tuple: Trained embedding matrices (U, I, F, W)
    """
    # Initialize random embeddings
    np.random.seed(random_seed)
    U = np.random.randn(U_num, U_dim) * 0.01
    I = np.random.randn(I_num, I_dim) * 0.01
    F = np.random.randn(F_total, F_dim) * 0.01
    W = np.random.randn(W_num, W_dim) * 0.01

    # Initialize gradient accumulators
    grad_accum_U = np.zeros_like(U)
    grad_accum_I = np.zeros_like(I)
    grad_accum_F = np.zeros_like(F)
    grad_accum_W = np.zeros_like(W)

    # Create gradient functions
    grad_U = grad(minibatch_sparse_cost, argnum=0)
    grad_I = grad(minibatch_sparse_cost, argnum=1)
    grad_F = grad(minibatch_sparse_cost, argnum=2)

    # Training loop
    for iter in range(num_iter):
        start_time = time.time()
        verbose = iter % 100 == 0

        # Calculate gradients
        delta_U = grad_U(U, I, F, W, uiaw_list, uw_freq_mat,
                         ui_rating_dict, uia_senti_dict, iaw_freq_dict,
                         lambda_reg, lambda_r, lambda_s, lambda_o,
                         neg_rate, lambda_bpr, batch_size, verbose)

        delta_I = grad_I(U, I, F, W, uiaw_list, uw_freq_mat,
                         ui_rating_dict, uia_senti_dict, iaw_freq_dict,
                         lambda_reg, lambda_r, lambda_s, lambda_o,
                         neg_rate, lambda_bpr, batch_size, verbose)

        delta_F = grad_F(U, I, F, W, uiaw_list, uw_freq_mat,
                         ui_rating_dict, uia_senti_dict, iaw_freq_dict,
                         lambda_reg, lambda_r, lambda_s, lambda_o,
                         neg_rate, lambda_bpr, batch_size, verbose)

        # Update gradient accumulators
        grad_accum_U += eps + np.square(delta_U)
        grad_accum_I += eps + np.square(delta_I)
        grad_accum_F += eps + np.square(delta_F)

        # Compute adaptive learning rates
        adapt_lr_U = lr / np.sqrt(grad_accum_U)
        adapt_lr_I = lr / np.sqrt(grad_accum_I)
        adapt_lr_F = lr / np.sqrt(grad_accum_F)

        # Update parameters
        U -= adapt_lr_U * delta_U
        I -= adapt_lr_I * delta_I
        F -= adapt_lr_F * delta_F

        # Non-negative projection
        U = np.maximum(U, 0)
        I = np.maximum(I, 0)
        F = np.maximum(F, 0)

        # Progress monitoring
        if iter % config.print_every_times == 0:
            epoch_time = time.time() - start_time

            # Generate predictions
            pred_matrix = U @ np.hstack((I, np.tile(F[-1], (I_num, 1)))).T

            # Build evaluation set
            eval_data = []
            for composite_key, true_rating in ui_rating_dic_test.items():
                u, i = map(int, composite_key[1:-1].split(','))
                eval_data.append([u, i, true_rating, pred_matrix[u, i]])

            # Calculate metrics
            metric = config.metric_class()
            mae = metric.MAE(eval_data)
            rmse = metric.RMSE(eval_data)

            # Write log
            with open(config.log_path, 'a') as f:
                f.write(f"Iteration {iter} - "
                        f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, "
                        f"Time: {epoch_time:.2f}s\n")

    return U, I, F, W