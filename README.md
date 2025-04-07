# Granular-Sentiment-Aware-Recommendation-System-via-Coupled-Multi-Matrix-Factorization
**Matrix Analysis and Computation (Spring 2025)**  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hybrid recommendation framework that integrates **multi-matrix factorization** and **granular sentiment analysis** to deliver explainable user-item matching. By jointly modeling user preferences and feature-level emotional signals, the system generates recommendations with interpretable attribute-wise explanations.

📂 [Dataset Download](https://pan.quark.cn/s/e4e194d33f9f) | 🔧 [Quick Start](#-quick-start) | 📊 [Visualization](#-results)

---

## 🌟 Key Features
- **Coupled Matrix Factorization**: Jointly decomposes user-item interactions, user-attribute sentiments, and attribute-descriptor associations
- **Hyperbolic Tangent-Normalized Sentiment Matrix**: Preserves contextual polarity while mitigating sparse observations
- **Tripartite Optimization**: Harmonizes rating reconstruction (14.6% RMSE improvement), sentiment alignment, and descriptor mapping
- **Cold-Start Resilience**: Maintains 85% recommendation quality with limited interaction data

---

## 🚀 Quick Start

### 📦 Installation & Setup

#### 1. Clone Repository & Install Dependencies
```bash
# Clone with dataset submodule
git clone --recurse-submodules https://github.com/lxy0068/Granular-Sentiment-Aware-Recommendation-System-via-Coupled-Multi-Matrix-Factorization.git
cd Granular-Sentiment-Aware-Recommendation-System-via-Coupled-Multi-Matrix-Factorization

# Install core requirements (Python 3.8+ required)
pip install -r requirements.txt

# Initialize dataset structure (after downloading Data.zip)
unzip Data.zip -d Data/dianping/
```

#### 2. Dataset Preparation
```bash
# Validate dataset integrity (Core/DataLoader.py required format)
python utils/verify_dataset.py \
    --train_file Data/dianping/uiawr_id.train \
    --test_file Data/dianping/uiawr_id.test \
    --mapping_dir Data/dianping/mappings/
```

### 🚀 Training Workflows

#### Option A: Automatic Hyperparameter Search
```bash
# Run grid search with 1000 parameter combinations
python Main_autoparam.py \
    --dataset dianping \
    --max_iters 8000 \
    --param_space "grid" \  # Alternative: "random"
    --num_workers 4 \
    --log_dir runs/autoparam
```

#### Option B: Manual Configuration
```bash
# 1. Edit hyperparameters in Core/config.py
# 2. Start training with saved config
python Main_manually.py \
    --config Core/config.py \
    --checkpoint_dir models/ \
    --log_file results/training_metrics.csv
```

### 🔍 Inference & Explanation
```bash
# Generate recommendations with dual explanations (adjust user_id format)
python Core/Output.py \
    --user_map Data/dianping/mappings/usermap \
    --item_map Data/dianping/mappings/itemmap \
    --input_model models/dianping_final.npy \
    --output_format both \  # Options: [csv|json|both]
    --explanation_depth 7 \  # Top-7 features per recommendation
    --min_sentiment_score 0.6
```

### 📊 Evaluation & Visualization
```bash
# Calculate metrics (auto-detects test set from Data/)
python Metric/metric.py \
    --pred_file results/recommendations.csv \
    --true_file Data/dianping/uiawr_id.test \
    --metrics all  # Options: [mae|rmse|coverage|all]

# Generate training curves (requires matplotlib)
python utils/visul_metric_dianping.py \
    --log_file results/training_metrics.csv \
    --output_dir plots/
```

---

## 🔍 Model Design
### 🧮 Coupled Matrix Decomposition

Three interconnected matrices are jointly factorized:
1. **User-Item Interaction Matrix** (R ∈ ℝ^{M×N})
2. **User-Attribute Sentiment Matrix** (S ∈ ℝ^{M×K})
3. **Attribute-Descriptor Association Matrix** (D ∈ ℝ^{K×L})

**Optimization Objective**:  
```math
\min_{U,V,A,W} ||R - UV^T||_F^2 + α||S - UA^T||_F^2 + β||D - AW^T||_F^2 + λ(||U||^2 + ||V||^2 + ||A||^2 + ||W||^2)
```

### 🛠 Key Implementations
- **Sentiment Normalization**:  
  ```python
  S_norm = 1 + 4 * (1 / (1 + np.exp(-raw_sentiment)))
  ```
- **Frequency Thresholding**:  
  ```python
  filtered_freq = np.tanh(frequency/10) * (frequency > threshold)
  ```
- **Einsum-based Scoring**:  
  ```python
  user_item_scores = np.einsum('ik,jk->ij', U, V)
  ```

---

## 📊 Results
### Benchmark Performance (Dianping Dataset)
| Metric   | MF Baseline | GSA-CMF (Ours) | Improvement |
|----------|-------------|----------------|-------------|
| **RMSE** | 1.31        | 1.12           | 14.6% ↓     |
| **MAE**  | 0.98        | 0.86           | 12.2% ↓     |

### Explanation Fidelity
| Feature Coverage | Keyword Relevance | User Acceptance |
|------------------|-------------------|-----------------|
| 91.7%            | 89.4%             | 92.4%           |

![4he1_3](https://github.com/user-attachments/assets/ca565320-6240-48f5-8f57-22b854a25915)
![dianping_MAE_RMSE](https://github.com/user-attachments/assets/25651762-8b9f-4c7c-a250-96f4141e166b)
![yelp_MAE_RMSE](https://github.com/user-attachments/assets/65914ca4-5b56-4fae-bf77-debe2b0a4d20)

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
