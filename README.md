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
- **Dual-Modal Explanations**: Generates radar charts and text summaries for interpretable recommendations

---

## 🚀 Quick Start

### 📦 Installation
```bash
# Clone repository
git clone https://github.com/your_username/granular-sentiment-recommendation.git
cd granular-sentiment-recommendation

# Install dependencies
pip install -r requirements.txt
```

### 🏃 Training & Inference
```bash
# Train model on Dianping dataset (8000 iterations)
python Core/Train.py --dataset dianping --num_iter 8000 --lr 0.1

# Generate Top-100 recommendations for user 123
python Core/Output.py --user_id 123 --top_k 100 --output_dir results/
```

---

## 🧩 Project Architecture
### 📂 Directory Structure
```
├── Core/                   # Core algorithms
│   ├── config.py           # Hyperparameter configurations
│   ├── DataLoader.py       # User-item-feature-word interaction loader
│   ├── Loss.py             # Multi-task loss function (MSE + MAE + L2)
│   └── Train.py            # Adagrad optimizer with non-negative constraints
│
├── Data/                   # Dataset management
│   ├── dianping/           # Chinese review dataset
│   │   ├── uiawr_id.train  # Training interactions
│   │   └── word.senti.map  # Precomputed sentiment lexicon
│   └── yelp/               # Cross-cultural validation dataset
│
├── utils/                  # Data processing & visualization
│   ├── dianping_datapro.py # Feature engineering pipeline
│   ├── transDataToNpy.py   # Matrix format conversion
│   └── visul_metric_*.py   # Performance visualization tools
│
└── results/                # Recommendation outputs
```

---

## 🔍 Model Design
### 🧮 Coupled Matrix Decomposition
<img src="https://via.placeholder.com/600x200?text=Architecture+Diagram" width="600">

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

---

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
