# Experimental Design for Graph-FedMORL

## 1. Datasets & Heterogeneity
To rigorously evaluate the proposed framework, we employ datasets with varying degrees of complexity and heterogeneity.

### 1.1. Datasets
- **CIFAR-10**: Standard benchmark for image classification.
- **Fashion-MNIST / FEMNIST**: Selected specifically for fairness evaluation due to their inherent user heterogeneity.

### 1.2. Data Partitioning (Non-IID)
- **Dirichlet Distribution**: We simulate Non-IID settings by partitioning data using a Dirichlet distribution with concentration parameter $\alpha \in \{0.1, 0.5\}$.
    - $\alpha=0.1$: Extreme heterogeneity (clients have very few classes).
    - $\alpha=0.5$: Moderate heterogeneity.

## 2. Attack Scenarios
We evaluate robustness against 20% malicious clients under the following attack modes:

1.  **Label Flipping**: Malicious clients flip labels $y \to 9-y$.
2.  **Sign Flipping**: Malicious clients scale updates by a negative factor ($w = -|m|w'_{true}$).
3.  **IPM / ALIE (A Little Is Enough)**: Sophisticated attacks that add small, calculated noise to evade distance-based defenses while degrading the global model.

## 3. Baselines
We compare Graph-FedMORL against four groups of baselines:

1.  **Classical FL**: `FedAvg`, `FedProx`.
2.  **Robust FL**: `Krum`, `Median`, `Trimmed Mean`.
3.  **Adaptive/RL-based**: `FedAA` (The base method).
4.  **Fairness-aware FL**: `Ditto` (Personalization baseline).

## 4. Evaluation Metrics
1.  **Global Accuracy**: Test accuracy on the server's held-out validation set.
2.  **Fairness (StdDev)**: Standard deviation of test accuracy across all benign clients (Lower is better).
3.  **Worst 10% Accuracy**: Average accuracy of the bottom 10% performing clients (Higher is better). This metric specifically highlights the "participation fairness" enforced by our KL-regularization.

## 5. Research Questions (RQs) & Ablation Studies

### RQ1: Overall Performance
- **Hypothesis**: Graph-FedMORL outperforms FedAA and robust baselines in both Accuracy and Fairness metrics, especially under ALIE attacks.
- **Setup**: Run all methods with 20% attackers on CIFAR-10 ($\alpha=0.1$).

### RQ2: Component Analysis (Ablation)
We incrementally add components to the Base (FedAA) to verify their individual contributions:
- **M0**: Base (FedAA)
- **M1**: Base + **Innov 1 (KL-Regularized Actor)** -> Expect improved Fairness (lower StdDev).
- **M2**: Base + **Innov 2 (Multi-Objective Critic)** -> Expect better Pareto trade-off.
- **M3**: Base + **Innov 3 (GASR)** -> Expect significantly higher Robustness against ALIE/Sign-flipping.
- **M4**: Full Graph-FedMORL (M1+M2+M3).

### RQ3: Visualization
- **t-SNE Embedding**: Visualize the state representations learned by the GASR module.
- **Goal**: Demonstrate that GASR successfully clusters benign clients together while isolating malicious clients (even under stealthy attacks), whereas simple distance metrics fail to do so.
