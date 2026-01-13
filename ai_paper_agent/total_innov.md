# Innovation 1: KL-Regularized Aggregation for Explicit Fairness

## 1. Motivation
The baseline method, FedAA, relies on a Deep Deterministic Policy Gradient (DDPG) agent to dynamically assign aggregation weights to clients. The reward function is defined solely as the validation accuracy on the server ($r = \text{Acc}(w_g, D_g)$).

**Critical Flaw**: 
This formulation assumes that maximizing global accuracy implicitly guarantees fairness. However, in highly heterogeneous Non-IID settings, a greedy RL agent often exhibits **"Client Collapse"**: it assigns near-zero weights to clients with "hard" data distributions to quickly maximize rewards on "easy" classes. This leads to:
1.  **Poor Fairness**: Benign clients with difficult data are ignored.
2.  **Overfitting**: The global model overfits to the dominant clients, reducing generalization.

**Goal**: 
To introduce an explicit regularization mechanism that forces the RL agent to maintain a diverse portfolio of client contributions, thereby mathematically guaranteeing participation fairness while retaining the capability to filter out malicious attackers.

## 2. Mathematical Formulation

We propose modifying the Actor's objective function by introducing a **Kullback-Leibler (KL) Divergence** penalty term. This term penalizes the deviation of the learned weight distribution $a$ from a uniform distribution $U$.

### 2.1. The New Objective
Let $a = \pi(s|\theta^\pi)$ be the aggregation weight vector output by the Actor, where $a \in \mathbb{R}^M, \sum_{i=1}^M a_i = 1, a_i \ge 0$.
Let $U = [\frac{1}{M}, \dots, \frac{1}{M}]$ be the uniform distribution over the selected $M$ clients.

The original Actor loss is:
$$ \mathcal{L}_{orig}(\theta^\pi) = - \mathbb{E}_{s} [Q(s, \pi(s))] $$

The proposed **KL-Regularized Loss** is:
$$ \mathcal{L}_{new}(\theta^\pi) = - \mathbb{E}_{s} [Q(s, \pi(s))] + \lambda \cdot D_{KL}(a || U) $$

### 2.2. Derivation
Expanding the KL term:
$$ D_{KL}(a || U) = \sum_{i=1}^M a_i \log \frac{a_i}{1/M} = \sum_{i=1}^M a_i \log a_i - \sum_{i=1}^M a_i \log(1/M) $$
Since $\sum a_i = 1$ and $\log(1/M)$ is a constant, minimizing $D_{KL}$ is equivalent to minimizing the **Negative Entropy**:
$$ \min D_{KL}(a || U) \iff \min \sum_{i=1}^M a_i \log a_i \iff \max H(a) $$

Thus, the effective loss function for implementation is:
$$ \mathcal{L}_{new}(\theta^\pi) = - Q(s, a) + \lambda \sum_{i=1}^M a_i \log (a_i + \epsilon) $$
*(Note: $\epsilon$ is added for numerical stability)*

## 3. Theoretical Analysis

### 3.1. Gradient Dynamics & The Barrier Effect
Differentiating the regularization term w.r.t a specific weight $a_k$:
$$ \frac{\partial}{\partial a_k} (\lambda \sum a_i \log a_i) = \lambda (1 + \log a_k) $$

Combining this with the Q-value gradient:
$$ \nabla_{a_k} \mathcal{L}_{new} = - \frac{\partial Q}{\partial a_k} + \lambda (1 + \log a_k) $$

**Interpretation**:
- **The Q-term ($-\frac{\partial Q}{\partial a_k}$)**: Pushes $a_k$ up if client $k$ improves global accuracy.
- **The Regularization term ($\lambda \log a_k$)**: Acts as a **Soft Barrier**. As $a_k \to 0$, $\log a_k \to -\infty$. This creates a strong negative gradient in the loss (or positive force on the weight) that prevents $a_k$ from collapsing to absolute zero.
- **Result**: This theoretically guarantees that every selected client retains a non-zero influence on the global model, ensuring **Participation Fairness**.

### 3.2. Robustness vs. Fairness Trade-off
One might worry that this forces the inclusion of attackers. However:
- For an attacker, $\frac{\partial Q}{\partial a_k}$ is likely large and positive (meaning increasing this weight *increases* the loss / *decreases* the Q-value drastically).
- As long as the penalty from the Q-critic (detecting performance drop) outweighs the entropy benefit ($\lambda \log a_k$), the attacker's weight will still be suppressed to a very small value, though not strictly zero.
- $\lambda$ acts as a tunable "Fairness Tolerance" parameter.

## 4. Expected Improvement
1.  **Fairness**: Significant reduction in the variance of test accuracy across benign clients compared to vanilla FedAA.
2.  **Generalization**: Improved convergence speed and final accuracy on "hard" classes that were previously under-represented.
3.  **Stability**: The entropy term acts as a regularizer for the policy network, preventing the DDPG agent from getting stuck in deterministic local optima.

# Innovation 2: Multi-Objective Critic for Pareto-Optimal Aggregation

## 1. Motivation
In the original FedAA framework, the Critic network $Q(s, a)$ estimates a scalar reward $r = \text{Acc}(w_g, D_g)$. This scalarization collapses the multi-dimensional nature of Federated Learning objectives (Accuracy vs. Fairness) into a single number *before* the learning process begins. 

While Innovation 1 (KL-Regularized Aggregation) imposes a "soft" constraint on the policy distribution, it does not explicitly model the environment's response to fairness. The agent doesn't "know" if an action is unfair; it is just penalized for low entropy. 

**The Gap**: The RL agent lacks a distinct understanding of how its actions affect fairness separate from accuracy. It cannot navigate the Pareto frontier dynamically.

## 2. Mathematical Formulation

### 2.1. Vector-Valued Reward Signal
Since the server cannot access client local data to measure fairness directly, we use **Per-Class Accuracy Variance** on the server's balanced validation set $D_g$ as a proxy. In Non-IID settings, overfitting to specific clients manifests as high variance across class performance.

We define the reward vector $\mathbf{r}(t) \in \mathbb{R}^2$:
$$ \mathbf{r}(t) = \begin{bmatrix} r_{perf} \\ r_{fair} \end{bmatrix} = \begin{bmatrix} \text{Acc}(w_g, D_g) \\ 1 - \sigma(\{\text{Acc}_c(w_g, D_g)\}_{c \in C}) \end{bmatrix} $$
where $\sigma(\cdot)$ denotes the standard deviation across classes $C$.

### 2.2. Vector-Valued Critic Network
We extend the Critic to estimate a vector Q-function $\mathbf{Q}(s, a)$:
$$ \mathbf{Q}(s, a|\theta^Q) = \begin{bmatrix} Q_{perf}(s, a) \\ Q_{fair}(s, a) \end{bmatrix} $$

The Critic loss is the sum of component-wise TD errors:
$$ \mathcal{L}_{Critic} = \mathbb{E} \left[ \sum_{i \in \{perf, fair\}} (y_i - Q_i(s, a))^2 \right] $$
where $y_i = r_i + \gamma Q_i(s', \pi(s'))$.

### 2.3. Scalarized Actor Update
To update the Actor $\pi(s|\theta^\pi)$, we use a preference vector $\mathbf{\omega} = [\omega_1, \omega_2]^T$ (where $\omega_1 + \omega_2 = 1$) to scalarize the Q-values dynamically:

$$ \nabla_{\theta^\pi} J = \mathbb{E}_{s} \left[ \nabla_a (\mathbf{\omega}^T \mathbf{Q}(s, a)) |_{a=\pi(s)} \cdot \nabla_{\theta^\pi} \pi(s) \right] $$

## 3. Theoretical Analysis
*   **Decoupling**: Unlike a fixed reward function $r = Acc + \lambda Fair$, the Vector Critic learns the dynamics of Accuracy and Fairness independently. This prevents the "catastrophic interference" where the gradient for accuracy overwhelms the gradient for fairness during backpropagation.
*   **Pareto Stationarity**: By optimizing the scalarized objective $\mathbf{\omega}^T \mathbf{Q}$, the policy is guaranteed to converge to a solution on the Pareto Frontier defined by $\mathbf{\omega}$.

## 4. Expected Improvement
*   **Controllability**: Allows dynamic adjustment of priorities during training (e.g., prioritize $r_{perf}$ early for fast convergence, then shift $\omega$ towards $r_{fair}$ to balance the model).
*   **Robustness**: The explicit $r_{fair}$ signal (class variance) acts as a strong detector for "mode collapse" attacks where the model forgets specific classes.

# Innovation 3: Graph Attention State Representation (GASR)

## 1. Motivation
The current state representation in FedAA, $s = [d_1, ..., d_M]$, relies solely on the scalar Euclidean distance of client updates from the global model. This representation suffers from a critical **Information Bottleneck**:
1.  **Loss of Directionality**: It cannot distinguish between a benign update and a malicious "sign-flipping" update that has the same magnitude of distance but an opposite direction.
2.  **Independence Assumption**: It treats each client's distance independently, failing to capture the **correlations** or **collusion patterns** among attackers (e.g., a group of attackers sending similar malicious updates).

To address this, we propose modeling the federated aggregation process as a **Graph Learning** problem, where the RL agent perceives the "topology" of client updates rather than just their magnitude.

## 2. Mathematical Formulation

### 2.1. Graph Construction
We construct a fully connected graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ where:
- **Nodes ($\mathcal{V}$)**: Each selected client $k \in \{1, ..., M\}$ is a node.
- **Node Features ($h_k^{(0)}$)**: A compressed representation of the client's model update $\Delta w_k$.
  $$ h_k^{(0)} = \text{PCA}(\Delta w_k) \quad \text{or} \quad \text{Layer}_{\text{last}}(\Delta w_k) $$

### 2.2. Graph Attention Mechanism (GAT)
We employ a Graph Attention Network (GAT) to learn the relationships between clients. The attention coefficient $e_{ij}$ indicates the importance of client $j$'s features to client $i$:

$$ e_{ij} = \text{LeakyReLU}(\mathbf{a}^T [ \mathbf{W} h_i^{(0)} || \mathbf{W} h_j^{(0)} ]) $$

The normalized attention weights $\alpha_{ij}$ are computed via Softmax:
$$ \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})} $$

The updated node embedding $h_i^{(1)}$ aggregates information from neighbors, effectively capturing the "consensus" or "outlier" status of each client in the context of others:
$$ h_i^{(1)} = \sigma \left( \sum_{j \in \mathcal{N}_i} \alpha_{ij} \mathbf{W} h_j^{(0)} \right) $$

### 2.3. State Generation
The final state $s$ fed to the RL agent is a global readout of the graph embeddings, ensuring the agent sees the full relational structure:
$$ s = \text{GlobalMeanPooling}(\{ h_1^{(1)}, ..., h_M^{(1)} \}) \quad \oplus \quad \text{GlobalMaxPooling}(\{ h_1^{(1)}, ..., h_M^{(1)} \}) $$

## 3. Theoretical Analysis
- **Permutation Invariance**: Like the original FedAA, the GNN-based state is invariant to the ordering of clients, which is essential for FL.
- **Collusion Detection**: The attention mechanism $\alpha_{ij}$ explicitly models pairwise similarities. If a group of attackers sends similar updates, they will form a strong "clique" in the graph. The GAT layers will amplify these features, allowing the RL agent to identify and down-weight the entire group simultaneously.

## 4. Expected Improvement
- **Robustness**: Significantly higher defense rate against **Sign-flipping** and **IPM (Inner Product Manipulation)** attacks compared to distance-based baselines.
- **Information Gain**: Provides the RL agent with a richer, structure-aware context, accelerating convergence.

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
