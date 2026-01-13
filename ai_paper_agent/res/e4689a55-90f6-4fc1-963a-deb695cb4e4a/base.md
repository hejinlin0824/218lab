# FAVOR: Optimizing Federated Learning on Non-IID Data with Reinforcement Learning

## 1. Problem Definition
**Core Pain Point**: In Federated Learning (FL), data across mobile devices is typically **Non-IID** (Non-Independent and Identically Distributed).
- **Gap**: Standard algorithms like **FedAvg** randomly select clients for aggregation. When data is Non-IID, the local gradients/weights diverge significantly from the global objective. Random selection fails to counterbalance this bias, leading to:
    1.  **Slow Convergence**: Requires significantly more communication rounds to reach target accuracy.
    2.  **Instability**: Model accuracy fluctuates or diverges.
- **Objective**: Design an active client selection strategy that intelligently chooses a subset of devices $K$ from total $N$ devices in each round to maximize global model improvement and minimize communication rounds.

## 2. Core Methodology
The authors propose **FAVOR**, a Deep Reinforcement Learning (DRL) based control framework.

### 2.1. Mathematical Intuition
The paper posits an implicit connection between the data distribution $p^{(k)}$ on device $k$ and its trained model weights $w^{(k)}$.
**Weight Divergence Bound**:
$$ \| w_1^{(k')} - w_1^{(k)} \| \le \eta g_{max}(w_{init}) \sum_{i=1}^C \| p^{(k')}(y=i) - p^{(k)}(y=i) \| $$
This inequality suggests that the distance between model weights reflects the distance between data distributions, justifying the use of model weights as the state for the RL agent.

### 2.2. MDP Formulation
The problem is modeled as a Markov Decision Process (MDP):

*   **State ($s_t$)**: A concatenation of the global model weights and the local model weights of **all** $N$ devices.
    $$ s_t = (w_t, w_t^{(1)}, \dots, w_t^{(N)}) $$
    *   *Dimensionality Reduction*: Since deep networks have millions of parameters, **PCA (Principal Component Analysis)** is applied to compress the weight vectors into a low-dimensional space (e.g., 100 dimensions) before feeding them into the RL agent.

*   **Action ($a_t$)**: Selecting a subset of devices.
    *   *Simplification*: The agent is trained to select **one** device (action space size $N$).
    *   *Execution*: During deployment, the agent computes Q-values for all $N$ devices and selects the **Top-K** devices with the highest $Q(s_t, a)$.

*   **Reward ($r_t$)**: Designed to encourage high accuracy and penalize more rounds.
    $$ r_t = \Omega(\omega_t - \Phi) - 1 $$
    *   $\omega_t$: Test accuracy on the validation set after round $t$.
    *   $\Phi$: Target accuracy.
    *   $\Omega$: A constant factor (e.g., 64) to amplify accuracy gains.
    *   $-1$: A constant penalty for each communication round used.

### 2.3. Algorithm: Double DQN (DDQN)
To stabilize training, Double DQN is used. The loss function minimizes the error between the Q-network and the target:
$$ \mathcal{L}_t(\theta_t) = (Y_t^{DoubleQ} - Q(s_t, a; \theta_t))^2 $$
$$ Y_t^{DoubleQ} = r_t + \gamma Q(s_{t+1}, \arg\max_a Q(s_{t+1}, a; \theta_t); \theta_t') $$
where $\theta_t$ are online weights and $\theta_t'$ are target network weights.

## 3. Theoretical Proofs
The paper provides a bound on weight divergence (mentioned in 2.1) derived from the Lipschitz continuity of the gradient and the update rule of SGD.
Let $g_{max}(w) := \max_i \| \nabla_w \mathbb{E}_{x|y=i}[\log f_i(x, w)] \|$.
The bound proves that if data distributions $p^{(k)}$ and $p^{(k')}$ are different, the resulting weights after one step of SGD will diverge, proportional to the distribution distance.

## 4. Experimental Setup
*   **Datasets**: MNIST, FashionMNIST, CIFAR-10.
*   **Non-IID Synthesis**:
    *   $\sigma$: Degree of Non-IID.
    *   Example ($\sigma=0.8$): 80% of data on a device belongs to a single dominant class, 20% is random.
*   **Baselines**:
    1.  **FedAvg**: Random client selection.
    2.  **K-Center**: Clustering-based selection (clusters clients based on weight similarity and selects representative centers).
*   **Metrics**: Number of communication rounds to reach a fixed target accuracy (e.g., 99% for MNIST, 55% for CIFAR-10).

## 5. Implementation Details
*   **Hardware**: AWS EC2 p2.2xlarge (K80 GPU).
*   **PCA**: Reduces weights from ~431k (CNN) to 100 dimensions. PCA basis is computed on initial weights and reused.
*   **RL Network**: Two-layer MLP (512 hidden units).
*   **Hyperparameters**:
    *   Discount factor $\gamma \in (0, 1]$.
    *   Target accuracy $\Phi$: 99% (MNIST), 85% (FashionMNIST), 55% (CIFAR-10).
    *   Number of clients $N=100$, Selected per round $K=10$.

## 6. Critical Analysis (The "Gap")
*   **Scalability**: The state space requires maintaining weights for **all** $N$ devices. For $N=100$ this is fine, but for $N=1,000,000$ (real-world scale), the input dimension to the RL agent is unmanageable ($N \times \text{PCA\_dim}$).
*   **Staleness**: The state $s_t$ relies on $w_t^{(k)}$. If device $k$ hasn't been selected for many rounds, $w_t^{(k)}$ is extremely stale and may not reflect the device's current utility.
*   **Server Overhead**: Requires running PCA and forward passing the RL network for all $N$ clients every round.
