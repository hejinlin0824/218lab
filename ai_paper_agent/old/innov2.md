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