# 创新点1：

# 基于互信息的鲁棒状态构建 (MI-State Construction)

## 问题背景：

favor等方法是基于Q值，然后状态是本轮Q值最大的几个客户端的ACC和全局ACC作为状态，这个用ACC做状态有点经验化了，你说不清楚为啥用ACC做状态，也就是说他们强行绑定ACC为状态，经验化了；

## 我们的方法：

我们定义这一创新点的目标是：从众多的候选统计量中，筛选出 $$K$$ 个包含最大信息量的指标，构成RL的观察状态 $$s\_t$$。

**1. 候选特征池构建：**

我们需要涵盖 **客户端本地属性** 和 **客户端-服务端交互属性**。 假设在 Warm-up 阶段（第 $$t$$轮），对于客户端 i，上传了模型更新(梯度) $$Δθ\_i^t$$ 。此时服务端维护全局模型 $$Δθ^t$$。

我们可以提取一个包含 $$$$ 个统计量的候选集合 $$X\_i^t = (x\_{i,1}^t,x\_{i,2}^t,...,x\_{i,M}^t)$$,建议的候选统计量（一定要用数学符号定义，显得专业）：

**梯度范数 (Gradient Norm)** - 表征更新幅度；

**梯度方差 (Gradient Variance)** - 表征参数更新的不确定性;

**余弦相似度 (Cosine Similarity)** - **关键交互指标**，表征客户端更新方向与上一轮全局更新方向 $$\Delta \theta\_{global}^{t-1}$$的一致性（衡量Non-IID程度）;

**标量损失/精度 (Local Loss/Acc)** - 如果协议允许上传;

**梯度的偏度/峰度 (Skewness/Kurtosis)** - 表征分布形态;

**2.目标变量定义 (Target Variable)**

我们需要衡量这些特征到底对谁有贡献。目标变量 Y 应当是**本轮全局模型的性能提升：**

$$Y^t=Acc\_{global}^{t+1}-Acc\_{global}^t$$

或者用验证集Loss的下降量。

**3.互信息计算 (Mutual Information Calculation)**

在 Warm-up 阶段（例如前 $$E\_{warm$$ 轮），我们采用随机采样策略（Random Policy）收集数据。我们构建一个数据集 $$\mathcal{D} = \{(x\_{j}, Y) \}\_{samples}$$，其中 $$x\_$$ 是第 $$$$ 个候选特征的观测值序列。

我们要计算每个候选特征 $$x\_$$ 与目标 $$$$ 之间的互信息 $$I(x\_j; Y)$$。

由于 $$x\_$$ 和 $$$$ 都是连续变量，我们在论文中给出通用的微分熵（Differential Entropy） 定义公式：

$$I(x\_j; Y) = H(Y) - H(Y | x\_j)$$

$$I(x\_j; Y) = \iint p(x\_j, y) \log \frac{p(x\_j, y)}{p(x\_j)p(y)} dx\_j dy$$

**4.状态选择与生成 (State Generation)**

根据计算出的互信息值对特征进行排序：

$$\mathcal{S}\_{indices} = \text{Top-K}(\{I(x\_1; Y), I(x\_2; Y), ..., I(x\_M; Y)\})$$

正式的RL状态 $$s\_$$ 定义为：

对于当前轮次的每一个被选中的客户端 i（或者是所有候选客户端），我们将筛选出的 $$$$ 个特征拼接，并加上全局状态（如当前全局轮数 t，当前全局Acc），构成最终状态向量：

$$s\_t = [ \mathbf{f}\_{selected}^{(1)}, \mathbf{f}\_{selected}^{(2)}, ..., \mathbf{f}\_{selected}^{(N)}, \text{Global\\_Feat} ]$$

其中 $$\mathbf{f}\_{selected}^{(i)$$ 包含客户端 $$$$ 的那 Top-K 个关键指标。

# 创新点2：

# 奖励函数的设计：

这一部分至关重要。如果我们只是简单地说“把指数换成Sigmoid”，Reviewer会觉得这只是在调参（，不够格上顶会。

我们需要给这个Sigmoid穿上一件“鲁棒统计学（Robust Statistics）”**和**“有界效用理论（Bounded Utility Theory）”的外衣。我们提出Saturation-Aware Robust Reward（SARR）

首先我们得批判一下指数函数对异常值太敏感了，坏人只要制造一点点巨大的虚假提升，RL就会被带跑偏。

我们要提出的不叫“Sigmoid奖励”，而叫做 **“SARR机制”**。 我们引入一个在鲁棒统计学中常见的概念：**有界影响函数（Bounded Influence Function）**。

## 公式定义：

我们将奖励函数 $$R\_$$ 定义为关于性能增量 $$\Delta \Phi\_$$ 的双曲正切变形（Sigmoid的各种变形本质都是Tanh，但Tanh形式在数学推导上更对称、更漂亮）：

$$R\_t(\Delta \Phi\_t) = \mathcal{C} \cdot \tanh \left( \frac{\sigma \cdot \Delta \Phi\_t}{\mathcal{C}} \right)$$

其中：

$$\Delta \Phi\_t = \text{Acc}\_{t+1} - \text{Acc}\_$$：本轮全局模型的性能提升量（可以是负数）。

$$\mathcal{C$$：**Saturation Capacity（饱和容量）**。这是最重要的超参数，代表了我们允许单轮最大奖励的“物理上限”。

$$\sigm$$：**Sensitivity Factor（敏感因子）**。控制在 0 附近的线性增益斜率。

## 为什么这个公式高级？（Storytelling）

不要小看这个 tanh。我们可以对它进行 泰勒级数展开（Taylor Expansion） 来解释它的双重特性：

**Near-Zero Regime (正常学习区):**

当 $$\Delta \Phi\_t \to $$ 时（大部分正常的FL训练轮次）：

$$\tanh(x) \approx x \implies R\_t \approx \sigma \cdot \Delta \Phi\_t$$

**解释：**在正常训练下，它退化为线性奖励，保证了梯度传导的无偏性，能够灵敏地捕捉细微的模型提升。

**Saturation Regime (攻击/异常区):**

当 $$\Delta \Phi\_t \to \inft$$ 时（由于后门攻击或标注噪声导致的异常Loss剧变）：

$$\tanh(x) \to 1 \implies R\_t \to \mathcal{C}$$

**解释：**它自动对异常值进行了 Soft-Clipping（软截断）。无论攻击者制造了多大的梯度爆炸，Agent收到的奖励永远不会超过C。这从数学原理上切断了Poisoning Attack通过Reward Signal影响策略梯度的路径。

我们可以在附录（或正文）中加入一个关于 **“Influence Function（影响函数）”** 的简短推导。

**定理（非正式）**：

> 定义 PPO 的策略梯度更新量为 $\nabla J(\theta)$。在 SARR 机制下，单个恶意客户端 k 对策略梯度的影响上限是常数有界的。

## 推导逻辑：

PPO的Loss主要依赖于 Advantage Function $A\_t$。而 $$A\_$$ 是 Reward 的累积。

$$A\_t \approx \sum \gamma^k R\_{t+k} - V(s)$$

如果 $$R\_$$ 是指数型的，则 $\sup |A\_t| = \infty$（无界）。

如果 $$R\_$$ 是 SARR 设计的，则 $\sup |A\_t| \le \frac{\mathcal{C}}{1-\gamma}$（有界）。

**一句话总结（写在Paper里）**：

> "Unlike exponential rewards which imply an infinite influence bound, SARR imposes a strictly bounded influence function on the policy update, ensuring B-robustness (Bias-robustness) against Byzantine failures."

> (这句话引用了鲁棒统计学的 B-robustness 概念，非常唬人且准确。)

## 这样我的三个创新点就是：

创新点1（MI状态）是为了**看清**谁是好人。

创新点2（SARR奖励）是为了**防御**结果造假。

创新点3（Risk-Sensitive）是为了**规避**潜在的不确定性风险。

# 创新点3：不确定性感知的风险规避 PPO (URA-PPO)

## **理论动机 (Theoretical Motivation)**

在传统的 RL-FL 中，Critic 网络 $$V\_\phi(s$$ 仅仅估计状态的期望价值（Expected Value）：

$$V(s) = \mathbb{E}\_{\pi} \left[ \sum \gamma^t r\_t \mid s\_0 = s \right]$$

这意味着，如果有两个选择：

* 选择A：收益稳定是 10。
* 选择B：收益要么是 -100（被攻击），要么是 120（超长发挥），平均也是 10。

普通 PPO 认为 A 和 B 是一样的。但在 FL 中，B 是致命的，因为一次 -100 的攻击可能摧毁全局模型。我们需要 Agent 能**感知并规避 B**。

## 核心架构：双头 Critic 网络 (Dual-Head Critic Architecture)

我们要修改 Critic 网络的结构。它不再只输出一个标量 Value，而是输出两个值：

1. **价值均值 (Mean Value)** $$\mu\_\phi(s$$：预测未来的期望收益。
2. **任意不确定性 (Aleatoric Uncertainty)** $$\sigma\_\phi^2(s)$$：预测未来收益的方差（风险）。

**Paper 里的网络定义**：

> "We redesign the Critic network as a probabilistic estimator modeled by a Gaussian distribution $\mathcal{N}(\mu\_\phi(s), \sigma\_\phi^2(s))$, where $\sigma\_\phi^2(s)$ captures the intrinsic variability of the FL environment caused by data heterogeneity and potential adversaries."

## 损失函数重构 (Loss Function Engineering)

为了训练这个双头 Critic，我们不能用简单的 MSE（均方误差）。我们需要用 **负对数似然损失 (Negative Log-Likelihood, NLL)** 来同时学习均值和不确定性。

Critic 的 Loss 函数 $$L\_{Critic$$ 定义为：

$$L\_{Critic}(\phi) = \frac{1}{N} \sum\_{t=1}^N \left( \frac{(R\_t - \mu\_\phi(s\_t))^2}{2\sigma\_\phi^2(s\_t)} + \frac{1}{2} \log \sigma\_\phi^2(s\_t) \right)$$

* **直观解释**：
* 第一项鼓励 $$\mu\_\phi$$ 逼近真实回报 $R\_t$（同MSE）。
* 但如果某个状态的 $$R\_t$$ 波动很大（很难预测），网络会倾向于调大 $\sigma\_\phi^2$（分母变大）来降低第一项 Loss。
* 第二项 $$\log \sigma\_\phi^2$$ 是惩罚项，防止网络无脑把 $\sigma$ 预测得无穷大来作弊。
* **结果**：网络自动学会了：**“哪里不稳定，哪里的** $$\sigm$$ **就大”**。

## 风险敏感的优势函数 (Risk-Sensitive Advantage)

这是最核心的创新点。我们将这种不确定性注入到 PPO 的决策过程中。

传统的 Advantage 是 $$A\_t = R\_t + \gamma V(s\_{t+1}) - V(s\_t)$$。

我们提出 Risk-Penalized Advantage (RPA)：

$$\hat{A}\_{risk}(s\_t, a\_t) = \hat{A}\_{GAE}(s\_t, a\_t) - \lambda \cdot \sqrt{\sigma\_\phi^2(s\_t)}$$

其中：

* $$\hat{A}\_{GAE$$ 是基于 $$\mu\_\phi$$ 计算的标准广义优势估计。
* $$\lambd$$ 是 **风险厌恶系数 (Risk-Aversion Coefficient)**。
* $$\sqrt{\sigma\_\phi^2(s\_t)$$ 是 Critic 预测的标准差。

**决策逻辑**： 当 Agent 处于一个高风险状态（比如选了一组历史表现极其不稳定的 Client）时，$$\sigm$$ 会很大，导致 Advantage 变小甚至变为负数。PPO 算法会认为“这是一个糟糕的动作”，从而降低未来选择该动作的概率。

## 理论升华：从 Expected Return 到 CVaR

为了让 Paper 显得更有深度，你可以提到这本质上是在优化 **条件风险价值 (Conditional Value at Risk, CVaR)** 的下界，或者是 **Mean-Variance Optimization** 的一种形式。

> "By incorporating the uncertainty penalty into the advantage estimation, our URA-PPO algorithm implicitly optimizes a risk-constrained objective, shifting the policy from maximizing expected utility to maximizing the **Risk-Adjusted Return**, effectively filtering out clients with high epistemic uncertainty."

## 全文逻辑链条最终确认 (The Final Storyline)

现在的论文逻辑像锁链一样：

1. **Introduction**: FL 很难，因为数据 Non-IID 且有攻击。现有的 RL 方法太天真（Naive），只看 Accuracy，不看风险和鲁棒性。
2. **Innovation 1 (Input - State)**: **基于互信息的特征筛选**。
3. *作用*：去伪存真。把无意义的噪音过滤掉，只把真正能反映 Client 贡献潜力的物理量（梯度分布、交互信息）喂给 RL。
4. **Innovation 3 (Algorithm - Policy)**: **URA-PPO (双头 Critic)**。
5. *作用*：避险。Agent 不仅学会了哪个 Client 准确率高，还学会了哪个 Client “情绪稳定”。对于那些忽高忽低（疑似攻击）的节点，Agent 会因为其高方差而产生惩罚，主动避开。
6. **Innovation 2 (Feedback - Reward)**: **SARR (Tanh Reward)**。
7. *作用*：兜底。万一前两步都漏了，某个坏人真的混进来了，并且产生了一个极端的虚假 Accuracy，SARR 的饱和区（Saturation）会把这个奖励“切断”，防止策略梯度爆炸，保证模型不会崩。