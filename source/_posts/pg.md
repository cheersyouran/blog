---
title: 策略梯度
date: 2018-06-17 10:37:22
tags: [策略梯度,强化学习,机器学习]
mathjax: true
---

*本文涉及算法的实现请参照我的[Github](https://github.com/cheersyouran/reinforcement-learning)*
## 1. 背景

按照**有无模型**分类，强化学习可以分为Model-Based、Model-Free两类，其中Model-Free又可以分为Prediction、Control两类。
按照**优化目标**来分类，强化学习可以分为Value-Based、Policy-Based、Actor-Critic三类。

本文主要说一下Policy-Based方法中的**Stochastic Policy**、**Deterministic policy**以及**Actor-Critic**方法。

## 2. Policy-Based方法

Value-Based方法是通过DP、TD或MC来更新$Q(s,a)$值，并基于$Q(s,a)$值采用 $greedy$或$\epsilon-greedy$策略选择下一个动作。所以我们的策略是根据$Q(s,a)$值**间接**得到的。

Policy-Based方法不需要学习$Q(s,a)$值，而是根据DP、TD或MC的回报值$R_t$去直接更新策略。所以我们策略是根据输入的状态s**直接**得到的。

Policy Based方法具体分为两种：Stochastic policy 和 Deterministic policy。区别是前者输出的是动作的概率，后者输出的是一个确定的动作。Policy-Based和Value-Based的优缺点如下：

| 方法 | 适用场景 | 优点 | 缺点 |
| --- | --- | --- | --- |
| Value-Based | 离散有限的动作空间 | 可以每步更新，算法效率较高；可以收敛到全局最优 | 无法处理高维连续动作空间 |
| Policy-Based | 高维连续的动作空间 | 更好的收敛性；随机策略自带探索性质 | 每个episode更新一次，效率较低，方差较大；容易收敛到局部最优； |

### 2.1 随机策略(Stochastic Policy)

随机策略常用如下公式表示：$$\pi_\theta(s,a) = P[a|s,\theta]$$其意义是：在状态s时，动作a符合参数为$\theta$的概率分布。我们的目标是找到最佳策略$\pi_\theta(s,a)$对应的参数$\theta$，因此我们必须有效衡量一个策略的好坏。我们把这个衡量方式称为Performance Objective $J(\theta)$，在神经网络中又称损失函数（目标函数）。常用的$J(\theta)$有如下三种：

| 名称 | 公式 |
| --- | --- |
| start value |$J_1(\theta)=V^{\pi_\theta}(s_1)=E_{\pi_\theta}[V_1]$|
| average value|$J_{avV}(\theta)=\sum_sd^{\pi_\theta}(s)V^{\pi_\theta}(s)$|
|average reward per time-step|$J_{avR}(\theta)=\sum_sd^{\pi_\theta}(s)\sum_a\pi_\theta(s,a)R_s^a$|

有了$J(\theta)$，我们就可以设计一个神经网络来最大化$J(\theta)$（或最小化$-J(\theta)$）。总结来说，在这个神经网络中，输入是当前agent获得的状态s，输出是策略$\pi_\theta(s,a)$，目标是最大化损失函数$J(\theta)$，参数$\theta$的更新方式是梯度上升。

那随机策略$\pi_\theta(s,a)$究竟是什么呢，又如何表示呢？在介绍策略之前，先介绍一下策略的梯度：
$$
\begin{equation}\begin{split}
\nabla_\theta\pi_\theta(s,a)&=\pi_\theta(s,a)\frac{\nabla_\theta\pi_\theta(s,a)}{\pi_\theta(s,a)}\\\\
&=\pi_\theta(s,a)\nabla_\theta log\pi_\theta(s,a)
\end{split}\end{equation}
$$推导很简单，其中$\nabla_\theta log\pi_\theta(s,a)$又被称为score function。我们采用$J_{avR}(\theta)$，则$\nabla_ \theta J(\theta)$为:$$\begin{equation}\begin{split}
\nabla_ \theta J(\theta)&=\sum_sd^{\pi_\theta}(s)\sum_a\pi_\theta(s,a)\nabla_\theta log\pi_\theta(s,a)R_s^a\\\\
&=E_{s \sim d^\pi, a \sim \pi_\theta}[\nabla_\theta log\pi_\theta(s,a)r]\\\\
&=E_{\pi_\theta}[\nabla_\theta log\pi_\theta(s,a)r]
\end{split}\end{equation}
$$通过公式我们可以发现，$\nabla_ \theta J(\theta)$只跟score function和reward有关。
回到策略，常用的随机策略有以下两种：

|名称|公式|score function|性质|
| --- | --- | --- | --- |
|Softmax策略|$\pi_\theta(s,a) \propto e^{\phi(s,a)^T\theta}$|$\phi(s,a)-E_{\pi_\theta}[\phi(s,·)]$|适用于离散动作空间|
|Gaussian策略|$a \sim N(\mu,\sigma^2), \mu=\phi(s)^T\theta$|$\frac{(a-\mu)\phi(s)}{\sigma^2}$|适用于连续动作空间|

Softmax策略需要将动作离散化，然后按照softmax之后的Q(s,a)值来随机选择动作。Gaussian策略最终会生成一个高斯分布函数，然后按照该分布随机选择动作。如果使用神经网络实现，Softmax策略只需要在网络的最后一层加入一个softmax函数，然后按照softmax之后的概率值选择动作；Gaussian策略一般默认$\sigma$是一个常量，然后用网络来回归$\mu=\phi(s)^T\theta$中的参数$\theta$，然后基于正态分布$N(\mu,\sigma^2)$来随机动作。

到这里，随机策略梯度的原理基本清晰了。但有几点值得注意：

1. Policy based方法中，我们只能得到策略梯度$\nabla J(\theta)$的公式，却并不存在关于一个损失函数的公式。**策略梯度定理**通过严格的证明(参见[证明](https://www.kth.se/social/files/58d506eff27654042836ace7/AllPolicyGradientLecture.pdf))，得到了与[环境状态转移(transition probability)]无关的$\nabla J(\theta)$。网上普遍说的损失函数$Loss = log\pi_\theta(s,a) r$只是根据$\nabla J(\theta)$反推出来的或根据交叉熵理论推演出来，尚未找到理论证明。在tensorflow中，可以不用tf.minimize而是直接用tf.compute_gradient和tf.apply_gradient来跳过损失函数的问题，当然，使用这种交叉熵$\nabla J(\theta)$也可以同样的结果。
2. $Loss = log\pi_\theta(s,a) r$函数中的r需要精心设计。应设计避免 r恒>0或 r恒<0，否者容易导致更新方向固定（恒增或恒减），导致过快收敛于局部最优。理想的r是$E[r]=0$，可参见[Github](https://github.com/cheersyouran/reinforcement-learning)中的例子。若无法避免，应降低学习率，防止过快收敛于局部最优。这也符合Advantages函数的思想。
3. policy based方法必须等episode结束才能获得reward，才能更新。这种每回合更新一次的算法叫Monte-Carlo Policy Gradient，又叫REINFORCE。参数的更新公式如下：$$\Delta\theta_t=\alpha\nabla_\theta log\pi_\theta(s_t, a_t)v_t$$$$v_t=R_{t+1} + \gamma R_{t+2} + ...+\gamma^{T-1}R_{t+2}$$完整的REINFORCE算法如下：<img src="/images/reinforce.png" width = "500" alt="REINFROCE algorithm" align=center />

4. 严格的来讲，上述PG算法属于on-policy PG，即目标策略和采样策略是相同的策略。off-policy PG公式如下：$$\begin{equation}\begin{split}
\nabla_\theta J_\beta(\pi_\theta) &= E_{s \sim \rho^\beta, a \sim \pi_\theta}[\nabla_\theta log \pi_\theta(s,a) Q^\pi(s,a)]\\\\
& = E_{s \sim \rho^\beta, a \sim \beta}[\frac{\pi_\theta(s,a)}{\beta_\theta (s,a)} \nabla_\theta log \pi_\theta(s,a) Q^\pi(s,a)]
\end{split}\end{equation}$$其中，$\frac{\pi_\theta(s,a)}{\beta_\theta (s,a)}$是重要性采样。

### 2.2 Actor-Critic算法
### 2.2.1 QAC

在2.1所描述的REINFROCE算法中，$v_t$虽然是$Q^{\pi_\theta}(s,a)$的抽样无偏估计(unbiased)，但是由于MC抽样的随机性，具有较高的方差(high variance)。如果有可以用一个函数来估计价值函数 $Q_w(s,a) \approx Q^{\pi_\theta}(s,a)$，此时$$\nabla_ \theta J(\theta) \approx E_{\pi_\theta}[\nabla_\theta log\pi_\theta(s,a)Q_w(s,a)]$$那么不仅可以减少方差，还可以达到**每步更新**，大大提高了算法的效率。这个算法的确有，叫Actor-Critic。

Actor-Critic算法由Actor和Critic两个模块构成：

| 模块名称 | 作用 | 更新方式 | 更新公式 |
| --- | --- | --- | --- |
| Actor| 选择动作 | update $\theta$ by Policy Gradient Ascent |$\theta=\theta+\alpha\nabla_\theta log\pi_\theta(s_t, a_t)Q_w(s,a)$|
| Critic| 估计$Q^{\pi_\theta}(s,a)$ | update $w$ by TD(0) | $w = w + \beta(r + \gamma Q_w(s',a') - Q_w(s,a))\nabla_\theta Q_w(s,a)$

完整的Actor-Critic算法如下：
<img src="/images/ac.png" width = "500" alt="ac algorithm" align=center />

注意：QAC算法中，仅使用一个线性价值函数$Q_w(s,a)=\phi(s,a)^Tw$来逼近状态行为价值函数$Q^{\pi_\theta}(s,a)$，而没用非线性的神经网络。

### 2.2.2 Compatible Function Approximation

在2.2.1中，我们得到了$Q^{\pi_\theta}(s,a)$的近似表示$Q_w(s,a)=\phi(s,a)^Tw$，但很遗憾，$Q_w(s,a)$始终是有偏估计，一个有偏的Q值下得到的策略梯度不一是最好的。比如近似值函数$Q_w(s,a)$可能会引起状态重名等。
当$Q_w(s,a)$满足如下两个条件时:
1. $ \nabla_w Q_w(s,a)=\nabla_\theta log\pi_\theta(s,a)$
2. 参数使误差$\varepsilon$最小化：$ \varepsilon = E_{\pi_\theta}[(Q^{\pi_\theta}(s,a) - Q_w(s,a))^2]$

那么我们就可以得到：$$Q_w(s,a) = Q^{\pi_\theta}(s,a)$$$$\nabla_ \theta J(\theta) = E_{\pi_\theta}[\nabla_\theta log\pi_\theta(s,a)Q_w(s,a)]$$过程很容易证明，只需要令(2)式$\nabla_w \varepsilon$=0即可。略。

### 2.2.3 Advantages Actor-Critic

在2.2.2中，我们得到了$Q^{\pi_\theta}(s,a) = Q_w(s,a)$的条件。再此基础上，我们还可以继续改进AC算法。考虑到不同的$Q^{\pi\theta}(s,a)$差距非常大，如上一次100下一次-100等，这样会造成很大的方差。
引入Baseline $B(s)$可以一定程度上消除方差。$B(s)$是一个仅与状态值函数$V(s)$有关，跟动作值函数$Q(s,a)$无关的函数。并有如下性质：$$\begin{equation}\begin{split}
E_{\pi_\theta}[\nabla_\theta log_{\pi\theta}(s,a)B(s)]&=\sum_s d^{\pi_\theta}(s) \sum_a \nabla_\theta \pi_\theta(s,a)B(s)\\\\
&=\sum_s d^{\pi_\theta}(s)B(s) \nabla_\theta \sum_a \pi_\theta(s,a)\\\\
&=0
\end{split}\end{equation}$$故可知：$$E_{\pi_\theta}[\nabla_\theta log_{\pi\theta}(s,a)Q^{\pi_\theta}(s,a)]=
E_{\pi_\theta}[\nabla_\theta log_{\pi\theta}(s,a)(Q^{\pi_\theta}(s,a) \pm B(s))]$$

B(s)可以有很多种选择方式，比较常见的是选择$B(s) = V^{\pi_\theta}(s)$，此时可以定义adavantages function $A^{\pi_\theta}(s,a)$为：$$A^{\pi_\theta}(s,a)=Q^{\pi_\theta}(s,a) - V^{\pi_\theta}(s)$$同时$\nabla_ \theta J(\theta)$重新表示为：$$\nabla_ \theta J(\theta) = E_{\pi_\theta}[\nabla_\theta log\pi_\theta(s,a)A^{\pi_\theta}(s,a)]
$$势函数$A^{\pi_\theta}(s,a)$的含义：在s状态采取行为a时，结果相对于状态s的平均值的改善程度。由于减去了状态s的平均估值，故而有效的减小了$Q^{\pi\theta}(s,a)$的方差，使得训练过程更平稳。

现在我们有了2个价值模型$V^{\pi_\theta}(s)$和$Q^{\pi_\theta}(s,a)$，故需要2个函数$V_v(s)和Q_w(s,a)$去逼近。
$$
\begin{equation}\begin{split}
V_v(s)&\approx V^{\pi_\theta}(s)\\\\
Q_w(s,a) &\approx Q^{\pi_\theta}(s,a)\\\\
A(s,a) &= Q_w(s,a) - V_v(s)\\\\
\end{split}\end{equation}
$$

现在，我们用两个函数已经可以表示$A^{\pi_\theta}(s,a)$，但这样做显然很复杂。有一种很好地解决方法：
$$\begin{equation}\begin{split}
\delta^{\pi_\theta}&=r+\gamma V^{\pi_\theta}(s') - V^{\pi_\theta}(s)\\\\
E_{\pi_\theta}[\delta^{\pi_\theta}|s,a] &= E_{\pi_\theta}[r+\gamma V^{\pi_\theta}(s')|(s,a)] - V^{\pi_\theta}(s)\\\\
&=Q^{\pi_\theta}(s,a) - V^{\pi_\theta}(s)\\\\
&=A^{\pi_\theta}(s,a)\\\\
所以，\nabla_ \theta J(\theta) &= E_{\pi_\theta}[\nabla_\theta log\pi_\theta(s,a)\delta^{\pi_\theta}]
\end{split}\end{equation}
$$实际中，我们用TD的函数逼近形式：$$\delta_v=r+\gamma V_v(s') - V_v(s)$$这样就只需要更新一个函数的参数v。
现在，我们重新整理一下Actor-Critic方法：

|模型|MC|TD(0)|TD($\lambda$)|
|---|---|---|---|
|Actor|$\Delta \theta = \alpha(v_t - V_v(s_t)) · $ <br>$ \nabla_\theta log \pi_\theta(s_t,a_t)$|$\Delta \theta = \alpha(r + \gamma V_v(s_{t+1}) - V_v(s_t)) · $ <br>$ \nabla_\theta log \pi_\theta(s_t,a_t)$|$\Delta \theta = \alpha(v^\lambda_t - V_v(s_t)) · $ <br>$ \nabla_\theta log \pi_\theta(s_t,a_t)$|
|Critic|$\Delta \theta = \alpha(v_t - V_\theta(s))\phi(s)$|$\Delta \theta = \alpha(r + \gamma V(s') - V_\theta(s))\phi(s)$|$\Delta \theta = \alpha(v^\lambda_t - V_\theta(s))\phi(s)$|

随机策略梯度的多种形式的总结：<img src="/images/pg-summary.png" width = "500" alt="REINFROCE algorithm" align=center />

### 2.3 确定性策略(Deterministic policy)
之前说的Stochastic Policy、Actor-Critic都是基于随机策略的。确定性策略跟随机策略不同，确定性策略在同一个状态s的动作a是唯一确定的。确定性策略的公式为：$$a = \mu_\theta(s)$$简单比较一下确定性策略和随机策略：

|策略|优点|缺点|
|---|---|---|
|确定性策略|$\nabla_\theta J(\mu_\theta)$只对状态积分，故需要采样的数据少，效率高|容易局部最优|
|随机策略|将探索和改进集成到一个策略中|$\nabla_\theta J(\pi_\theta)$对状态和动作一起积分，算法效率低，需要大量训练数据|
确定性策略的Performance Objective：$$\begin{equation}\begin{split}
J(\mu_\theta) &= \int_S \rho^u(s)r(s,u_\theta(s))ds\\\\
&=E_{s \sim \rho^\mu}[r(s,u_\theta(s))]\\\\
&=E_{s \sim \rho^\mu, a = u_\theta(s)}[r(s,a)]\\\\
&=E_{\mu_\theta}[r(s,a)]\\\\
\end{split}\end{equation}$$确定性策略梯度：
$$\begin{equation}\begin{split}
\nabla_\theta J(\mu_\theta) =\int_S \rho^u(s)\nabla_\theta\mu_\theta(s)\nabla_a Q^{\mu}(s,a)|_{a=\mu\theta(s)}ds\\\\
\end{split}\end{equation}
$$

$$=E_{s \sim \rho^\mu, a = u_\theta(s)}[\nabla_\theta\mu_\theta(s)\nabla_a Q^{\mu}(s,a)]$$

实际上，Silver在论文中已证明，确定性策略是随机策略在方差为0时的一种特殊形式。

### 2.4 Determistic Actor-Critic

在2.3节中，我们将随机行为策略（Actor）与价值策略（Critic）结合，引入了Stochastic Actor-Critic算法。同样的，我们也可以将确定性行为策略（Actor）与价值策略（Critic）结合构造一个Determistic Actor-Critic算法，又叫Determistic Policy Gradient算法。

与QAC算法相似的，在DPG中引入一个近似函数$Q^w(s,a)$来估计$Q^{\mu_\theta}$，$Q^w(s,a) \approx Q^{\mu_\theta}(s,a)$。此时我们得到$$\nabla_\theta J(\mu_\theta) \approx E_{s \sim \rho^\mu, a = u_\theta(s)}[\nabla_\theta\mu_\theta(s)\nabla_a Q^w(s,a)]$$

DPG算法的更新方法如下：
$$\begin{equation}\begin{split}
\delta_t &= r_t+ \gamma Q^w(s_{t+1}, \mu_theta(s_{t+1})) - Q^w(s_t, a_t)\\\\
w_{t+1} &= w_t + \alpha_w \delta_t \nabla_w Q^w(s_t, a_t)\\\\
\theta_{t+1} &= \theta_t + \alpha_\theta \nabla_\theta \mu_\theta(s_t) \nabla_\alpha Q^w(s_t, a_t)
\end{split}\end{equation}
$$

值得注意的是，DPG又细分为on-policy和off-policy。上述DPG算法是on-policy的，即想要学习的策略和用于采样的策略相同，critic用sarsa算法训练。off-policy中想要学习的策略和用于采样的策略不同，critic用q-learning训练。不同的采样策略会导致状态s服从不同的分布，所以策略梯度公式会稍稍有变化，参数的更新方式不变：$$\nabla_\theta J(\mu_\theta) \approx E_{s \sim \rho^\beta, a = u_\theta(s)}[\nabla_\theta\mu_\theta(s)\nabla_a Q^w(s,a)]$$

此外，当满足一下CFA条件时，$Q^w(s,a) = Q^{\mu_\theta}(s,a)$:
1. $ \nabla_\theta \mu_\theta(s)^Tw = \nabla_a Q^w(s,a)|_{a=\mu\theta(s)}$
2. 参数使误差$\varepsilon$最小化：$ \varepsilon = E_{\mu_\theta}[(Q^w(s,a)- Q^{\mu_\theta}(s,a))^2]$

满足上述条件的$Q^w(s,a)$的线性拟合有很多，比如$Q^w(s,a) = A^w(s,a) + V^v(s)$, 其中$A^w(s,a) = \phi(s,a)^Tw = [\nabla_\theta \mu_\theta(s)(a - \mu_\theta(s))]^Tw$（这里我也不知道为什么要设计成$A^w(s,a) + V^v(s)$）。

### 2.5 Deep Determistic Policy Gradient

DDPG是将深度学习神经网络融合进off-policy DPG的策略学习方法。DDPG针对DPG做了如下改进：
1. DPG使用线性函数通过线性回归拟合$Q^{\mu_\theta}$，而DDPG使用神经网络拟合$Q^{\mu_\theta}$。
2. 借鉴DQN的成功经验，采用经验回放（experience replay）的方式做离线训练，打破数据的时序关联性，保证算法的收敛。
3. 借鉴DQN的成功经验，引入了online和target两种网络。并采用soft的方式从online critic和online actor对target critic和target actor两个网络进行更新，使学习过程更加稳定。
4. 采用Batch Normalization解决feature量纲不同的问题

DDPG算法的伪代码：<img src="/images/ddpg.png" width = "550" alt="ddpg algorithm" align=center />

*本文涉及算法的实现请参照我的[Github](https://github.com/cheersyouran/reinforcement-learning)*

## 3. 参考

[1] David Silver. et al. 2014. Deterministic policy gradient algorithms. ICML'14, Vol. 32. JMLR.org I-387-I-395.
[2] Richard S. Sutton. et al. 1999. Policy gradient methods for reinforcement learning with function approximation. NIPS'99, MIT Press, Cambridge, MA, USA, 1057-1063.
[3] Lillicrap, Timothy P. et al. “Continuous control with deep reinforcement learning.” CoRR abs/1509.02971 (2015): n. pag.





