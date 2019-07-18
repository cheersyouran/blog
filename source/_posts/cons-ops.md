---
title: 约束优化
date: 2018-08-16 12:05:15
tags: 凸优化
mathjax: true
description: 文章主要讲解了几种常见的约束优化问题求解方法。
---
## 0. 前言

这几天碰到一个业务，问题的目标不仅仅是提高CVR，而是既保证CVR又要保证广告分发的结构。一开始以为是一个约束优化问题，但仔细思考了一下，发现并不是，因为这里的约束不是已知变量x而是y，无奈只能采用别的方法。
回归本文主题，我也算是借这个机会回顾了一下优化论的东西，所以在这里做一个简单的总结和整理。

## 1. 优化问题分类

首先，优化问题分为以下三种：
1. 无约束光滑优化：主要采用梯度下降的方法进行求解
2. 无约束非光滑优化：主要采用次梯度下降的方法进行求解
3. 有约束优化：主要采用投影梯度下降，ADMM，Uzawa等算法

本文主要针对有约束优化做一下讨论。

## 2. SVM中的约束优化问题

首先简单推导一下SVM。假设两个支持向量点$x_1$ 和 $x_2$分别在如下两条边界线上：
\begin{equation}
\begin{cases}
&lt;w ,x_1&gt; +\;b = 1, &(2.1)\\\\
&lt;w ,x_1&gt; +\;b = -1, &(2.2)
\end{cases}
\end{equation}根据$x_1$和$x_2$的连线与法向量平行，可得公式(2.3)；公式(2.1)(2.2)相减可得(2.4)：
\begin{equation}
\begin{cases}
x_1 - x_2 = C · w， &\quad (2.3)\\\\
&lt;w ,x_1 - x_2&gt;\;= 2, &\quad (2.4)
\end{cases}
\end{equation}结合(3)，(4)两项我们得到：$$C=\frac{2}{\parallel w\,\parallel^2_2}$$$margin = \parallel x_1 - x_2\,\parallel$，我们的目标是margin最大化，即$max \parallel x_1 - x_2\parallel_2 = |C|·\parallel  x\parallel_2 = \frac{2}{\parallel w\parallel_2}$，
将最大化问题转化为于最小化，我们就得到了目标的不等式约束优化问题：
\begin{equation}
\begin{cases}
\min　\frac{1}{2}\parallel w\,\parallel^2\\\\
s.t.　y_i(&lt;w,x_i&gt; +\;b)\geq 1, \quad i=1,2,...,m; 
\end{cases}
\quad (2.5)
\end{equation}这里，我们采用拉格朗日乘子法求解不等式约束优化问题。
首先构造拉格朗日函数：$$L(w,b,\lambda)=\frac{1}{2}\parallel w \parallel^2 + \sum^m_{i=1}\lambda_i[1−y_i(w^Tx_i+b)]\quad (2.6)$$则我们的原问题表示为如下(推导过程略，有空再补上)：$$\min \limits_{w,b} \max \limits_{\lambda} L(w,b,\lambda)$$这个公式又称为*Primal Problem*。
对应的，还有一个公式称为*Dual Problem*，公式如下：$$\max \limits_{\lambda} \min \limits_{w,b} L(w,b,\lambda)$$当满足slater condition时，强对偶性成立，也就是说此时Primal和Dual是等价的。此时，我们就可以通过求解Dual Problem问题来求解Primal Problem。当我们得到Dual Problem的解$\lambda^\*$，通过KKT条件可得到Primal Problem的解$w^\*, b^\*$。

首先求解对偶问题中的$\min \limits_{w,b} L(w,b,\alpha)$，分别令$L(w,b,\alpha)$对$w, b$的导数为0：
\begin{equation}
\begin{cases}
\frac{\partial L(w,b,\lambda)}{\partial w} = w - \sum_{i=1}^{m}\lambda_i y_i x_i = 0, &(2.7)\\\\
\frac{\partial L(w,b,\lambda)}{\partial b} = \sum_{i=1}^{m}\lambda_i y_i= 0, &(2.8)\\\\
\end{cases}
\end{equation}将公式(2.7)，(2.8)带入$\min \limits_{w,b} L(w,b,\lambda)$中化简得到：
$$\min \limits_{w,b} L(w,b,\lambda) = - \sum_{i=1}^{m} \lambda_i + \frac{1}{2}\sum_{i=1}^{m} \sum_{j=1}^{m} \lambda_i \lambda_j y_i y_j x_i^T x_j $$最终我们将对偶问题$\max \limits_{\lambda} \min \limits_{w,b} L(w,b,\lambda)$转化为如下公式：
\begin{equation}
\begin{cases}
\max \limits_{\lambda} \sum_{i=1}^{m} \lambda_i - \frac{1}{2}\sum_{i=1}^{m} \sum_{j=1}^{m} \lambda_i \lambda_j y_i y_j x_i^T x_j \\\\
s.t.\quad \sum_{i=1}^{m}\lambda_i y_i= 0\\\\
\quad\quad\;\;\lambda_i \geq 0
\end{cases}
\quad (2.9)
\end{equation}

现在，我们已经将svm约束问题(2.5)转化成为了新的约束问题(2.9)，转化后的问题只包含一类变量$\lambda$，我们可以用任意一种二次优化方法求解，其中一种比较高效的算法叫SMO，当然，也可以用PGD方法来求解，下一节会讲到PDG方法。(关于SMO算法，我会在单独写一篇博文来详细介绍)。求出了对偶问题的最优解$\lambda^\*$，根据KKT条件，我们就可以得到原问题的解了。

## 3. 投影梯度下降法（Projected Gradient Descend）

定义如下Indicator函数：
\begin{equation}
I_s(x) = 
\begin{cases}
0 &if\quad x \in S\\\\
+\infty &if\quad x \notin S
\end{cases}
\end{equation}则约束优化$$\min \limits_{x \in \mathbb R^n} f(x) \quad s.t. \; x \in S $$ 可以改写为如下形式：$$\min \limits_{x \in \mathbb R^n} f(x) + I_s(x), \quad (3.1)$$从公式(3.1)我们可以看出，第一项$f(x)$是一个光滑凸函数，第二项$I_s(x)$是一个非光滑凸函数。这样，我们就把约束问题转换成了无约束问题，可以方便的应用forward-backward splitting(FBS)算法求解。
FBS算法如下（FBS本质来自于次梯度下降，会另起一篇文章介绍）：$$ x^{(k+1)} = prox_{\alpha g}(x^{(k)} - \alpha_k \nabla f(x^{(k)})),$$$$其中，prox_{\alpha g}(y) = \arg\min \limits_{x \in \mathbb R^n} \frac{1}{2} ||x - y||^2 + \alpha g(x) $$当$g(x) = I_s(x)$时，此时的FBS算法又叫投影梯度下降算法：
\begin{equation}
\begin{split}
P_s(y) &= prox_{\alpha I}(x^{(k)})(y)\\\\
&= \arg\min \limits_{x \in \mathbb R^n} \frac{1}{2} ||x - y||^2 + \alpha I(x) \\\\
&= \arg\min \limits_{x \in S} \frac{1}{2} ||x - y||^2
\end{split}
\end{equation}则最终的投影梯度更新方程如下：
$$x^{(k+1)} = P_{s}(x^{(k)} - \alpha_k \nabla f(x^{(k)})$$**以刚才的SVM对偶问题为例**，我们可以用PGD来求解。
定义矩阵$K \in \mathbb R^{m \times n}$，其中$K_{i,j} = y_i y_j x_i^T x_j, $，则对偶问题可以改写为：
\begin{equation}
\begin{cases}
\max \limits_{\lambda} -\frac{1}{2} \lambda^T K \lambda + \lambda^T 1\\\\
s.t. \quad \lambda \geq 0\\\\
\quad \quad \;\; \lambda^Tb = 0
\end{cases}
\end{equation}我们将投影域表示为$ S \in \lbrace \lambda | \lambda \geq 0, \lambda^Tb = 0\rbrace $
对于$ \forall \mu \in \mathbb R^m $，$$ P_s(\mu) = \arg\min \limits_{\lambda \in S} \frac{1}{2} ||\lambda - \mu||^2 $$对于这个约束优化函数，我们可以构造拉格朗日函数求解：$$ L(\lambda, \alpha, \beta) = \frac{1}{2}||\lambda - \mu||^2_2 + \lambda^T\alpha + \beta \lambda^T b $$对偶函数：$$ \max \limits_{\alpha \leq 0, \beta} \min \limits_{\lambda} \frac{1}{2}||\lambda - \mu||^2_2 + \lambda^T\alpha + \beta \lambda^T b （3.2）$$ 对 $ \min \limits_{\lambda} $求导，得到$ \lambda = (\mu - \beta b) - \alpha $带入（3.2）中，得到
\begin{equation}
\begin{split}
&\max \limits_{\alpha \leq 0, \beta} -\frac{1}{2} ||\alpha + \beta b - \mu ||^2_2\\\\
= &\min \limits_{\beta} \min \limits_{\alpha \leq 0}\frac{1}{2} ||\alpha + \beta b - \mu ||^2_2（3.3）
\end{split}
\end{equation}对于（3.3）中的$\min \limits_{\alpha \leq 0}$项，我们容易求得$\alpha = (\mu - \beta b)_{-}$，最终公式（3.2）转换为一个单变量优化问题：

\begin{equation}
\begin{split}
\min \limits_{\beta} \frac{1}{2}||(\mu - \beta b)_{+}||^2_2
\end{split}
\end{equation}求出对偶问题的最优解$\alpha^\*$和$\beta^\*$后，再根据KKT条件，即可求出原问题的最优解$\lambda^\*$。

最终，SVM对偶问题的PDG求解公式如下：

\begin{equation}
\begin{split}
\lambda^{(k+1)} = P_s(\lambda^{(k)} - \delta_k(K \lambda^{(k)} - 1))
\end{split}
\end{equation}

\begin{equation}
\begin{split}
其中，P_s(\mu) = (\mu - \beta b)_{+}，
\end{split}
\end{equation}

$$\beta = \arg \min \limits_{\beta} \frac{1}{2} ||(\mu - \beta b )_{+}||^2_2$$

## 4. Uzawa算法
PGD是一个非常重要的算法。由上面的介绍我们知道，PGD面向原问题$f(x)$，先按梯度方向更新$x$，然后把更新后的$x$在其自己的约束域上投影。而有些时候，对偶问题$d(\lambda, \mu)$是个光滑函数，$\lambda, \mu$的约束也十分简单，更方便求解投影。这个算法就叫Uzawa算法。 

\begin{equation}
\begin{cases}
\min \limits_{x \in \mathbb R^n}& f(x)\\\\
s.t. &g_i(x) \leq 0, i = 1,2,...,p\\\\
&h_i(x) = 0, i = 1,2,...,q
\end{cases}
\end{equation}

算法形式如下：
\begin{equation}
\begin{cases}
x^{(k+1)} = \arg \min \limits_{x \in \mathbb R^n} L(x, \lambda^{(k)}, \mu^{(k)})\\\\
\lambda^{(k+1)}_i = (\lambda^{(k)}_i + \alpha_k g_i(x^{(k+1)}))_{+}, for i=1,2,...,p\\\\
\mu^{(k+1)}_i = \mu^{(k)}_i + \alpha_k h_i(x^{(k+1)}), for i=1,2,...,q\\\\
\end{cases}
\end{equation}
## 5. ADMM算法

Uzawa和PGD算法中，约束的形式通常是不等式约束，他们对于下述问题可能不方便求解：
\begin{equation}
\begin{cases}
\min \limits_{x, y} f(x) + g(y)\\\\
s.t. Ax - y = 0
\end{cases}
\end{equation}
或者有时候，我们可以把无约束问题转化为有约束问题，更方便求解，如：
\begin{equation}
\min \limits_{x} f(x) + g(Ax) =>
\begin{cases}
\min \limits_{x, y} f(x) + g(y)\\\\
s.t. Ax - y = 0
\end{cases}
\end{equation}
以上问题可以尝试使用ADMM算法求解。求解过程如下：
1. 构造增广拉格朗日函数（Augmented Lagrangian Function）$$L_{\gamma}(x, y, \mu) = f(x) + g(y) + \mu^T(Ax - y) + \frac{\gamma}{2} ||Ax - y||^2_2$$
2. 迭代计算：
\begin{equation}
\begin{cases}
x^{(k+1) = \arg\min\limits_{x} L_{\gamma}(,x, y^{(k)}), \mu^{(k)}}\\\\
y^{(k+1) = \arg\min\limits_{y} L_{\gamma}(,x^{(k+1)}, y, \mu^{(k)}}\\\\
\mu^{(k+1)} = \mu^{(k)} + \alpha_k(Ax^{(k+1)} - y^{(k+1)})
\end{cases}
\end{equation}

