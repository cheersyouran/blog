---
title: 置信域算法
date: 2018-08-23 15:15:01
tags: 凸优化
mathjax: true
description: 文章主要讲解最优化算法Trust Region算法的原理。
---

## 1.什么是Trust Region？

Trust Region和Line Search是两个最基本的最优化算法。
1. Line Search：“先方向，后步长”，即先寻找正确的更新方向（比如梯度的负方向），再寻找最合适和步长$\alpha$
2. Trust Region：“先步长，后方向”，即先选取一个可信赖的区域，然后在区域内求解近似模型的最优解，在此基础上增量迭代。

## 2.Trust Region算法流程

1. 基于当前位置$x_i$，给定一个可信赖区域$\Omega = \lbrace x|\; \parallel x - x_i\parallel \leq R\rbrace$，$R$是区域半径。
2. 在$x_i$处构造原函数$f(x)$的近似函数$\hat{f}(p) \approx f(x+p)$（通常是泰勒二阶近似）。
3. 求最优解得到试探步长：$$p_i = \arg\min \limits_{p} \hat{f}(p), \; s.t. \parallel p\parallel \leq R$$
4. 计算$r_i$：$$r_i = \frac{f(x_i) - f(x_i + p_i)}{\hat{f}(0) - \hat{f}(p_i)}$$
	a. 若$r_i \geq 0.75$，说明近似效果好，则可以适当增加$R = 2R$并重新计算$p_i$；
	b. 若$0.25 < r_i < 0.75$，保持$R$不变，并认为步长$p_i$是可靠的；
	c. 若$r_i \leq 0.25$，说明近似效果很差，则减小$R = \frac{||p_i||}{4}$并重新计算$p_i$；
	
5. 循环上述步骤，直到达到任意结束条件：
	a. 达到最大迭代次数
	b. $\Delta x$小于阈值
	c. 下降梯度小于阈值

置信域是一种很重要的算法，强化学习中的TRPO以及PPO算法都是基于置信域理论，后期会单独聊一下这两个算法。前几天鹅厂竟然邀请到了Pieter Abbeel来做分享会，有幸听到了Pieter Abbeel对元学习、模仿学习、强化学习的一些见解和研究，简直激动，有机会也会整理出来。

