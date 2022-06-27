# $Fisher$ 线性判别

## 形式推导

对于得到的低维投影，我们希望**不同类别的投影能尽可能的区分开来，而同一类别的投影尽可能地靠近**。如下图所示，两类的线性判别问题可以看做是把所有样本都投影到一个方向上，然后再这个一维空间中确定一个分类的**阈值**，过这个阈值点且与投影方向垂直的超平面就是两类的分类面。

<img src="FisherFisher 线性判别.assets/A4F2DBDED3FB9C28DAB516D561ED1BF0.png" alt="img" style="zoom: 25%;" />

为了定量地描述这一思想，我们需要一些基本概念：

给定数据集$D=\{(x_i,y_i)\}_{i=1}^m,y_i\in\{0,1\}$，令$X_i,\mu_i,\varSigma_i$分别表示类示例的集合、均值向量，协方差矩阵。

也即
$$
\mu_1'=\dfrac{1}{N_1}\sum_{x\in X_1}x\\
\mu_2'=\dfrac{1}{N_2}\sum_{x\in X_2}x\\
\varSigma_1=\dfrac{1}{N_1}\sum_{x\in X_1}(x-\mu_1)(x-\mu_1)^T\\
\varSigma_2=\dfrac{1}{N_2}\sum_{x\in X_2}(x-\mu_2)(x-\mu_2)^T\\
$$
协方差矩阵一般情况下前面的系数都是$\dfrac{1}{N-1}$。

像在$PCA$中那样，给定任意的投影方向$w$，我们都可以假设$\|w\|=1$，因此$x_i$的投影值为$w^Tx_i$，所以投影值的均值为
$$
\mu_1=w^T\mu_1'\\
\mu_2=w^T\mu_2'\\
$$
自然，正类样本投影值的方差为
$$
\begin{aligned}
&\dfrac{1}{N_1}\sum_{x\in X_1}(w^Tx-w^T\mu_1)^2\\
=&\dfrac{1}{N_1}\sum_{x\in X_1}(w^Tx-w^T\mu_1)(w^Tx-w^T\mu_1)^T\\
=&w^T\left(\dfrac{1}{N_1}\sum_{x\in X_1}(x-\mu_1)(x-\mu_1)^T\right)w\\
=&w^T\varSigma_1w
\end{aligned}
$$
标准差为$\sigma_1=\sqrt{w^T\varSigma_1w}$。注意$w^T\varSigma_1w$是一个数，而不是像$\varSigma_1$是一个矩阵，因为投影到$w$方向相当于变成了一维的。

同样，负样本的方差和标准差为$w^T\varSigma_2w,\sigma_2=\sqrt{w^T\varSigma_2w}$。

所以我们想要最大化$\dfrac{类间间距}{类内间距}$可以写为如下的准则：$max\dfrac{|\mu_1-\mu_2|}{\sigma_1+\sigma_2}$或者$max\dfrac{|\mu_1-\mu_2|}{\sqrt{\sigma_1^2+\sigma_2^2}}$。

## 散度矩阵和协方差矩阵

绝对值和平方根会使优化变得十分复杂，所以我们可以等价变换为最大化$\dfrac{(\mu_1-\mu_2)^2}{\sigma_1^2+\sigma_2^2}$。这就是$Fisher$准则函数。也可以显式地表示成如下的形式
$$
max\ \dfrac{w^T(\mu_1'-\mu_2')(\mu_1'-\mu_2')^Tw}{w^T(\varSigma_1+\varSigma_2)w}
$$
散度矩阵与协方差矩阵除了是否乘以点的数量这一微小差异之外，互相等价，其定义为$S=\sum\limits_{i=1}^n(x_i-\bar{x})(x-\bar{x})^T$。经典的$FLD$使用散度矩阵而非协方差矩阵，也就是说，$FLD$的目标函数还可以写为
$$
\max\ \dfrac{w^T(\mu_1'-\mu_2')(\mu_1'-\mu_2')^Tw}{w^T(S_1+S_2)w}
$$
为方便表示，我们取如下定义：
$$
S_b=(\mu_1'-\mu_2')(\mu_1'-\mu_2')^T\\
S_w=S_1+S_2
$$
我们称$S_B$为**类间散度矩阵**，度量的是由两个类别的均值导致的散度，测量的是两个不同类之间的离散程度；$S_w$被称为**类内散度矩阵**，度量的是在原始数据集中每个类内部的离散程度。

所以我们形式化的表示$FLD$的目标函数为
$$
max\ J(w)=\dfrac{w^TS_bw}{w^TS_ww}
$$
$\dfrac{w^TS_bw}{w^TS_ww}$被称为**广义瑞利商**。直观来讲，就是找一个投影方向使得类间散度尽可能大，类内散度尽可能小。

***

## 优化

求出$J$关于$w$的导数并置$0$，我们可以得出
$$
\dfrac{\part J}{\part w}=\dfrac{2\left((w^TS_ww)S_bw-(w^TS_bw)S_ww\right)}{(w^TS_ww)^2}=0
$$
因此最优性的一个必要条件是$S_bw=\dfrac{w^TS_bw}{w^TS_ww}S_ww$。注意到$\dfrac{w^TS_bw}{w^TS_ww}$是一个标量值，这个条件实际上是说$S_bw=\lambda S_ww$，而$\lambda$恰好就是目标函数$J$。这个条件就是说$w$应该是$S_b$和$S_w$的广义特征向量，而$J$就是对应的广义特征值。所以我们只需要找到最大广义特征值对应的广义特征向量就是目标$w^*$。但是还有一种更简单的方法！

最优性的必要条件我们还可以表示成
$$
\begin{aligned}
S_ww&=\dfrac{w^TS_ww}{w^TS_bw}S_Bbw\\
&=\dfrac{w^TS_ww}{w^TS_bw}(\mu_1'-\mu_2')(\mu_1'-\mu_2')^Tw\\
&=\dfrac{w^TS_ww}{w^TS_bw}(\mu_1'-\mu_2')^Tw(\mu_1'-\mu_2')\\
&=c(\mu_1'-\mu_2')
\end{aligned}
$$
$c$是一个标量值不影响$w$的方向，所以最优的投影方向为$S_w^{-1}(\mu_1'-\mu_2')$。然后规范化即可。

***

我们还可如下求解：因为我们的目的是求得使$J(w)$最大的投影方向即可，$w$的幅值并不影响方向，所以我们可以令$\dfrac{w^TS_bw}{w^TS_ww}$的分母为一常数，故优化问题转化为
$$
\begin{aligned}
&max\quad w^TS_bw\\
&s.t.\quad w^TS_ww=c
\end{aligned}
$$
转化成拉格朗日函数的无约束极值问题：
$$
L(w,\lambda)=w^TS_bw-\lambda(w^TS_ww-c)
$$
在极值处应该满足$\dfrac{\partial L(w,\lambda)}{\partial w}=0$，也即极值解$w^*$满足$S_bw^*-\lambda S_ww^*=0$，假定$S_w$是非奇异的（样本数大于维数时通常是非奇异的），可以得到$S_w^{-1}S_bw^*=\lambda w^*$。也就是说$w^*$是矩阵$S_w^{-1}S_b$的特征向量，我们将$S_b$的原形$(\mu_1'-\mu_2')(\mu_1'-\mu_2')^T$代入可得

$S_w^{-1}(\mu_1'-\mu_2')(\mu_1'-\mu_2')^Tw^*=\lambda w^*$，可以看出$(\mu_1'-\mu_2')^Tw^*$是标量，不影响$w^*$的方向，所以得到的$w^*$的方向是由$S_w^{-1}(\mu_1'-\mu_2')$决定的，这就是$Fisher$判别准则下的最优投影方向。

***

## 二分类问题的$FLD$求解步骤

* 求出$\mu_1',\mu_2'$和$S_w$。
* 计算$w\leftarrow S_w^{-1}(\mu_1'-\mu_2')$。
* 规范化：$w\leftarrow \dfrac{w}{\|w\|}$。

