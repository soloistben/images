##### Linear Regression 线性回归

<img src="/home/hwc/git/image/statistics/Linear_Regression/LR1.png" alt="LR1" style="zoom: 50%;" />

数据定义：N个p维样本，即X维度(N, p) （N > p，样本之间独立同分布）；真实值Y维度(N,1)；直线f(w) = w^T x + b（偏置b可先忽略）
+ **矩阵表达**
  + Least Squares 最小二乘估计法（最小平方法）
  + **L(w) = Σ||w^T x\_i - y\_i||^2** = Σ(w^T x\_i - y\_i)^2 = (w^T X^T - Y^T) (Xw - Y) = w^T X^T X w - 2 w^T X^T Y + Y^T Y
	+ **w\~ = argmin L(w)**
  + 求导 dL/dw =  2 X^T X w - 2 X^T Y = 0  ----> X^T X w = X^T Y
    + **w\~ = (X^T X)^-1 X^T Y** (伪逆：(X^T X)^-1 X^T)
  + x\_3的误差为(w^T x\_3 - y\_3)，即所有误差分成一小段一小段
  
+ **几何意义**
  + f(w) = w^T x <=> f(β) = x^T β
  + 可以将数据X看成p维的空间，Y是不在该p维空间内
  + 目标：在p维空间中找到一条直线f(β)离Y最近，即Y在p维空间的投影
    + 若向量a与向量b垂直，则 a^T b = 0
  
      <img src="/home/hwc/git/image/statistics/Linear_Regression/LR2.png" alt="LR2" style="zoom: 50%;" />
  
    + 虚线为(Y - Xβ) 与X的p维空间垂直，X^T (Y-Xβ) = 0 ----> β = (X^T X)^-1 X^T Y
    + 误差分散在p个维度上
+ **概率角度**
  + 最小二乘法 <=> 噪声为高斯分布的极大似然估计法（MLE with Gaussian noise）
  + 数据本身会带有噪声 ε\~N(0, σ^2)
  + y = f(w) + ε = w^T x + ε
    + y|x,w \~ N(w^T x, σ^2)  <=> **P(y|x,w)** =1/(sqrt(2\*pi)\*σ) exp(-(y - w^T x)^2/(2\*σ^2))
  + 定义log-likelihood： 
    + L\_MLE (w) = log P(y|x,w) = log Π P(y\_i|x\_i,w) = Σ log P(y\_i|x\_i,w) = Σ[log(1/(sqrt(2\*pi)\*σ)) + log(exp(-(y\_i - w^T x\_i)^2/(2\*σ^2)))] = Σ[log(1/(sqrt(2\*pi)\*σ)) -(y\_i - w^T x\_i)^2/(2\*σ^2)]
    + 样本之间独立同分布 
    + w~ = argmax L\_MLE (w) = argmax -(y\_i - w^T x\_i)^2/(2\*σ^2) = argmin (y\_i - w^T x\_i)^2
    + 则与最小二乘法定义一样 (**LSE <=> MLE with Gaussian noise**)

+ **Regularization 正则化** 
  + 若样本没有那么多，X维度(N, p)的N没有远大于p，则求w\~中的(X^T X)往往不可逆，（p过大，有无数种结果）会引起**过拟合**
    + 最直接是加样本数据
    + 降维or特征选择or特征提取 (PCA)
    + 正则化（损失函数加个约束）：argmin [L(w)+λP(w)]
  + L1 -> Lasso
    + P(w) = ||w||\_1
  + L2 -> Ridge 岭回归
    + P(w) = ||w||\_2 = w^T w
    + 权值衰减
    + J(w) = Σ||w^T x\_i - y\_i||^2 + λ w^T w = w^T X^T X w - 2 w^T X^T Y + Y^T Y + λ w^T w = w^T(X^T X + λ I) w - 2 w^T X^T Y + Y^T Y
      + w\~ = argmin J(w)
      + dJ/dw = 2 (X^T X + λ I)w - 2 X^T Y = 0, w\~ = (X^T X + λ I)^-1 X^T Y
      + X^T X 是半正定矩阵+对角矩阵=(X^T X + λ I)正定矩阵，必然**可逆**
  + 贝叶斯的角度
    + 参数w服从分布，w\~N(0,σ\_0^2) ---> **P(w)** = 1/(sqrt(2\*pi)\*σ\_0) exp(||w||^2/(2\*σ\_0^2))
    + P(w|y) = P(y|w)P(w) / P(y)
    + MAP: w\~ = argmax P(w|y) = argmax P(y|w)P(w) = argmax log(P(y|w)P(w)) = argmax log[1/(2\*pi\*σ\_0\*σ) exp(-(y\_i - w^T x\_i)^2/(2\*σ^2) -||w||^2/(2\*σ\_0^2))] = argmin [(y\_i - w^T x\_i)^2 + σ^2/σ\_0^2||w||^2] = argmin [L(w)+λP(w)]
    + λ = σ^2/σ\_0^2
    + **Regularized LSE <=> MAP with Gaussian noise and Gaussian prior**

##### Linear Classification
+ 线性回归------>激活函数，降维------->线性分类

+ 硬分类：0/1
  + 线性判别分析 (Fisher)
  + 感知机
  
+ 软分类：[0,1]区间内概率
  + 生成式：Gaussian Discriminant Analysis（转换用贝叶斯求解）
  + 判别式：Logisitic Regression（直接学习P(Y|X)）

+ **Perceptron 感知机** (1957年)

  <img src="/home/hwc/git/image/statistics/Linear_Classification/LC1.png" alt="LC1" style="zoom: 67%;" />

  + 样本：{(x\_i, y\_i)}, N个
  + 思想：错误驱动（先初始化w，检查分错的样本，前提是线性可分）（感知错误，纠正错误）
  + 模型：f(x) = sign(w^T x + b) （w^T x大于等于0表示为1（分类正确），反之为-1（分类错误））
  + 策略：loss function（被错误分类的样本个数）
    + L(w) = Σ I{y\_i \* (w^T x\_i) < 0} （非连续函数，不可导）
    + L(w) = Σ -y\_i w^T x\_i, dL = -y\_i x\_i
  + 若是非线性可分，可是使用pocket algorithm
  
+ **线性判别分析**

  <img src="/home/hwc/git/image/statistics/Linear_Classification/LC2.png" alt="LC2" style="zoom:50%;" />
  
  + 样本：N个p维样本，二分类(+1,-1)，正样本个数N\_1，均值X\_c1，方差S\_c1，负样本个数N\_2，均值X\_c2，方差S\_c2，（S\_c1 = 1/N\_1 Σ (x\_i - X\_c1)(x_i - X\_c1)^T）
  + 思想：类内小，类间大
    + 将所有样本映射到一个Z平面（模型学习找最优平面），设定阈值，根据类的方差将样本分类
    + 类内样本距离应该更紧凑（高内聚），类间更松散（松耦合）
    + Z平面的法向量为最后找到的分类函数 w^T x（因为垂直，则Z平面即w向量）
      + （前提设置||w||=1）
      + 则样本点投影到Z平面为：|x\_i|cos(x\_i,w) = |x\_i||w|cos(x\_i,w) =x\_i w = w^T x\_i
  + 模型：分别求出两类投影在Z平面上的**均值Z\_1,Z\_2**和**方差S\_1,S\_2** 
    + N\_1 = 1/N\_1 Σ w^T x\_i
    + S\_1 = 1/N\_1 Σ (w^T x\_i - Z\_1) (w^T x\_i - Z\_1)^T
    + 类间：(Z\_1-Z\_2)^2
    + 类内：S\_1+S\_2
  + 策略：L(w) = (Z\_1-Z\_2)^2 / (S\_1+S\_2) = [w^T (X\_c1 - X\_c2)(X\_c1 - X\_c2)^T w] / [ w^T (S\_c1+ S\_c2) w ]
    + 分子 = [w^T (1/N\_1 Σ x\_i - 1/N\_2 Σ x\_i)]^2 = [w^T (X\_c1 - X\_c2)]^2 = w^T (X\_c1 - X\_c2)(X\_c1 - X\_c2)^T w
    + 分母 =  w^T S\_c1 w +  w^T S\_c2 w =  w^T (S\_c1+ S\_c2) w
      + S\_1 =  1/N\_1 Σ (w^T x\_i - 1/N\_1 Σ w^T x\_j)(w^T x\_i - 1/N\_1 Σ w^T x\_j)^T=w^T [1/N\_1 Σ (x\_i - X\_c1)(x_i - X\_c1)^T] w = w^T S\_c1 w
    + 定义S\_b类内方差（between-class），S\_w类间方差（with-class）
    + L(w) = w^T S\_b w / w^T S\_w w
    + w~ = argmax L(w)
      + dL/dw = 2\*S\_b w (w^T S\_w w)^-1 + (w^T S\_b w) \* (-1) (w^T S\_w w)^-2 \* 2 \* S\_w w = 0
      + S\_b w (w^T S\_w w) = (w^T S\_b w) S\_w w （(w^T S\_w w) 最终计算得一个实数，一维，没有方向）（求解w～关心的是方向，因为平面的大小可以缩放，所以意义不大）
      + w = (w^T S\_w w)/(w^T S\_b w) \* S\_w^-1 \* S\_b w，正比于(S\_w^-1 \* S\_b w) ，正比于(S\_w^-1 \*(X\_c1 - X\_c2))
      + （S\_b w = (X\_c1 - X\_c2)(X\_c1 - X\_c2)^T w，(X\_c1 - X\_c2)^T w为实数）
      + （若S\_w是对角矩阵，各向同性，S\_w正比于单位矩阵，则w正比于(X\_c1 - tX\_c2)
  + **线性判别分析为早期分类方法，有很大局限性，目前不用**
  
+ **Logistic Regression**

  + 线性回归------>sigmoid------->线性分类
  + sigmoid(x) = 1/(1+e^-x)，将w^T x映射到处于[0,1]区间的概率值p
    + p\_1 = P(y=1|x) = sigmoid(w^T x) = 1/(1+e^(w^T x))
    + p\_0 = P(y=0|x) = sigmoid(w^T x) = e^(w^T x)/(1+e^(w^T x))
    + 综合表达：P(y|x) = p\_1^y * p\_0^(1-y)
  + w~ = argmax P(Y|X) = argmax log Π P(y\_i|x\_i) = argmax Σ log P(y\_i|x\_i) = argmax Σ [y\_i\*log p\_1 + (1-y\_i)\*log p\_0] （-cross entropy）
    + MLE <=> loss function (min cross entropy) 

+ **Gaussian Discriminant Analysis**
  
  + 生成模型、连续
  + Data：N个d维样本，二分类(0,1)，正样本个数N\_1，方差S\_1，负样本个数N\_2，方差S\_2
  + model
    + y~ = argmax P(y|x) = argmax P(x|y)P(y)
    + 分类：最终比较P(y=0|x)，P(y=1|x)大小
    + P(y|x)正比于P(x|y)P(y)，即联合概率P(x, y)
  + **prior probability** 
    + 先验概率服从伯努利分布
    + y \~ Bernoulli，P(y=1) = p，P(y=0) = 1-p
    + P(y) = p^y\*(1-p)^(1-y)
  + **conditional probability**
    + 条件概率服从高斯分布（样本足够大时服从高斯分布）
    + x|y=1 ~ N(u\_1, σ)
    + x|y=0 ~ N(u\_2, σ)
    + 方差一样（权值共享），均值不一样
    + P(x|y) = N(u\_1, σ)^y \* N(u\_2, σ)^(1-y)
  + **loss function**
    + log MLE => L(θ) =  log Π P(x\_i, y\_i) = Σ log [P(x\_i|y\_i)P(y\_i)] = Σ[log P(x\_i|y\_i) + log P(y\_i)] =  Σ[log N(u\_1, σ)^y\_i \* N(u\_2, σ)^(1-y\_i) + log p^y\_i\*(1-p)^(1-y\_i)] = Σ[y\_i\*log N(u\_1, σ) + (1-y\_i)\*log N(u\_2, σ) + y\_i\*log p + (1-y\_i)*log (1-p)]
    + θ = (u\_1, u\_2, σ, p)
    + θ ~ = argmax L(θ)
    + 求解4个参数
      + p
        + 相关部分 L = Σ [log p^y\_i + log (1-p)^(1-y\_i)]
        + dL/dp = Σ [y\_i/p - (1-y\_i)/(1-p)] = 0  =>  Σ [y\_i\*(1-p)- (1-y\_i)\*p] = Σ (y\_i - p) = 0
        + p~ = 1/N Σ y\_i  = N\_1/N（二分类（0,1），Σ y\_i = N\_1）
      + u\_1 （同理 u\_2）
        + 相关部分 L = Σ y\_i\*log N(u\_1, σ) = Σ y\_i\*log [1/((2\*pi)^(d/2)\*σ^(1/2)) exp((x\_i-u\_1)^T(x\_i-u\_1)/-2\*σ)
        + u\_1 = argmax L = argmax Σ y\_i \* [(x\_i-u\_1)^T(x\_i-u\_1)/-2\*σ] =  argmax -1/2 Σ y\_i \* [(x\_i-u\_1)^T(x\_i-u\_1)σ^-1] = argmax -1/2 Σ y\_i \* [x\_i^T σ^-1 x\_i - 2\*u\_1^T σ^-1 x\_i + u\_1^T σ^-1 u\_1] 
        + dL/du\_1 = -1/2 Σ y\_i \* [-2\*σ^-1 x\_i + 2*σ^-1 u\_1] = 0  =>   Σ y\_i \* (u\_1 - x\_i) = 0
        + u\_1 = Σ y\_i x\_i / Σ y\_i = Σ y\_i x\_i / N\_1
      + σ
        + 相关部分 L = Σ[y\_i\*log N(u\_1, σ) + (1-y\_i)\*log N(u\_2, σ)] = Σlog N(u\_1, σ) +  Σlog N(u\_2, σ)
          + （二分类，非0即1，可以拆分算，可以省去y\_i）
        + [详解](https://www.bilibili.com/video/BV1aE411o7qd?p=20)
        + σ = 1/N (N\_1\*S\_1 + N\_2\*S\_2)




