---
title: "与DeepSeek聊统计学基本概念2"
datePublished: Sun Oct 19 2025 08:30:57 GMT+0000 (Coordinated Universal Time)
cuid: cmgxg5ual000302k0bt1pe2hc
slug: deepseek2
cover: https://cdn.hashnode.com/res/hashnode/image/stock/unsplash/dBI_My696Rk/upload/8fbb03a088ba2e6d25edaae2b58af327.jpeg

---

## 参数估计方法

在统计学中一个重要的内容是根据样本信息来估计总体信息，实际情况是我们智能获取样本数据，所以必须根据样本统计量来估计总体参数，这就是参数估计。 参数估计主要有两种方式：

1. 点估计：估计一个人的身高是170cm，这就是点估计
    
2. 区间估计：估计一个人的身高在168-172cm之间，这就是参数估计
    

## 最小二乘估计

在平面上有一些点，它们看上去大致呈一种曲线的趋势。选择一个合适的曲线表达式，让这个曲线尽可能“贴近”所有的这些点。

* “二乘”：其实就是平方的意思，计算距离的平方，这样可以处理距离计算的负值
    
* “最小”：找到使得这个平方和最小的曲线的参数
    

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760669905553/a944a8e6-c7cc-4553-ac4d-c06b40cd666a.png align="center")

上面的图像散点是我基于一条真实函数（绿色）增加了一些噪声生成的，红色是由最小二乘法拟合生成的。

对于这种点分布，假设用一条一次函数来做拟合：

\\(y=ax+b\\)

那么有点误差

\\(e_i = y_i - (ax_i + b)\\)

我们的目标是让所有点的误差平方和最小。这个平方和（我们用 `S` 表示）就是我们的目标函数:

\\(S=\sum_{i=1}^{n}e_i^2=\sum_{i=1}^{n}[y_i - (ax_i + b)]^2\\)

接下来的目标就是找到最小值，需要求解a和b.这一步可以直接让计算机来算了，并不是统计专业的所以直接列公式。计算公式如下：

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760674758796/ce4f7066-2a28-4028-a9e3-5353784fddda.png align="center")

在python中也有一些能够计算最小二乘法的函数：

```python
# numpy.linalg.lstsq
import numpy as np
beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

# scipy.linalg.lstsq
# 与 numpy 版本类似，来自 SciPy 的 LAPACK 封装，选项更丰富
from scipy.linalg import lstsq
beta, residuals, rank, s = lstsq(X, y)  # rcond 等参数可选

# scikit-learn 线性模型
# LinearRegression（普通最小二乘）
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X, y)
lr.coef_, lr.intercept_
```

下面这是一个实例，首先选择一条一次函数，添加一些噪声生成许多散点，然后用 `np.linalg.lstsq` 做最小二乘法：

```python
import numpy as np
import matplotlib.pyplot as plt

# 设置字体为 LXGW WenKai
plt.rcParams['font.sans-serif'] = ['LXGW WenKai']
plt.rcParams['axes.unicode_minus'] = False

# 真实函数参数
a_true = 1.5
b_true = 2.0

# 生成自变量数据
np.random.seed(42)  # 保证可复现
x = np.linspace(0, 10, 100)

# 在真实函数基础上加高斯噪声
noise = np.random.normal(loc=0, scale=1, size=x.shape)  # 均值0，标准差1
y = a_true * x + b_true + noise

# 最小二乘法拟合 y = ax + b
A = np.vstack([x, np.ones(len(x))]).T
a_fit, b_fit = np.linalg.lstsq(A, y, rcond=None)[0]

print(f"真实函数：y = {a_true}x + {b_true}")
print(f"拟合结果：y = {a_fit:.3f}x + {b_fit:.3f}")

# 绘图
plt.figure(figsize=(8,5))
plt.scatter(x, y, color='blue', s=12, label='带噪声的样本点')
plt.plot(x, a_true*x + b_true, color='green', linewidth=2, label='真实函数')
plt.plot(x, a_fit*x + b_fit, color='red', linewidth=2, label='最小二乘拟合')
plt.title('最小二乘法线性拟合（带噪声数据）')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
```

## 贝叶斯估计

贝叶斯估计的核心思想就是：**用新的证据来更新我们原有的信念**。

`先验信念 + 新证据 → 更新 → 后验信念`

贝叶斯公式：

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760676274795/dabc8bf0-75d5-4931-ae8b-0c4617a7a0be.png align="center")

* P(A|B) 后验概率：在观察到B之后，A发生的概率
    
* P(B|A) 似然度：如果A发生，观察到B的概率
    
* P(A) 先验概率：在观察任何证据之前，A发生的概率
    
* P(B) 证据概率：观察到B的总概率
    

全概率公式：

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760676359126/f53ab12c-e511-4efc-8393-ed7325a3d995.png align="center")

**经典例子：**  
假设某种疾病的发病率是1%，即在随机人群中，每100人约有1人患病。

令A表示患病，B表示检测结果为阳性。那么有先验概率P(A)=1%

现在有一种检测方法：

* 如果确实患病，检测为**阳性**的概率是99%，那么有似然度P(B|A)=99%
    
* 如果健康，检测为**阳性**的概率是5%（有5%的误诊率）
    

求后验概率 P(A|B)：在检测为阳性的条件下，真正患病的概率是多少？

首先用全概率公式计算P(B)，及事件B发生的总概率，等于在各种可能"原因"下B发生的概率的加权和。这里就是计算检测结果为阳性的概率，那么可能是这个人病了检测阳性，也有可能是没病检测阳性：

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760677250471/c457bc32-a67a-421a-bfe9-b510b611c8c4.png align="center")

再将上述值带入贝叶斯公式可以计算得到“检测阳性，并且患病”的概率

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760677354437/96d95380-85cd-465c-82d7-239aba0915bb.png align="center")

## t检验

t检验帮助我们判断两个群体之间的差异到底是**真实的**，还是只是**偶然巧合**。

假设我们想知道一种新的学习方法是否真的能提高成绩：

* A组使用传统方法，平均分75
    
* B组使用新方法，平均分80
    

现在需要评估5分的差异是新方法有效，还是运气好。

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760679913737/caf0da8d-1d2b-45d9-add3-a6374cadbe74.png align="center")

通俗解释公式：t=组间差异(信号)/组内变异(噪声)

在实际的公式中，组间差异用均值差异B-bar和A-bar的差来表示; 组内噪声用标准差来表示。t 值越大，说明"信号"越强于"噪声"，差异越可能是真实的。

接下来用一组实例来表示t检验的流程

A组（传统方法）成绩：  
65, 72, 68, 75, 70, 62, 69, 74, 71, 67, 73, 66, 70, 68, 72, 69, 71, 67, 74, 70, 66, 73, 68, 72, 69

B组（新方法）成绩：  
72, 78, 75, 80, 77, 70, 76, 79, 78, 74, 81, 73, 77, 75, 79, 76, 78, 74, 80, 77, 73, 79, 75, 78, 76

计算得到AB两组的平均成绩：

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760679561512/349bbf24-d7a0-4ccf-ba3c-0e4d34b96d31.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760679567437/4021963f-fc86-4f8f-9452-800f4595a1e9.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760679593712/e51b6e93-eee7-4b17-bed6-1e68cdd84d24.png align="center")

计算AB两组的方差：

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760679688545/b7973f0d-50c3-4552-9825-2cd365878e62.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760679704746/0d98589f-711b-4be9-93ea-c44208743f03.png align="center")

  
计算合并方差与标准差：

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760679805781/c2100fb0-34e8-4a23-93f8-dc32b0668628.png align="center")

那么将上面计算出来的这些数据带入t检验公式可以得到：

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760680921727/5f92d02d-3121-4c28-97e4-16328fac4d33.png align="center")

自由度df=50-2=48,显著水平alpha=5%，查表可得t的临界值为2.011

也就是说t&gt;2.011就说明差异显著。而事实情况是7.492&gt;2.001，那么说明新方法有效。