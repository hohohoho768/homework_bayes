import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 

# 优化的程度
data = np.array([1.5, 2.0, 1.8, 2.1, 1.7])

# 使用正态分布拟合数据
mu, std = stats.norm.fit(data)

# 现在你可以使用这些参数来定义你的分布函数
target_dist = stats.norm(mu, std)


# MCMC参数设置
n_samples = 100000
burn_in = n_samples//5*3  # 舍弃样本数
proposal_std = 0.9  # 提议分布的标准差
 
# 初始化状态
current_state = target_dist.rvs()
 
# 存储采样结果
samples = [current_state]
 
for _ in range(n_samples - 1):
    # 提议新的状态
    proposed_state = current_state + proposal_std * np.random.randn()
 
    # 计算接受概率
    acceptance_ratio = target_dist.pdf(proposed_state) / target_dist.pdf(current_state)
 
    # Metropolis-Hastings接受规则
    if np.random.rand() < acceptance_ratio:
        current_state = proposed_state
 
    samples.append(current_state)
 
# 舍弃样本
posterior_samples = samples[burn_in:]
 
# 绘制采样结果与目标分布
plt.hist(posterior_samples, bins=50, density=True, alpha=0.5, label='MCMC Samples')
x = np.linspace(-4, 4, 1000)
plt.plot(x, target_dist.pdf(x), 'k', lw=2, label='True Distribution')

mu, std = stats.norm.fit(posterior_samples)
target_dist = stats.norm(mu, std)
x = np.linspace(-4, 4, 1000)
plt.plot(x, target_dist.pdf(x), 'r', lw=1, label='Simulated Distribution')

plt.legend()
plt.savefig("./1.jpg")
plt.show()
