# y = x**2 - 2*x +1
# y = (x-1)^2     x = 1  最优值
# x = 1000   随机值
# 求导
# x = x - learing_rate(导数)
# x ====> 1
# e = f(w)
import torch
# 随机值
x = torch.Tensor([100])
# 求导数
x.requires_grad = True
# 学习率
learning_rate = 0.1

# 迭代 梯度下降
epoch = 100
x_list = []

for e in range(epoch):
    # 损失函数
    y = x ** 2 - 2 * x + 1
    y.backward(retain_graph=True)
    with torch.autograd.no_grad():
        x -= learning_rate * x.grad
        x_list.append(x.detach().clone().numpy())
        # 导数清0
        x.grad.zero_()

print(x.detach().clone().numpy())
# 可视化
import matplotlib.pyplot as plt

plt.plot(range(epoch), x_list)
plt.show()