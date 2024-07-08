import torch
# 定义张量
x = torch.Tensor([5])   # 5
#会开始跟踪针对 x的所有操作。
x.requires_grad=True

# 操作/计算1:求平方
y = x**2   # 2*x
# 调用 .backward() 来自动计算所有梯度
y.backward()

# 张量的梯度将累加到.grad 属性中
print(x.grad)  # 10
# 设置为0
x.grad.zero_()
# #操作
# #
# # /计算2：
z = 2*x   # 2
z.backward()
print(x.grad)   # 12