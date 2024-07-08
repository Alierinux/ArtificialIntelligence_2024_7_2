import torch
import sklearn # 数据集
import sklearn.datasets

data, target = sklearn.datasets.load_iris(return_X_y=True)

x = torch.Tensor(data[0:100])
y = torch.Tensor(target[0:100]).view(100, 1)

# # 构建模型：线性运算模型
# y = w*x+b
# # (x1,x2,x3,x4)  (y)
# # (w1,w2,w3,w4)
# # 权重-创建学习参数：使用随机数
w = torch.randn(1, 4)
b = torch.randn(1)
print(w)
print(b)

#
# # 记录操作
w.requires_grad = True
b.requires_grad = True
#
epoch = 1000
learning_rate = 0.01
#
# 批大小  ====  轮大小
for e in range(epoch):
    # 目标值 y    预测值:y_？
    y_ = torch.nn.functional.linear(input=x, weight=w, bias=b)

    # 激活函数:(0 1)范围 s形函数
    sy_ = torch.sigmoid(y_)
    # 损失函数 交叉熵损失函数   :基于w,b的计算/操作
    loss = torch.nn.functional.binary_cross_entropy(sy_, y, reduction="mean")
    loss.backward()
    with torch.autograd.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()

        sy_[sy_ > 0.5] = 1
        sy_[sy_ <= 0.5] = 0
        correct_rate = (sy_ == y).float().mean()
        print(F"轮数：{e:05d},损失：{loss:10.6f},准确率：{correct_rate*100.0:10.2f}%")