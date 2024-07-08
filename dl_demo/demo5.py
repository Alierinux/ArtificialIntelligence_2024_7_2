import torch
import sklearn.datasets
# 1、数据集准备
data, target = sklearn.datasets.load_iris(return_X_y=True)
x = torch.Tensor(data)
y = torch.LongTensor(target)

# 2、构建算法模型
w1 = torch.randn(10,4)
b1 = torch.randn(10)
w1.requires_grad = True
b1.requires_grad = True
w2 = torch.randn(6,10)
b2 = torch.randn(6)
w2.requires_grad = True
b2.requires_grad = True
w3 = torch.randn(3,6)
b3 = torch.randn(3)
w3.requires_grad = True
b3.requires_grad = True
def foward(inupt):
    # N*X(x1,x2,x3,x4)
    # 4 10 6 3
    l1 = torch.nn.functional.linear(inupt,w1,b1)
    l2 = torch.nn.functional.linear(l1, w2, b2)
    output = torch.nn.functional.linear(l2, w3, b3)
    return output

# 3.训练：轮 批  y_ = model(x)  损失（误差）函数、学习率-优化器
epoch = 10000
learning_rate = 0.001
for e in range(epoch):
    y_ = foward(x)
    loss = torch.nn.functional.cross_entropy(y_,y)
    loss.backward()
    with torch.autograd.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w3 -= learning_rate * w3.grad
        b1 -= learning_rate * b1.grad
        b2 -= learning_rate * b2.grad
        b3 -= learning_rate * b3.grad
        w1.grad.zero_()
        w2.grad.zero_()
        w3.grad.zero_()
        b1.grad.zero_()
        b2.grad.zero_()
        b3.grad.zero_()
    result = y_.log_softmax(dim=1)
    result = result.argmax(dim=1)
    print(result)

# 4.验证模型、保存模型