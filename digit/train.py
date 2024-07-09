import load
import torch
from lenet import  LeNet
# load.load_image_fromfile()
# load.load_label_fromfile()
train_imgs = load.load_image_fromfile('data/train-images.idx3-ubyte')
train_labels = load.load_label_fromfile('data/train-labels.idx1-ubyte')
test_imgs = load.load_image_fromfile('data/t10k-images.idx3-ubyte')
test_labels = load.load_label_fromfile('data/t10k-labels.idx1-ubyte')
# print(train_imgs.shape)
# data==》Tenser
# 多分类：交叉熵损失函数：（y）长整型
# N * 28 * 28 ===>   N * C * H * W
x = torch.Tensor(train_imgs).view(
    train_imgs.shape[0],
    1,
    train_imgs.shape[1],
    train_imgs.shape[2])
y = torch.LongTensor(train_labels)

test_x = torch.Tensor(test_imgs).view(
    test_imgs.shape[0],
    1,
    test_imgs.shape[1],
    test_imgs.shape[2])
test_y = torch.LongTensor(test_labels)

train_dataset = torch.utils.data.TensorDataset(x, y)
test_dataset = torch.utils.data.TensorDataset(test_x,test_y)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=1024)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,shuffle=True,batch_size=10000)


model = LeNet()
epoch = 100
# 损失函数
crel = torch.nn.CrossEntropyLoss()
# 优化器
opt = torch.optim.Adam(model.parameters(), lr=0.01)
# 轮epoch
for e in range(epoch):
    # 批 batch
    for data,target in train_loader:
        # 小批量梯度下降
        opt.zero_grad()
        output = model(data)
        loss = crel(output,target)
        loss.backward()
        # 更新权重
        opt.step()
    # 测试当前轮 模型的准确率和损失值  测试集
    with torch.no_grad():
        for x,y in test_loader:
            y_ = model(x) # N*[,,,,,]
            y_ = torch.nn.functional.log_softmax(y_,dim=1)
            predict = torch.argmax(y_,dim=1)
            c_rate= (predict == y).float().mean()
            print(f"轮：{e},------准确率：{c_rate}")
            # [1,1,1,0]
            # [1,1,0,0]
            # [1,1,0,1]  (1+1+0+1)/4
    # 保存模型
    state_dict = model.state_dict()
    torch.save(state_dict, './lenet.pth')



