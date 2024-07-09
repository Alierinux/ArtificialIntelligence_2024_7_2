import os.path
from torchvision.models import resnet18
import torch
from loaddata import load_data
class TrainResNet18:
    def __init__(self, img_dir="./dataset", epoch=100, batch_size=128, learing_rate=0.01):
        super().__init__()
        print("准备训练....")
        # 模型保存的位置
        self.model_file = "garbage.mod"
        # GPU是否可用 True False
        self.CUDA = torch.cuda.is_available()
        # 数据集
        self.tr, self.ts, self.cls_idx = load_data(img_dir, batch_size=batch_size)
        # 初始化模型结构
        self.net = resnet18()
        fc_in = self.net.fc.in_features
        self.net.fc = torch.nn.Linear(fc_in, 40)
        if self.CUDA:
            self.net.cuda()  #cpu==>gpu
        if os.path.exists(self.model_file):
            print("加载本地模型，继续训练")
            state_dict = torch.load(self.model_file)
            self.net.load_state_dict(state_dict)
        else:
            print("从头训练")

        self.lr = learing_rate
        self.epoch = epoch
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.loss_function = torch.nn.CrossEntropyLoss()
        if self.CUDA:
            self.loss_function = self.loss_function.cuda()

    def train(self):
        print("开始训练")
        for e in range(self.epoch):
            self.net.train()
            for samples, labels in self.tr:
                self.opt.zero_grad()
                if self.CUDA:
                    samples = samples.cuda()
                    labels = labels.cuda()
                y = self.net(samples.view(-1,3,224,224))
                loss = self.loss_function(y, labels)
                loss.backward()
                self.opt.step()
            # 测试
            c_rate = self.validate()
            print(F"轮数：{e},准确率：{c_rate}")
            # 保存模型
            torch.save(self.net.state_dict(),self.model_file)
    @torch.no_grad()
    def validate(self):
        num_samples = 0
        num_correct = 0
        for samples, labels in self.ts:
            if self.CUDA:
                samples = samples.cuda()
                labels = labels.cuda()
            num_samples += len(samples)
            out = self.net(samples.view(-1,3,224,224))
            out = torch.nn.functional.softmax(out, dim=1)
            y = torch.argmax(out,dim=1)
            num_correct += (y==labels).float().sum()
        return  num_correct * 100 / num_samples

t = TrainResNet18(epoch=2)
t.train()