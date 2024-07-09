import torch
class LeNet(torch.nn.Module):
    def  __init__(self):
        super().__init__()
        #准备工作
        # 卷积运算
        self.layer1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=(5,5),
            padding=2)
        self.layer2 = torch.nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=(5,5),
            padding=0)
        self.layer3 = torch.nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=(5,5),
            padding=0)
        self.layer4 = torch.nn.Linear(120,84)
        self.layer5 = torch.nn.Linear(84, 10)
    def foward(self,input):
        # N * C * H * W   N * 1 * 28 *28
        # 模型计算过程
        o1 = self.layer1(input)
        #  N * 6 * 28 * 28
        o1 = torch.nn.functional.max_pool2d(o1,kernel_size=(2,2))
        #  N * 6 * 14 * 14

        o2 = self.layer1(o1)
        #  N * 16 * 10 * 10
        o2 = torch.nn.functional.max_pool2d(o2, kernel_size=(2, 2))
        #  N * 16 * 5 * 5

        o3 = self.layer1(o2)
        #  N * 120 * 1 * 1

        o3 = o3.squeeze()
        # N*120

        o4 = self.layer4(o3)
        # N * 84
        o5 = self.layer5(o4)
        # N * 10

        return o5


# model = LeNet()
# y = model(x)  #  自动调用LeNet的foward方法