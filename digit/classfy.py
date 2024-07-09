# Lenet：model结构     pth文件：模型参数
import torch
import cv2
from lenet import  LeNet
model = LeNet()
state = torch.load('./lenet.pth')
model.load_state_dict(state)
img = cv2.imread('5.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# N C H W
img = cv2.resize(img,(28,28))
img = 255 - img
img = torch.Tensor(img).view(1,1,img.shape[0],img.shape[1])
print(img.shape)
y = model(img)
# N C H W    N*1
y = torch.nn.functional.log_softmax(y,dim=0)
predict = torch.argmax(y,dim=0)
print(predict.numpy())

