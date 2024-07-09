from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
import json

model = resnet18(weights=ResNet18_Weights.DEFAULT)
# 定义图像转换
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载预训练的 ResNet-18 模型

model.eval()  # 设置为评估模式

# 加载和预处理图像
img = Image.open("datasets/酒瓶/fimg_3634.jpg")
img_tensor = preprocess(img)
img_tensor = img_tensor.unsqueeze(0)  # 添加批次维度

# 执行前向传播，获得预测结果
with torch.no_grad():
    output = model(img_tensor)

# 获取预测的类别
_, predicted = torch.max(output, 1)
print("Predicted class:", predicted.item())

# 如果需要知道类别的标签，可以加载 ImageNet 的类别标签
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = json.loads(requests.get(LABELS_URL).text)
print("Predicted label:", labels[predicted.item()])
