from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import  random_split, DataLoader

def load_data(img_dir, train_rate=0.8, batch_size=128):
    transform = transforms.Compose([
        transforms.RandomResizedCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.56719673, 0.5293289, 0.48351972], std=[0.20874391, 0.21455203, 0.22451781])
    ])
    # (数据集路径，数据集预处理)
    ds = ImageFolder(img_dir, transform=transform)
    # print(len(ds))
    # print(ds.classes)
    # print(ds.class_to_idx)
    # 手动切分训练集和测试集
    l = len(ds)
    num_train = int(l*train_rate)
    train, test = random_split(ds, [num_train, l-num_train])
    # dataloader
    train_loader = DataLoader(dataset=train, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(dataset=test, shuffle=True, batch_size=batch_size)
    return train_loader, test_loader, ds.classes

# train_l, test_l, cs= load_data("./dataset")


