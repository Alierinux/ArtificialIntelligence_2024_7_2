import struct
import numpy as np

def load_image_fromfile(filename):
    with open(filename, 'br') as fd:
        # 读取图像的信息
        header_buf = fd.read(16)   # 16字节，4个int整数
        # 按照字节解析头信息（具体参考python SL的struct帮助）  解包
        magic_, nums_, width_, height_ = struct.unpack('>iiii', header_buf)  # 解析成四个整数：>表示大端字节序，i表示4字节整数
        # 保存成ndarray对象
        imgs_ = np.fromfile(fd, dtype=np.uint8)
        imgs_ = imgs_.reshape(nums_, height_, width_)
    return imgs_


def load_label_fromfile(filename):
    with open(filename, 'br') as fd:
        header_buf = fd.read(8)
        magic, nums = struct.unpack('>ii' ,header_buf)
        labels_ = np.fromfile(fd, np.uint8)
    return labels_


train_imgs = load_image_fromfile('data/train-images.idx3-ubyte')
train_labels = load_label_fromfile('data/train-labels.idx1-ubyte')
print(train_imgs.shape)
print(train_labels.shape)
test_imgs = load_image_fromfile('data/t10k-images.idx3-ubyte')
test_labels = load_label_fromfile('data/t10k-labels.idx1-ubyte')
print(test_imgs.shape)
print(test_labels.shape)
img1 = train_imgs[50000]
import cv2
cv2.imwrite("img.png",img1)
