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



