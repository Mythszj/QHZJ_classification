import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import os


class MyDataSet(Dataset):
    # 参数data和label都是.npy文件
    def __init__(self, data, label, transform):
        # 确认文件存在
        assert os.path.exists(data), "dataset root: {} does not exist.".format(data)
        assert os.path.exists(label), "dataset root: {} does not exist.".format(label)
        # 读取
        self.data = np.load(data)
        self.label = np.load(label)
        # bug1：label的dtype是<U4会报错，需要转换
        self.label = torch.from_numpy(self.label.astype(np.uint8))
        # bug2：torch.NLLLoss需要LongTensor类型
        self.label = self.label.type(torch.LongTensor)
        self.transform = transform

    # 该数据集下所有样本的个数
    def __len__(self):
        return self.data.shape[0]

    # 根据索引返回image，label
    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        # bug3：不知道改的对不对，transform只能用于PIL，所以进行了转换
        data = transforms.ToPILImage()(np.uint8(data))
        if self.transform is not None:
            data = self.transform(data)
        return data, label

    # 使用ImbalancedDatasetSampler需要添加的函数
    def get_labels(self):
        return self.label