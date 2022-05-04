"""
在TO上测试，算将未见过的类别归为已有类别的概率
"""
import os
import math
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from torchsampler import ImbalancedDatasetSampler
import numpy as np
from PIL import Image

from my_dataset import MyDataSet
from models.ViT import vit_base_patch16_224_in21k
from models.ResNet import resnet34, resnet50, resnet101
from models.EfficientNet import efficientnet_b3, efficientnet_b5, efficientnet_b7
from models.Swin import swin_base_patch4_window7_224
from utils import train_one_epoch, evaluate, plot_data_loader_image, plot_original_image


def main(args):
    # 设备 gpu cpu
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 默认图像大小都是224，只有efficientnet需要判断
    img_size = 224
    if 'EfficientNet' in args.model_name:
        # EfficientNet对应输入图片的大小
        efficientnet_img_size = {"b0": 224, "b1": 240, "b2": 260, "b3": 300, "b4": 380, "b5": 456, "b6": 528, "b7": 600}
        num_model = args.model_name[-2:]
        img_size = efficientnet_img_size[num_model]

    # 数据预处理
    data_transform = transforms.Compose([transforms.Resize([img_size, img_size]),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 创建模型
    if args.model_name == 'ResNet50':
        model = resnet50(num_classes=args.num_classes).to(device)
    elif args.model_name == 'ResNet101':
        model = resnet101(num_classes=args.num_classes).to(device)
    elif args.model_name == 'ViT':
        model = vit_base_patch16_224_in21k(num_classes=args.num_classes, has_logits=False).to(device)
    elif args.model_name == 'EfficientNet_b3':
        model = efficientnet_b3(num_classes=args.num_classes).to(device)
    elif args.model_name == 'EfficientNet_b5':
        model = efficientnet_b5(num_classes=args.num_classes).to(device)
    elif args.model_name == 'EfficientNet_b7':
        model = efficientnet_b7(num_classes=args.num_classes).to(device)
    elif args.model_name == 'Swin':
        model = swin_base_patch4_window7_224(num_classes=args.num_classes).to(device)
    else:
        raise Exception("No model name {}".format(args.model_name))

    # 载入训练好的权重
    model_weight_path = './weights/{}-w.pth'.format(args.model_name)
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location=device)
    model.load_state_dict(pre_weights, strict=False)

    # 开集测试
    test_data = np.load('../QHZJ/new_test_data_to.npy')
    test_label = np.load('../QHZJ/new_test_label_to.npy')

    # prediction
    model.eval()

    total = 0
    wrong = 0

    with torch.no_grad():
        for data in test_data:
            print(wrong, total)
            img = Image.fromarray(data)
            img = data_transform(img)
            img = torch.unsqueeze(img, dim=0)
            # predict class
            output = torch.squeeze(model(img.to(device)))
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            max_assumption = torch.max(predict)
            total += 1
            if max_assumption >= 0.7:
                wrong += 1

    print(f'{args.model_name}将未见过的类别归为已有类别的概率：{wrong / total:.3f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 分类的类别
    parser.add_argument('--num_classes', type=int, default=1947)
    # 迭代的epoch
    parser.add_argument('--epochs', type=int, default=15)
    # 每个batch的大小
    parser.add_argument('--batch-size', type=int, default=8)
    # 学习率
    parser.add_argument('--lr', type=float, default=0.001)
    # 损失函数
    parser.add_argument('--loss', type=str, default='CrossEntropyLoss')
    # 最终学习率（乘以）
    parser.add_argument('--lrf', type=float, default=0.01)
    # 模型名称，ResNet，ViT
    parser.add_argument('--model-name', default='EfficientNet_b3', help="eg:ResNet50 / ViT / EfficientNet_b7 / Swin")
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # 设备
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    # 解析参数
    opt = parser.parse_args()

    main(opt)
