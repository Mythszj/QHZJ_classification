import os
import math
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from torchsampler import ImbalancedDatasetSampler


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
    # 因为字体靠下，所以直接resize就好（randomresize可能不包含字）；使用RandomRotation
    data_transform = {
        "train": transforms.Compose([transforms.Resize([img_size, img_size]),
                                     transforms.RandomRotation(degrees=10),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize([img_size, img_size]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(data='../QHZJ/train_data.npy',
                              label='../QHZJ/train_label.npy',
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(data='../QHZJ/new_test_data_tc.npy',
                            label='../QHZJ/new_test_label_tc.npy',
                            transform=data_transform["val"])
    # 批量大小
    batch_size = args.batch_size

    # 读取线程数
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    # DataLoader，bug：使用sampler不能加shuffle
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=nw,
                                               sampler=ImbalancedDatasetSampler(train_dataset))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=nw)

    # plot_original_image('../QHZJ/train_data.npy')
    # plot_data_loader_image(train_loader)

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

    # 载入预训练权重
    model_weight_path = './pretrain_weights/{}-pre.pth'.format(args.model_name)
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location=device)
    if args.model_name == 'ViT':
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del pre_weights[k]
        model.load_state_dict(pre_weights, strict=False)
    elif args.model_name == 'Swin':
        pre_weights = torch.load(model_weight_path, map_location=device)["model"]
        for k in list(pre_weights.keys()):
            if "head" in k:
                del pre_weights[k]
        model.load_state_dict(pre_weights, strict=False)
    else:
        # delete classifier weights
        pre_dict = {k: v for k, v in pre_weights.items() if model.state_dict()[k].numel() == v.numel()}
        missing_keys, unexpected_keys = model.load_state_dict(pre_dict, strict=False)

    model = model.to(device)

    # 冻结前面的参数（可选）
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # 需要训练的参数
    pg = [p for p in model.parameters() if p.requires_grad]

    # 优化器
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler，根据epoch训练次数来调整学习率
    # 学习率变化函数
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 记录最优值
    best_train_top1_acc = 0.0
    best_train_top5_acc = 0.0
    best_val_top1_acc = 0.0
    best_val_top5_acc = 0.0
    # 训练过程
    for epoch in range(args.epochs):
        # train，训练一个epoch
        train_loss, train_top1_acc, train_top5_acc = train_one_epoch(model=model,
                                                        optimizer=optimizer,
                                                        data_loader=train_loader,
                                                        device=device,
                                                        epoch=epoch,
                                                        loss=args.loss)

        # 更新学习率
        scheduler.step()

        # validate，每个epoch结束进行验证
        val_loss, val_top1_acc, val_top5_acc = evaluate(model=model,
                                                data_loader=val_loader,
                                                device=device,
                                                epoch=epoch,
                                                loss=args.loss)
        # 更新，选择在测试集top1最准的保存
        if train_top1_acc > best_train_top1_acc:
            best_train_top1_acc = train_top1_acc
        if train_top5_acc > best_train_top5_acc:
            best_train_top5_acc = train_top5_acc
        if val_top1_acc > best_val_top1_acc:
            best_val_top1_acc = val_top1_acc
            torch.save(model.state_dict(), "./weights/{}-w.pth".format(args.model_name))
        if val_top5_acc > best_val_top5_acc:
            best_val_top5_acc = val_top5_acc

    # 打印全剧信息
    print('Finish training!')
    print('best_train_top1_acc: {:.3f}'.format(best_train_top1_acc))
    print('best_train_top5_acc: {:.3f}'.format(best_train_top5_acc))
    print('best_val_top1_acc: {:.3f}'.format(best_val_top1_acc))
    print('best_val_top5_acc: {:.3f}'.format(best_val_top5_acc))


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
    parser.add_argument('--model-name', default='ResNet50', help="eg:ResNet50 / ViT / EfficientNet_b7 / Swin")
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # 设备
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    # 解析参数
    opt = parser.parse_args()

    main(opt)
