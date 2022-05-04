import os
import sys
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from focal_loss import FocalLoss

# 计算 topk 准确率，参数：logits预测值，targets是目标值，k是前几
# 修改：返回的是预测准确的个数
def calculate_top_k_accuracy(logits, targets, k=2):
    # values是值，indices是对应的索引
    values, indices = torch.topk(logits, k=k, sorted=True)
    # 转换维度
    y = torch.reshape(targets, [-1, 1])
    # correct是个矩阵，正确预测的位置值是1
    correct = (y == indices) * 1.
    # 计算准确的个数
    top_k_sum = correct.sum()
    # 计算最后的准确率
    top_k_accuracy = torch.mean(correct) * k
    return top_k_sum.item()

# 训练一个epoch
def train_one_epoch(model, optimizer, data_loader, device, epoch, loss):
    # 训练模式
    model.train()
    # 损失函数
    if loss == 'CrossEntropyLoss':
        loss_function = torch.nn.CrossEntropyLoss()
    elif loss == 'FocalLoss':
        loss_function = FocalLoss(num_class=1947)
    # 累计损失
    accu_loss = torch.zeros(1).to(device)
    # 累计预测正确的样本数
    top1_correct_sum = torch.zeros(1).to(device)
    top5_correct_sum = torch.zeros(1).to(device)
    # 清零梯度
    optimizer.zero_grad()
    # 样本总数
    sample_num = 0
    # 封装dataloader，得到进度条
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        # 样本总数
        sample_num += images.shape[0]
        # 前向计算
        pred = model(images.to(device))
        # 计算损失
        loss = loss_function(pred, labels.to(device))
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 清零梯度
        optimizer.zero_grad()
        # 正确数
        top1_correct = calculate_top_k_accuracy(pred, labels.to(device), k=1)
        top5_correct = calculate_top_k_accuracy(pred, labels.to(device), k=5)
        # 累计
        top1_correct_sum += top1_correct
        top5_correct_sum += top5_correct
        # pred_classes = torch.max(pred, dim=1)[1] # torch.max返回两个，values indices，取[1]就是indices
        # accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        # 累计损失
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, top1_acc: {:.3f}, top5_acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               top1_correct_sum.item() / sample_num,
                                                                               top5_correct_sum.item() / sample_num)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

    return accu_loss.item() / (step + 1), top1_correct_sum.item() / sample_num, top5_correct_sum.item() / sample_num


# 进行验证，不需要计算梯度
@torch.no_grad()
def evaluate(model, data_loader, device, epoch, loss):
    # 损失函数
    if loss == 'CrossEntropyLoss':
        loss_function = torch.nn.CrossEntropyLoss()
    elif loss == 'FocalLoss':
        loss_function = FocalLoss(1947)
    # 验证模式
    model.eval()
    # 累计预测正确的样本数
    top1_correct_sum = torch.zeros(1).to(device)
    top5_correct_sum = torch.zeros(1).to(device)
    # 累计损失
    accu_loss = torch.zeros(1).to(device)
    # 样本总数
    sample_num = 0
    # 封装dataloader
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        # 样本总数
        sample_num += images.shape[0]
        # 前向计算
        pred = model(images.to(device))
        # 正确数
        top1_correct = calculate_top_k_accuracy(pred, labels.to(device), k=1)
        top5_correct = calculate_top_k_accuracy(pred, labels.to(device), k=5)
        # 累计
        top1_correct_sum += top1_correct
        top5_correct_sum += top5_correct
        # 计算正确的数目
        # pred_classes = torch.max(pred, dim=1)[1]
        # accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        # 计算累计损失
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss
        # 展示的是平均损失和正确率
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, top1_acc: {:.3f}, top5_acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               top1_correct_sum.item() / sample_num,
                                                                               top5_correct_sum.item() / sample_num)

    return accu_loss.item() / (step + 1), top1_correct_sum.item() / sample_num, top5_correct_sum.item() / sample_num


# 展示原始图片
def plot_original_image(data_path):
    plot_num = 4
    sum = 0
    dataset = np.load(data_path)
    for data in dataset:
        sum += 1
        image = Image.fromarray(data)
        image.show()
        if sum == 10:
            break


# 展示处理过后的图片
def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(str(label))
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()