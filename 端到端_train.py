import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from 端到端_model import PCB
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
import numpy as np

# 设置 PyTorch 随机种子
torch.manual_seed(0)
# 设置 NumPy 随机种子
np.random.seed(0)
# 在使用 GPU 时，设置 CUDA 随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
# 其他 PyTorch 相关的随机数设置
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import random
random.seed(0)  # 设置随机种子为123

# 设置超参数
batch_size = 42#32
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("最终使用的设备：", device)


padding_width = 20
padding_height = 0
import random
# 定义填充函数
def pad_with_prob(img):
    if random.random() < 1:  # 50%的概率进行填充操作
        return transforms.Pad((padding_width, padding_height), fill=(123, 117, 104), padding_mode='constant')(img)
    else:
        return img

transform = transforms.Compose([
    #使用RandomApply以一定概率应用填充操作,p=0.5太大了，换成0.2比较好
    transforms.Resize((336, 168), interpolation=InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.6, scale=(0.02, 0.2), ratio=(0.3, 2), value=[0.485, 0.456, 0.406], inplace=False)
])

# 创建数据集和数据加载器
train_dataset = ImageFolder("bounding_box_train", transform=transform)
# train_dataset = ImageFolder("CUHK03_labled/datasets/bounding_box_train", transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
from own_sampler import BalancedBatchSampler
batch_sampler_train = BalancedBatchSampler(train_dataset, n_classes=batch_size//2, n_samples=2)
train_loader= DataLoader(train_dataset, batch_sampler=batch_sampler_train,
                                         num_workers=8)# 8 workers may work faster

# 一个 epoch（训练轮次）中需要迭代的次数
total_steps = len(train_loader)

# 初始化模型
num_classes = len(train_dataset.classes)
part = 3
model = PCB(num_classes, part)
# print(num_classes)
# in_features = model.fc.in_features
# model.fc = torch.nn.Linear(in_features, num_classes)
model.to(device)

# 设置优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9, nesterov=True)
# # 定义优化器，例如使用Adam优化器
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
from torch.optim import lr_scheduler
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
# 定义学习率调整策略，指定在训练轮次为20、40、50、60、70时学习率乘以0.1
# from torch.optim.lr_scheduler import MultiStepLR
# milestones = [20, 40, 50, 60, 70]
# exp_lr_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

# # # 定义分类损失函数
CrossEntropy_Loss = torch.nn.CrossEntropyLoss().cuda()

class TripletLoss:
    def __init__(self, margin=0.8):
        self.margin = margin

    def compute_triplets(self, outputs, labels):
        pos_adj_mat = (labels.unsqueeze(0) == labels.unsqueeze(1)).byte()
        neg_adj_mat = pos_adj_mat ^ 1
        pos_adj_mat.fill_diagonal_(0)
        triplets = torch.where(pos_adj_mat.unsqueeze(2) * neg_adj_mat.unsqueeze(1))
        triplets = (triplets[0].to(outputs.device), triplets[1].to(outputs.device), triplets[2].to(outputs.device))
        return triplets

    def compute_loss(self, outputs, labels):
        eps = 1e-8

        x_norm = F.normalize(outputs + eps, p=2, dim=1)

        triplets = self.compute_triplets(x_norm, labels)

        pos_loss = torch.mean(torch.norm(x_norm[triplets[0]] - x_norm[triplets[1]], p=2, dim=1))
        neg_loss = torch.mean(torch.norm(x_norm[triplets[0]] - x_norm[triplets[2]], p=2, dim=1))

        loss = torch.clamp(pos_loss - neg_loss + self.margin, min=eps)
        loss = loss[loss > 0].mean()

        return loss
triplet_loss = TripletLoss(margin= 1.1)#1.1
triplet_loss1 = TripletLoss(margin=0.9)#0.9大

# 定义 best_loss 初始值，设为无穷大
best_loss = float('inf')
best_rank1 = float('-inf')

for epoch in range(num_epochs):
    sum_loss=0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()  # 清除之前计算过程中的梯度信息，确保每次迭代时都是新的梯度
        # 前向传播
        output, outputs,fea = model(images)
        ##############################

        ce_loss = CrossEntropy_Loss(output, labels)
        ce_tri_loss = triplet_loss.compute_loss(output, labels)

        ls_loss = CrossEntropy_Loss(outputs[0], labels)
        for j in range(part-1):
            ls_loss += CrossEntropy_Loss(outputs[j + 1], labels)

        tri_features = torch.cat((outputs[0], outputs[1], outputs[2]), dim=1)
        tri_loss = triplet_loss.compute_loss(tri_features, labels)
        # tri_loss = triplet_loss1.compute_loss(fea, labels)
        loss = ce_loss + ce_tri_loss+ls_loss+tri_loss


        loss.backward()#计算损失函数关于模型参数的梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), 20)  # 限制梯度的范数，以防止梯度爆炸的问题
        optimizer.step()#根据计算得到的梯度更新模型的参数
        sum_loss += loss.item()
        # 打印训练信息
        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], ce_loss:{:.4f} ,ce_tri_loss:{:.4f} ,tri_loss:{:.4f} ,cent_loss:{:.4f} ,Loss: {:.4f} '
                  .format(epoch + 1, num_epochs, i + 1, total_steps,ce_loss.item(), ce_tri_loss.item(), ls_loss.item(),
                          tri_loss.item(), loss.item()))
    exp_lr_scheduler.step()
    print('第{}轮训练结束总损失：{}'.format(epoch + 1, sum_loss))
    if epoch <= 70:
        if sum_loss < best_loss:
            best_loss = sum_loss
            torch.save(model.state_dict(), "end.pth")
            print('新的模型已经保存')

import os
os.system("shutdown")



