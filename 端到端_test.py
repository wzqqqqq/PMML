import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import re
from 端到端_model import PCB
from torchvision.transforms.functional import InterpolationMode

part = 3
model = PCB(767, part)

# checkpoint = torch.load("reid_model_Triplet_all_CA.pth")  # 加载保存的模型和优化器状态
# model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型参数
model.load_state_dict(torch.load("end.pth"))
# model.load_state_dict(torch.load("net_last.pth"))
#这个取消的不是model.classifier0，而是model.classifier0.classifier
# model.model.avgpool = nn.Sequential()
# model.model.fc = nn.Sequential()
# Remove the final fc layer and classifier layer
model.classifierAll.classifier = nn.Sequential()

model.classifier0.classifier = nn.Sequential()
model.classifier1.classifier = nn.Sequential()
model.classifier2.classifier = nn.Sequential()

# model.classifier3.classifier = nn.Sequential()
# model.classifier4.classifier = nn.Sequential()
# model.classifier5.classifier = nn.Sequential()

# model.classifier6.classifier = nn.Sequential()
# model.classifier7.classifier = nn.Sequential()
# model.classifier8.classifier = nn.Sequential()
# in_features = model.fc.in_features
# model.fc = torch.nn.Linear(in_features, 1000)
#将模型设置为推理模式
model.to('cuda')
model.eval()

# 图像预处理
transform = transforms.Compose([
    # transforms.Resize((256, 128)),  # 调整图片大小
    transforms.Resize((336, 168), interpolation=InterpolationMode.BICUBIC),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # 图片转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

#  加载目标集的所有图像并获取特征向量列表
# gallery_dataset = ImageFolder("CUHK03_labled/datasets/bounding_box_test", transform=transform)
gallery_dataset = ImageFolder("bounding_box_test", transform=transform)
gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False)

#获取gallery图像摄像头列表
camera_ids_gallery = []
for image_path, _ in gallery_dataset.imgs:
    filename = os.path.basename(image_path)  # 获取文件名
    # match = re.search(r'_c(\d+)', filename)
    camera_id = re.search(r'_c(\d+)', filename).group(1)
    camera_id= int(camera_id)
    camera_ids_gallery.append(camera_id)
# print(camera_ids_gallery)


#准备将目标集通过模型转换成目标向量
gallery_vectors = []
features = torch.FloatTensor()
label_list=[]#标签列表
number=0#处理数据可视化目的
for i, (images, labels) in enumerate(gallery_loader):
    images = images.to('cuda')
    labels = labels.to('cuda')
    #抄的
    n, c, h, w = images.size()
    print(images.size())
    ff = torch.FloatTensor(n, 2048, part).zero_()
    with torch.no_grad():
        output, outputs,output1 = model(images)
    f1 = output.data.cpu()
    for j in range(part):
        f = outputs[j].data.cpu()
        ff[:, :, j] = torch.cat((f1, f), 1)
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(part)
    ff = ff.div(fnorm.expand_as(ff))
    ff = ff.view(ff.size(0), -1)

    # f1 = output.data.cpu()
    # for j in range(part):
    #     f = outputs[j].data.cpu()
    #     ff = torch.cat((f1, f), 1)

    # #加权平均
    # f1 = output.data.cpu()
    # for j in range(part):
    #     f = outputs[j].data.cpu()
    #     ff = f1+f


    features = torch.cat((features, ff), 0)
    # gallery_vectors.extend(target_vectors.tolist())#先转化成列表再放入gallery_vectors
    gallery_vectors = features
    label_list.extend(labels)#标签列表取出类名称
    number += 1
    print(number,' is done')
# gallery_vectors = torch.tensor(gallery_vectors).to('cuda')
gallery_vectors_init = gallery_vectors.clone().detach().to('cuda')
label_list_gallery = [gallery_dataset.classes[item.item()] for item in label_list]

# print(label_list_gallery)
# print(gallery_vectors.shape)
#----
#循环读取查询集，挺好的，处理完一个再处理查询集
# query_dataset = ImageFolder("CUHK03_labled/datasets/query", transform=transform)
query_dataset = ImageFolder("query", transform=transform)
query_loader = DataLoader(query_dataset, batch_size=1, shuffle=False)


#这个是计算rank1,rank5,rank10,mAP的
rank1=0
rank5=0
rank10=0
total=0


#获取query图像摄像头列表
camera_ids_query = []
for image_path, _ in query_dataset.imgs:
    filename = os.path.basename(image_path)  # 获取文件名
    camera_id = re.search(r'_c(\d+)', filename).group(1)
    camera_id=int(camera_id)
    camera_ids_query.append(camera_id)
# print(camera_ids_query)

aps = []
for i, (image, label) in enumerate(query_loader):
    print(image.size())
    image = image.to('cuda')
    # label = label.to('cuda')
    label = query_dataset.classes[label]
    with torch.no_grad():
        query, querys,query1 = model(image)

    # #加权平均
    # f1 = query.data.cpu()
    # for j in range(part):
    #     f = querys[j].data.cpu()
    #     ff = f1 + f

    # #串联
    # f1 = query.data.cpu()
    # for j in range(part):
    #     f = querys[j].data.cpu()
    #     ff = torch.cat((f1, f), 1)

    # 全局-局部
    ff = torch.FloatTensor(1, 2048, part).zero_()
    f1 = query.data.cpu()
    # ff[:, :, 0] = ff[:, :, 0] + f
    for j in range(part):
        f = querys[j].data.cpu()
        # ff[:, :, j+1] = ff[:, :, j+1] + f
        ff[:, :, j] = torch.cat((f1, f), 1)
    # g = query_vector[part].data.cpu()
    # ff[:, :, part] = ff[:, :, part] + g
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(part)
    ff = ff.div(fnorm.expand_as(ff))
    ff = ff.view(ff.size(0), -1)


    query_vector = ff
    ff = ff.to('cuda')


    query_vector = query_vector.cpu()
    query_vector = query_vector.numpy()
    gallery_vectors = gallery_vectors.cpu()
    feature_vectors = gallery_vectors.numpy()
    camera_id=camera_ids_query[i]
    # print(camera_id)
    # print('看看是字符串还是数字', type(camera_id))
    # good index
    query_index = np.argwhere(np.array(label_list_gallery) == label)
    #-1为任务标签的可以排除
    negative_index0 = np.argwhere(np.array(label_list_gallery) == '0000')
    negative_index1 = np.argwhere(np.array(label_list_gallery) == '1')
    # print(query_index)
    camera_index = np.argwhere(np.array(camera_ids_gallery) == camera_id)
    # print(camera_index)
    intersection = np.intersect1d(query_index, camera_index)
    print(intersection)

    # # 计算相似度，使用的是余弦相似度，每一个query与gallery的向量矩阵进行计算
    similarities = np.dot(query_vector, feature_vectors.T) / (
                np.linalg.norm(query_vector) * np.linalg.norm(feature_vectors, axis=1))
    # 将相似度列表中与交集索引对应的值替换为负无穷小
    np.put(similarities, intersection, -2)
    #np.put(similarities, negative_index0, -2)
    np.put(similarities, negative_index1, -2)


    # 检查similarities中是否有NaN值
    if np.isnan(np.sum(similarities)):
        # 将NaN值替换为-2
        similarities[np.isnan(similarities)] = -2
    #
    similarities = similarities.ravel().tolist()#转换成一维列表



    # 实现mAp功能
    # 将与label相等的元素设为1，其余元素设为0
    y_true = np.array(label_list_gallery) == label
    np.put(y_true, intersection, 0)
    y_true = y_true.astype(int)
    from sklearn.metrics import average_precision_score

    # 计算 Average Precision
    ap = average_precision_score(y_true, similarities)
    aps.append(ap)

    #字典记住每个序列对应的相似度值
    sim_dict = {index: value for index, value in enumerate(similarities)}
    #降序取前10个进行计算
    sorted_keys = sorted(sim_dict, key=sim_dict.get, reverse=True)[:10]
    one_zero_list=[]
    #遍历制作10个数的列表，1表示匹配上了，0表示没匹配上
    for i in sorted_keys:
        match_index = label_list[i]
        match_label = gallery_dataset.classes[match_index]
        if match_label==label:
            one_zero_list.append(1)
        else:
            one_zero_list.append(0)
    if one_zero_list[0] == 1:
        rank1 += 1
    if 1 in one_zero_list[:5]:
        rank5 += 1
    if 1 in one_zero_list[:10]:
        rank10 += 1
    total+=1
    print(total, ' is done')
mAP = np.mean(aps)
rank1_rate=rank1/total
rank5_rate=rank5/total
rank10_rate=rank10/total
print("行人重识别rank1准确率：", rank1_rate)
print("行人重识别rank5准确率：", rank5_rate)
print("行人重识别rank10准确率：", rank10_rate)
print('mAP:',mAP)



