#-- coding: utf-8 --
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
import torch.nn.functional as F
import numpy as np
import math

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


def weights_init_new(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
            # add_block += [nn.PReLU()]
        if dropout:
            add_block += [nn.Dropout(p=0.1)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        self.add_block = add_block

        ##### original code, if use cosmargin, comment the below lines ####
        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        feature = x
        ##### original code, if use cosmargin, comment the below lines ####
        x = self.classifier(x)
        return x, feature


#空间注意力模型
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        # (B,C,H,W)---(B,1,H,W)---(B,2,H,W)---(B,1,H,W)
        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mean_f = torch.mean(x, dim=1, keepdim=True)
        max_f, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([mean_f, max_f], dim=1)
        out = self.conv1(cat)
        return x*self.sigmoid(out)

#通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, channel, rate=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc_avg = nn.Sequential(
            nn.Conv2d(channel, channel // rate, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // rate, channel, 1, bias=False)
        )
        self.fc_max = nn.Sequential(
            nn.Conv2d(channel, channel // rate, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // rate, channel, 1, bias=False)
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg = self.avg_pool(x)
        avg_feature = self.fc_avg(avg)

        max = self.max_pool(x)
        max_feature = self.fc_max(max)

        # 计算每列的二范数
        max_feature1 = torch.norm(max_feature, dim=0, p=2)
        avg_feature1= torch.norm(avg_feature, dim=0, p=2)
        weight1 = max_feature1/(max_feature1+avg_feature1)
        weight2 = avg_feature1 / (max_feature1 + avg_feature1)

        out = weight1*max_feature + weight2*avg_feature
        out = self.sig(out)
        return x * out

class GeneralizedMeanPooling(nn.Module):
    """对由多个输入平面组成的输入信号应用二维平均功率平均自适应池化。
    计算的函数为：math：'f（X） = pow（sum（pow（X， p））， 1/p）'
        - 在 p = 无穷大时，得到最大池化
        - 在 p = 1 时，得到 Average Pooling
    输出尺寸为 H x W，适用于任何输入尺寸。
    输出要素的数量等于输入平面的数量。
    参数：
        output_size：图像的目标输出大小，形式为H x W。
                     可以是元组 （H， W） 或单个 H，表示正方形图像 H x H
                     H 和 W 可以是 ''int'' 或 ''None''，这意味着大小将
                     与输入相同。
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        """ just for infer """
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        """ """
        out = torch.clip(x, min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(out, self.output_size).pow(1. / self.p)

    def __repr__(self):
        """ """
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'


class SoftPool2D(nn.Module):
    def __init__(self, kernel_size, stride):
        super(SoftPool2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.soft_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        return x

    def soft_pool2d(self, x, kernel_size=2, stride=None):
        kernel_size = (kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        else:
            stride = (stride, stride)
        _, c, h, w = x.size()
        e_x = torch.sum(torch.exp(x), dim=1, keepdim=True)
        return F.max_pool2d(x * e_x, kernel_size, stride=stride) * (sum(kernel_size)) / (F.max_pool2d(e_x, kernel_size, stride=stride) * (sum(kernel_size)))

#我的热力图
class LIP2D(nn.Module):
    def __init__(self, kernel=3, stride=2, padding=1):
        super(LIP2D, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def forward(self, x, logit):
        weight = logit.exp()
        # 对加权特征图进行平均池化
        weighted_x = x * weight
        pooled_weighted_x = F.max_pool2d(weighted_x, self.kernel, self.stride, self.padding)

        # 对权重因子进行平均池化
        pooled_weight = F.max_pool2d(weight, self.kernel, self.stride, self.padding)

        # 计算最终的池化结果
        lip_output = pooled_weighted_x / pooled_weight

        return lip_output


# 位置注意力模块（PAM）
class PAM(nn.Module):
    def __init__(self, in_dim):
        super(PAM, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, _, height, width = x.size()
        # 计算Query
        proj_query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        # 计算Key
        proj_key = self.key_conv(x).view(batch_size, -1, height * width)
        # 计算能量（相似度）
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        # 计算Value
        proj_value = self.value_conv(x).view(batch_size, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, -1, height, width)
        out = self.gamma * out + x

        return out

class PCB(nn.Module):
    def __init__(self, class_num, part=6):
        super(PCB, self).__init__()
        self.part = part  # We cut the pool5 to 6 parts
        model_ft = models.resnet101(pretrained=True)
        self.model = model_ft
        self.avgpool1 = GeneralizedMeanPooling(norm=3)
        self.avgpool = GeneralizedMeanPooling(norm=3, output_size=(self.part, 1))#局部特征不能用这个
        self.dropout = nn.Dropout(p=0)

        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)


        #定义空间注意力模块
        self.attention = SpatialAttention()
        # self.attention2 = SpatialAttention()
        # self.attention3 = SpatialAttention()
        # self.attention4 = SpatialAttention()
        #之前的通道注意力
        self.channel_attention1 = ChannelAttention(channel=256)
        self.channel_attention2 = ChannelAttention(channel=512)
        self.channel_attention3 = ChannelAttention(channel=1024)
        self.channel_attention4 = ChannelAttention(channel=2048)


        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(2048, class_num, True, True, 1024))
        name = 'classifierAll'
        setattr(self, name, ClassBlock(2048, class_num, True, True, 1024))


        # self.model.layer2[2].conv2 = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        #     SpatialAttention(),
        #     ChannelAttention(channel=128)
        # )
        # self.model.layer3[11].conv2 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        #     SpatialAttention(),
        #     ChannelAttention(channel=256)
        # )

        self.pam_module1 = PAM(256)
        self.pam_module2 = PAM(512)
        self.pam_module3 = PAM(1024)
        self.pam_module4 = PAM(2048)
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)#局部特征来说激活函数不要
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        # x_1 = self.pam_module1(x)
        # x = x + x_1
        x = self.model.layer2(x)
        x = self.channel_attention2(x)
        x = self.attention(x)
        x_2 = self.pam_module2(x)
        x = x + x_2

        #二层卷积加空间注意力
        x = self.dropout(x)

        x = self.model.layer3(x)
        x = self.channel_attention3(x)
        x = self.attention(x)
        x_3 = self.pam_module3(x)
        x = x + x_3



        x = self.dropout(x)

        x = self.model.layer4(x)
        x = self.channel_attention4(x)
        # x_4 = self.pam_module4(x)
        # x = x + x_4

        n,_,_,_ = x.size()
        x1 = x

        x1 = self.avgpool1(x1)
        x1 = self.dropout(x1)
        x1 = torch.flatten(x1, 1)
        name = 'classifierAll'
        c1 = getattr(self, name)
        y1, f1 = c1(x1)

        x = self.avgpool(x)
        x = self.dropout(x)

        part = {}
        predict = {}
        features = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            # part[i] = x[:, :, i]
            part[i] = x[:, :, i].squeeze(dim=-1) if x[:, :, i].size(0) == 1 else x[:, :, i].squeeze()
            name = 'classifier' + str(i)
            c = getattr(self, name)
            predict[i], features[i] = c(part[i])

        y = []
        features_list = []
        for i in range(self.part):
            y.append(predict[i])
            features_list.append(features[i])
        ff = torch.FloatTensor(n, 2048, self.part).zero_()

        for j in range(self.part):
            f = features_list[j].data
            # ff[:, :, j+1] = ff[:, :, j+1] + f
            ff[:, :, j] = torch.cat((f1, f), 1)
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(self.part)
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.view(ff.size(0), -1)

        return y1, y,ff


if __name__ == '__main__':
    # 生成随机数据
    input_data = torch.randn(3,3,336,168)
    ls = PCB(751, 3)
    print(ls)
    # 将数据输入到模型中进行处理
    output,output1,output2 = ls(input_data)
    print(output.shape)
    print(output1[0].shape)
    print(output2.shape)

