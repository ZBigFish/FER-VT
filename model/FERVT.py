import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision

from model.transformer import Transformer


# backbone + token_embedding + position_embedding
class Backbone(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def __init__(self):
        super(Backbone, self).__init__()

        resnet = torchvision.models.resnet34(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        #  feature resize networks
        # shape trans 128
        self.convtran1 = nn.Conv2d(128, 3, 21, 1)
        self.bntran1 = nn.BatchNorm2d(3)
        self.convtran2 = nn.Conv2d(256, 3, 7, 1)
        self.bntran2 = nn.BatchNorm2d(3)
        self.convtran3 = nn.Conv2d(512, 3, 2, 1,1)
        self.bntran3 = nn.BatchNorm2d(3)
        # Visual Token Embedding.
        self.layernorm = nn.LayerNorm(192)
        self.dropout = nn.Dropout(0.2)
        self.line = nn.Linear(192, 192)
        # class token init
        self.class_token = nn.Parameter(torch.zeros(1, 192))
        # position embedding
        self.pos_embedding = nn.Parameter(torch.zeros(4, 192))

        self.apply(self.weight_init)

    def forward(self, x):
        batchsize = x.shape[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)

        x = self.layer2(x)

        # L1  feature transformation from the pyramid features
        l1 = F.leaky_relu(self.bntran1(self.convtran1(x)))
        # L1 reshape to (1 x c h w)
        l1 = l1.view(batchsize,1,-1)
        # L1 token_embedding to T1    L1(1xCHW)--->T1(1xD)
        # in this model D=128
        l1 = self.line(self.dropout(F.relu(self.layernorm(l1))))

        x = self.layer3(x)
        l2 = F.leaky_relu(self.bntran2(self.convtran2(x)))
        l2 = l2.view(batchsize,1,-1)
        l2 = self.line(self.dropout(F.relu(self.layernorm(l2))))

        x = self.layer4(x)
        l3 = F.leaky_relu(self.bntran3(self.convtran3(x)))
        l3 = l3.view(batchsize,1,-1)
        l3 = self.line(self.dropout(F.relu(self.layernorm(l3))))

        x = torch.cat((l1, l2), dim=1)
        x = torch.cat((x, l3), dim=1)
        x = torch.cat((self.class_token.expand(batchsize, 1, 192), x), dim=1)
        x = x + self.pos_embedding.expand(batchsize, 4, 192)

        return x


#  refer to SubSection 3.3
# input: img(batchsize,c,h,w)--->output: img_feature_map(batchsize,c,h,w)
# in FER+ (b,3,48,48)
class GWA(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m,nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
        elif isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def __init__(self):
        super(GWA, self).__init__()
        # low level feature extraction
        self.conv1 = nn.Conv2d(3, 64, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 3, 1)
        self.bn2 = nn.BatchNorm2d(3)
        # 图像分割，每块16x16，用一个卷积层实现
        self.patch_embeddings = nn.Conv2d(in_channels=3,
                                          out_channels=9408,
                                          kernel_size=(56, 56),
                                          stride=(56, 56))
        # 使用自适应pool压缩一维
        self.aap = nn.AdaptiveAvgPool2d((1, 1))

        self.apply(self.weight_init)

    def forward(self, x):
        img = x
        batchsize = x.shape[0]
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(-1, -2).view(batchsize, 16, 3, 56, 56)  # （batchsize,9,768）（batchsize,9,3,256
        temp = []
        for i in range(x.shape[1]):
            temp.append(F.leaky_relu(self.bn2(self.conv2(
                F.leaky_relu(self.bn1(self.conv1(x[:, i, :, :, :])))))).unsqueeze(0).transpose(0, 1))

        # x = x.view(batchsize, 9, 3, 16, 16)
        # x = F.softmax(torch.matmul(x, torch.transpose(x, 3, 4)) / 3)
        x = torch.cat(tuple(temp), dim=1)
        query = x
        key = torch.transpose(query, 3, 4)
        attn = F.softmax(torch.matmul(query, key) / 56,dim=1)
        # nattn = torch.zeros(batchsize, 9, 3, 1, 1)
        temp = []
        for i in range(attn.shape[1]):
            temp.append(self.aap(attn[:, i, :, :, :]).unsqueeze(0).transpose(0, 1))
        pattn = torch.ones(56, 56).cuda() * torch.cat(tuple(temp), dim=1)
        pattn = pattn.permute(0,2,3,1,4).contiguous()
        pattn = pattn.view(batchsize, 3, 224, 224).cuda()
        map = pattn * img  # (b,3,48,48)
        return img, map


class GWA_Fusion(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, 0.1)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def __init__(self):
        super(GWA_Fusion, self).__init__()
        # 原图特征转换网络
        self.convt1 = nn.Conv2d(3, 3, (3, 3), 1, 1)
        self.bnt1 = nn.BatchNorm2d(3)
        # map特征转换网络
        self.convt2 = nn.Conv2d(3, 3, (3, 3), 1, 1)
        self.bnt2 = nn.BatchNorm2d(3)
        # RFN参与特征融合网络
        self.convrfn1 = nn.Conv2d(3,3,(3,3),1,1)
        self.bnrfn1 = nn.BatchNorm2d(3)
        self.prelu1 = nn.PReLU(3)
        self.convrfn2 = nn.Conv2d(3, 3, (3, 3), 1, 1)
        self.bnrfn2 = nn.BatchNorm2d(3)
        self.prelu2 = nn.PReLU(3)
        self.convrfn3 = nn.Conv2d(3, 3, (3, 3), 1, 1)
        self.sigmod = nn.Sigmoid()

        self.apply(self.weight_init)

    def forward(self, img, map):
        img_trans = F.relu(self.bnt1(self.convt1(img)))
        map_trans = F.relu(self.bnt2(self.convt1(map)))
        result = self.prelu1(self.bnrfn1(self.convrfn1(img_trans + map_trans)))
        result = self.prelu2(self.bnrfn2(self.convrfn2(result)))
        result = self.sigmod(self.convrfn3(result+img_trans + map_trans))

        return result


class VTA(nn.Module):
    def __init__(self):
        super(VTA, self).__init__()

        self.transformer = Transformer(num_layers=12, dim=192, num_heads=8,
                                       ff_dim=768, dropout=0.1)
        self.layernorm = nn.LayerNorm(192)
        self.fc = nn.Linear(192, 8)


    def forward(self, x):
        x = self.transformer(x)
        # x = x.transpose(1, 2)
        x = self.layernorm(x)[:, 0, :]
        x = self.fc(x)
        return x


class FERVT(nn.Module):
    def __init__(self, device):
        super(FERVT, self).__init__()
        self.gwa = GWA()
        self.gwa.to(device)
        self.gwa_f = GWA_Fusion()
        self.gwa_f.to(device)
        self.backbone = Backbone()
        self.backbone.to(device)
        self.vta = VTA()
        self.vta.to(device)

        self.to(device)
        # Evaluation mode on

    def forward(self, x):
        img,map = self.gwa(x)
        emotions = self.vta(self.backbone(self.gwa_f(img,map)))
        return emotions


# CrossEntropyLoss with Label Smoothing is added in pytorch 1.7.0+,change it will be ok if your version >1.7
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            if len(true_dist.shape) == 1:
                true_dist.scatter_(1, target.data.unsqueeze(0), self.confidence)
            else:
                true_dist.scatter_(1, target.data, self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
