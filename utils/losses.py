import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.utils import class_weight 
from utils.lovasz_losses import lovasz_softmax

def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    #cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

    weights = np.ones(7)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='none'):
        super(CrossEntropyLoss2d, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # weight = torch.FloatTensor(list(0.5 / np.asarray([0.843, 0.546, 0.413, 0.386, 0.309, 0.66, 0.536, 0.463, 0.493, 0.573, 0.194, 0.27, 0.175, 0.662]))).to(
        #     self.device)
        weight = torch.FloatTensor(
            list(0.5 / np.asarray([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))).to(
            self.device)
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target, backlabel=None):
        if backlabel is not None:
            wid, hei = backlabel.size()[1:3]
            batchsz = backlabel.size()[0]
            backpercent = (backlabel==0).sum()/(wid*hei*batchsz)
            backlabel = 0.1*(backlabel==0)/backpercent+backlabel
        else:
            backlabel = 1.0
        # print('check ce type', output.dtype, target.dtype)
        # print('ce loss out shape batch numclass wid hei', output.size(), target.size())
        loss = self.CE(output, target.long())*backlabel
        return loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target, backlabel=None):
        ###背景label
        if backlabel is not None:
            wid, hei = backlabel.size()[1:3]
            batchsz = backlabel.size()[0]
            backpercent = (backlabel==0).sum()/(wid*hei*batchsz)
            backlabel = 0.1*(backlabel==0)/backpercent+backlabel
        else:
            backlabel = 1.0

        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)###batch*class*h*w
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # weight = torch.FloatTensor(list(0.5 / np.asarray(
        #     [0.843, 0.546, 0.413, 0.386, 0.309, 0.66, 0.536, 0.463, 0.493, 0.573, 0.194, 0.27, 0.175, 0.662]))).to(
        #     self.device)
        weight = torch.FloatTensor(
            list(0.5 / np.asarray([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))).to(
            self.device)
        self.CE_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index, weight=weight)

    def forward(self, output, target, backlabel=None):
        ###背景label
        if backlabel is not None:
            wid, hei = backlabel.size()[1:3]
            batchsz = backlabel.size()[0]
            backpercent = (backlabel==0).sum()/(wid*hei*batchsz)
            backlabel = 0.1*(backlabel==0)/backpercent+backlabel
        else:
            backlabel = 1.0
        # print('check focal type', output.dtype, target.dtype)
        logpt = self.CE_loss(output, target.long())*backlabel
        pt = torch.exp(-logpt)
        loss = ((1-pt)**self.gamma) * logpt
        #print('check focal loss shape ', loss.shape)##batch*h*w
        if self.size_average:
            return loss.mean()
        return loss.sum()

class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='none', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
    
    def forward(self, output, target, backlabel=None):
        ###背景label
        if backlabel is not None:
            wid, hei = backlabel.size()[1:3]
            batchsz = backlabel.size()[0]
            backpercent = (backlabel==0).sum()/(wid*hei*batchsz)
            backlabel = 0.1*(backlabel==0)/backpercent+backlabel
        else:
            backlabel = 1.0

        CE_loss = self.cross_entropy(output, target.long())*backlabel
        dice_loss = self.dice(output, target, backlabel)
        return CE_loss.mean() + dice_loss

class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index
    
    def forward(self, output, target, backlabel=None):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss

class NCELoss(nn.Module):
    def __init__(self, scale=1.0):
        super(NCELoss, self).__init__()
        self.scale = scale

    def forward(self, pred, y_true, backlabel=None):
        ###消除错误赋值
        if (y_true == 255).sum() > 0:
            y_true[y_true == 255] = y_true.min()
        ###背景label
        if backlabel is not None:
            wid, hei = backlabel.size()[1:3]
            batchsz = backlabel.size()[0]
            backpercent = (backlabel==0).sum()/(wid*hei*batchsz)
            backlabel = 0.1*(backlabel==0)/backpercent+backlabel
        else:
            backlabel = 1.0
        # pred = F.log_softmax(pred, dim=1)
        pred = F.softmax(pred, dim=1)
        y_true = y_true.long()
        # label_one_hot = F.one_hot(y_true, self.num_classes).float().to(pred.device)
        label_one_hot = make_one_hot(y_true.unsqueeze(dim=1), classes=pred.size()[1])
        loss = -1 * torch.sum(label_one_hot * pred, dim=1) / (-pred.sum(dim=1))
        return self.scale * loss.mean()

class NMAE(nn.Module):
    def __init__(self, scale=1.0):
        super(NMAE, self).__init__()
        self.scale = scale

    def forward(self, pred, y_true, backlabel=None):
        ###消除错误赋值
        if (y_true == 255).sum() > 0:
            y_true[y_true == 255] = y_true.min()
        ###背景label
        if backlabel is not None:
            wid, hei = backlabel.size()[1:3]
            batchsz = backlabel.size()[0]
            backpercent = (backlabel==0).sum()/(wid*hei*batchsz)
            backlabel = 0.1*(backlabel==0)/backpercent+backlabel
        else:
            backlabel = 1.0
        pred = F.softmax(pred, dim=1)
        y_true = y_true.long()
        # label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = make_one_hot(y_true.unsqueeze(dim=1), classes=pred.size()[1])
        norm = 1 / (pred.size()[1] - 1)
        loss = 1. - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * norm * loss.mean()

###symmetric ce loss
class SCELoss(nn.Module):
    def __init__(self, alpha, beta):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        # weight = torch.FloatTensor(list(0.5/np.asarray([0.843,0.546,0.413,0.386,0.309,0.66,0.536,0.463,0.493,0.573,0.194,0.27,0.175,0.662]))).to(self.device)
        weight = torch.FloatTensor(
            list(0.5 / np.asarray([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))).to(
            self.device)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none', weight=weight)

    def forward(self, pred, y_true, backlabel=None):
        num_class = pred.size()[1]
        ###消除错误赋值
        if (y_true == 255).sum() > 0:
            y_true[y_true == 255] = y_true.min()

        ###背景label
        if backlabel is not None:
            wid, hei = backlabel.size()[1:3]
            batchsz = backlabel.size()[0]
            backpercent = (backlabel==0).sum()/(wid*hei*batchsz)
            backlabel = 0.1*(backlabel==0)/backpercent+backlabel
        else:
            backlabel = 1.0

        # CCE
        ce = self.cross_entropy(pred, y_true.long())

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        y_true = y_true.long()
        # print('sce loss pred size after clamp ', pred.dtype, y_true.dtype)
        label_one_hot = make_one_hot(y_true.unsqueeze(dim=1), classes=num_class)
        # label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * (backlabel*ce).mean() + self.beta * (backlabel*rce).mean()
        return loss
###General ce, smooth is between 0-1, 1=MAE 0=ce
class GCELoss(nn.Module):
    def __init__(self, smooth=.4, ignore_index=255):
        super(GCELoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target, backlabel=None):
        ###消除错误赋值
        if (target == 255).sum() > 0:
            target[target == 255] = target.min()
        ###背景label
        if backlabel is not None:
            wid, hei = backlabel.size()[1:3]
            batchsz = backlabel.size()[0]
            backpercent = (backlabel == 0).sum() / (wid * hei * batchsz)
            backlabel = 0.1 * (backlabel == 0) / backpercent + backlabel
        else:
            backlabel = 1.0

        target = target.long()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)###batch*class*h*w
        # output_flat = output.contiguous().view(-1)
        # target_flat = target.contiguous().view(-1)
        intersection = torch.sum(output * target, dim=1) * backlabel
        loss = (1 - intersection**self.smooth)/self.smooth
        return loss.mean()
###常规方法
class MAELoss(nn.Module):
    def __init__(self, scale=2.0):
        super(MAELoss, self).__init__()
        # self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, y_true, backlabel=None):
        ###消除错误赋值
        if (y_true == 255).sum() > 0:
            y_true[y_true == 255] = y_true.min()
        ###背景label
        if backlabel is not None:
            wid, hei = backlabel.size()[1:3]
            batchsz = backlabel.size()[0]
            backpercent = (backlabel==0).sum()/(wid*hei*batchsz)
            backlabel = 0.1*(backlabel==0)/backpercent+backlabel
        else:
            backlabel = 1.0
        pred = F.softmax(pred, dim=1)
        label_one_hot = make_one_hot(y_true.unsqueeze(dim=1), classes=pred.size()[1])###F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = 1. - torch.sum(label_one_hot * pred, dim=1)*backlabel
        return self.scale * loss.mean()

###asymetric unhinging loss
class AUELoss(nn.Module):
    ''' 参考https://zhuanlan.zhihu.com/p/420913134'''
    def __init__(self, a=1.5, q=0.9, eps=1e-7, scale=1.0):
        super(AUELoss, self).__init__()
        # self.num_classes = num_classes
        self.a = a
        self.q = q
        self.eps = eps
        self.scale = scale

    def forward(self, pred, y_true, backlabel=None):
        ###消除错误赋值
        if (y_true == 255).sum() > 0:
            y_true[y_true == 255] = y_true.min()
        ###背景label
        if backlabel is not None:
            wid, hei = backlabel.size()[1:3]
            batchsz = backlabel.size()[0]
            backpercent = (backlabel==0).sum()/(wid*hei*batchsz)
            backlabel = 0.1*(backlabel==0)/backpercent+backlabel
        else:
            backlabel = 1.0

        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        y_true = y_true.long()
        label_one_hot = make_one_hot(y_true.unsqueeze(dim=1), classes=pred.size()[1])#F.one_hot(labels, pred.size()[1]).float().to(pred.device)
        loss = (torch.pow(self.a - torch.sum(label_one_hot * pred, dim=1), self.q) - (self.a - 1) ** self.q) / self.q
        # loss = (torch.pow(self.a - torch.sum(label_one_hot * pred, dim=1)*backlabel, self.q) - (self.a-1)**self.q)/ self.q
        return loss.mean() * self.scale

class symmetric_focal_loss(nn.Module):
    """
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    """
    def __init__(self, delta=0.7, gamma=2.):
        super(symmetric_focal_loss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.cross_entropy = nn.CrossEntropyLoss()
    def forward(self, y_pred, y_true, backlabel=None):
        # axis = identify_axis(y_true.get_shape())
        # epsilon = K.epsilon()
        # y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        # cross_entropy = -y_true * K.log(y_pred)
        cross_entropy = self.cross_entropy(y_pred, y_true)
        #calculate losses separately for each class
        # back_ce = K.pow(1 - y_pred[:,:,:,0], self.gamma) * cross_entropy[:,:,:,0]
        back_ce = ((1 - y_pred[:, :, :, 0])**self.gamma) * cross_entropy[:, :, :, 0]
        back_ce = (1 - self.delta) * back_ce

        fore_ce = ((1 - y_pred[:, :, :, 1])**self.gamma) * cross_entropy[:, :, :, 1]
        fore_ce = self.delta * fore_ce

        # loss = (K.sum(tf.stack([back_ce, fore_ce],axis=-1),axis=-1)).mean()
        loss = (torch.sum(torch.stack([back_ce, fore_ce], dim=-1), dim=-1)).mean()

        return loss

from sklearn.metrics import confusion_matrix
'''分区loss'''
class region_wei_loss(nn.Module):

    """
    Parameters
    ----------
    delta : float, optional
        cross entropy权重
    backlab: optional
        前景车辆，背景其他，voc物体分割结果
    """
    def __init__(self, wei_confus=None, delta=0.7, batch_wi=0.01):
        super(region_wei_loss, self).__init__()
        self.delta = delta
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # weight = torch.FloatTensor(list(0.5 / np.asarray(
        #     [0.843, 0.546, 0.413, 0.386, 0.309, 0.66, 0.536, 0.463, 0.493, 0.573, 0.194, 0.27, 0.175, 0.662]))).to(
        #     self.device)
        weight = torch.FloatTensor(list(0.5 / np.asarray([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))).to(self.device)

        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none', weight=weight)
        # self.confusionm = confusion_matrix()
        self.wei_confus = wei_confus
        self.fus_wi = batch_wi
    def forward(self, y_pred,  y_true, backlabel=None):
        num_classes = y_pred.size()[1]##batch*numcls*hei*wid
        if self.wei_confus is None:
            self.wei_confus = torch.FloatTensor(num_classes, num_classes).zero_().to(y_pred.device)
        ###消除错误赋值
        if (y_true == 255).sum() > 0:
            y_true[y_true == 255] = y_true.min()
        ###背景label
        if backlabel is not None:
            wid, hei = backlabel.size()[1:3]
            batchsz = backlabel.size()[0]
            # backpercent = (backlabel==0).sum()/(wid*hei*batchsz)###背景比例
            # backlabel = 0.1*(backlabel==0)/backpercent+backlabel###加权
            backlabel = 0.4 * (backlabel == 0) + backlabel  ###加权

        else:
            backlabel = 1.0
        print('check backlabel szie ', backlabel.size(), backlabel.mean())
        cross_entropy = self.cross_entropy(y_pred, y_true.long())*backlabel
        ##计算混淆矩阵，weight_c为num_class*num_class权值
        y_pred = F.softmax(y_pred, dim=1)
        pred = y_pred.argmax(dim=1)
        bat_con = torch.from_numpy(confusion_matrix(y_true.view(-1).cpu().numpy(), pred.view(-1).cpu().numpy(), labels=range(num_classes), normalize='pred')).to(y_pred.device)
        self.wei_confus = (self.wei_confus + bat_con*self.fus_wi)/(1+self.fus_wi)###更新wei_confus
        self.wei_confus = self.wei_confus.float()

        ###利用混淆矩阵进行柔性权值计算
        losses = y_pred.clone()  ###取相同数据量
        y_pred = y_pred.permute(0, 2, 3, 1)###类相关权值放最后维度
        # print('check dim shape compatible ', y_true.size(), y_pred.size(), self.wei_confus[:, 0].size())
        #torch.where(self.wei_confus > 0.05, self.wei_confus, 0.0)###大于一定概率，则保存，否则置零
        for i in range(num_classes):
            cha_loss = 1.0*(y_true==i) - torch.matmul(y_pred, self.wei_confus[:, i]).squeeze(-1)##按类计算loss
            # cha_loss = cha_loss(cha_loss > 0)###取正值
            cha_loss[cha_loss < 0] = 0###非i类区域置零
            losses[:, i, :, :] = cha_loss
        xloss = backlabel*torch.mean(losses, dim=1)+cross_entropy*self.delta
        print('check xloss ', xloss.size())
        loss = xloss.mean()
        # print('check loss required grad', loss)
        return loss

###compared
class NLNL(torch.nn.Module):
    def __init__(self, train_loader, num_classes=14, ln_neg=1):
        super(NLNL, self).__init__()
        self.num_classes = num_classes
        self.ln_neg = ln_neg
        weight = torch.FloatTensor(num_classes).zero_() + 1.
        if not hasattr(train_loader.dataset, 'targets'):
            weight = [1] * num_classes
            weight = torch.FloatTensor(weight)
        else:
            for i in range(num_classes):
                weight[i] = (torch.from_numpy(np.array(train_loader.dataset.targets)) == i).sum()
            weight = 1 / (weight / weight.max())
        self.weight = weight.cuda()
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight)
        self.criterion_nll = torch.nn.NLLLoss()

    def forward(self, pred, labels, backlabel=None):
        labels_neg = (labels.unsqueeze(-1).repeat(1, self.ln_neg)
                      + torch.LongTensor(len(labels), self.ln_neg).cuda().random_(1, self.num_classes)) % self.num_classes
        labels_neg = torch.autograd.Variable(labels_neg)

        assert labels_neg.max() <= self.num_classes-1
        assert labels_neg.min() >= 0
        assert (labels_neg != labels.unsqueeze(-1).repeat(1, self.ln_neg)).sum() == len(labels)*self.ln_neg

        s_neg = torch.log(torch.clamp(1. - F.softmax(pred, 1), min=1e-5, max=1.))
        s_neg *= self.weight[labels].unsqueeze(-1).expand(s_neg.size()).cuda()
        labels = labels * 0 - 100
        loss = self.criterion(pred, labels) * float((labels >= 0).sum())
        loss_neg = self.criterion_nll(s_neg.repeat(self.ln_neg, 1), labels_neg.t().contiguous().view(-1)) * float((labels_neg >= 0).sum())
        loss = ((loss+loss_neg) / (float((labels >= 0).sum())+float((labels_neg[:, 0] >= 0).sum())))
        return loss

class RCELoss(nn.Module):
    def __init__(self, scale=1.0):
        super(RCELoss, self).__init__()
        self.scale = scale

    def forward(self, pred, y_true, backlabel=None):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=0.00001, max=1.0)
        y_true = y_true.long()
        label_one_hot = make_one_hot(y_true.unsqueeze(dim=1), classes=pred.size()[1])
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        loss = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * loss.mean()
###compared
class NCEandRCE(nn.Module):
    def __init__(self, alpha=.5, beta=1.):
        super(NCEandRCE, self).__init__()
        self.nce = NCELoss(scale=alpha)
        self.rce = RCELoss(scale=beta)

    def forward(self, pred, labels, backlabel=None):
        return self.nce(pred, labels, backlabel) + self.rce(pred, labels, backlabel)
###compared
class NCEandAUE(torch.nn.Module):
    def __init__(self, alpha=.5, beta=1., a=6, q=1.5):
        super(NCEandAUE, self).__init__()
        self.nce = NCELoss(scale=alpha)
        self.aue = AUELoss(a=a, q=q, scale=beta)

    def forward(self, pred, labels, backlabel=None):
        return self.nce(pred, labels, backlabel) + self.aue(pred, labels, backlabel)