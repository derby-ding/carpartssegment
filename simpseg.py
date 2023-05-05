###简化版seg代码，包括encode decode，用于理解语义分割
####本代码仅支持图像+seg图像形式的数据集。
import argparse
from dataloaders import ADE20K_AG, ADE20K_AGB
from utils.losses import CrossEntropyLoss2d, FocalLoss, LovaszSoftmax, SCELoss, GCELoss, symmetric_focal_loss, AUELoss, region_wei_loss, NCELoss, NCEandAUE, NCEandRCE
import models
from trainer import Trainer##, FusTrainer
from utils import Logger
#from core.model import WeTr
from models.resnet import resnet50, resnet101
import os, datetime
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from PIL import Image
import pandas as pd
from torchvision import transforms
from torchvision.utils import make_grid
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--arch', default="PSPNet")
parser.add_argument('--n_gpu', default=1, type=int)
parser.add_argument('--classes', default=14, type=int)
parser.add_argument('--num_cats', default=8, type=int, help='car types and camera scope ')
parser.add_argument('--loss', default='ce')
parser.add_argument('--optimizer', default="AdamW")
parser.add_argument('--differential_lr', default=True, type=bool)
parser.add_argument('--lr', default=0.08, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--momentum', default=0.95)
parser.add_argument('--ignore_index', default=-1)
parser.add_argument('--lr_scheduler', default="OneCycle", help='Poly and OneCycle')
parser.add_argument('--fix_seed', action='store_true', help='fix seed to reproduce precision')###虽然store true不传则为false，传则true生效

parser.add_argument('--use_synch_bn', default=False)###multi gpu
parser.add_argument('--backbone', default="resnet50")
parser.add_argument('--freeze_bn', default=False)
parser.add_argument('--freeze_backbone', default=False)

parser.add_argument('--addhsv', action='store_true', help='if hsv space is used for augmentation ')##后融合
parser.add_argument('--data_dir', default="../datasets/CarPartsSegment")#CarPartsSegment_standard##标准集
parser.add_argument('--background', action='store_true', help='if background is used for guided loss')
parser.add_argument('--batch_size', default=5, type=int)
parser.add_argument('--backbone_path', default='pretrained/resnet50s-a75c83cf.pth')
parser.add_argument('--base_size', default=512, type=int)
parser.add_argument('--crop_size', default=480, type=int)

parser.add_argument('--augment', default='6,15')
parser.add_argument('--scale_max', default=1.5)
parser.add_argument('--scale_min', default=0.5)
parser.add_argument('--blur', action='store_true', help='blur noise')
parser.add_argument('--color', action='store_true', help='color noise')
parser.add_argument('--split', default="train")
parser.add_argument('--num_workers', default=1, type=int)

parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--save_dir', default="saved/")
parser.add_argument('--save_period', default=50)
parser.add_argument('--monitor', default="max Mean_IoU")
parser.add_argument('--early_stop', default=100, type=int)

parser.add_argument('--tensorboard', action='store_true', help='summery writer')
parser.add_argument('--log_dir', default="saved/runs")
parser.add_argument('--log_per_iter', default=20, type=int)
parser.add_argument('--val', default=True)
parser.add_argument('--val_per_epochs', default=10, type=int)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

from utils import transforms as local_transforms
def multi_scale_predict(model, image, scales, num_classes, device, flip=False):
    input_size = (image.size(2), image.size(3))
    upsample = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
    total_predictions = np.zeros((num_classes, image.size(2), image.size(3)))

    image = image.data.data.cpu().numpy()
    # print('check image shape', image.shape)
    for scale in scales:
        scaled_img = ndimage.zoom(image, (1.0, 1.0, float(scale), float(scale)), order=1, prefilter=False)
        scaled_img = torch.from_numpy(scaled_img).to(device)
        # scaled_prediction = upsample(model(scaled_img).cpu())
        scot = model(scaled_img)[0]###[1*class*w*h], w = oriw*scale, h = orih*scale

        scaled_prediction = upsample(scot)###[1*class*oriw*orih]
        # print('check scot size ', scot.size(), scaled_prediction.size())
        if flip:
            fliped_img = scaled_img.flip(-1).to(device)
            fliped_predictions = upsample(model(fliped_img).cpu())
            scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)###class*oriw*orih

    total_predictions /= len(scales)
    # print('check total predict shape ', total_predictions.shape)
    return total_predictions

from skimage import exposure, filters
def equalize(x):
    ###input numpy, output torch tensor
    # out = x.clone()
    # out = out.detach().cpu().squeeze().numpy()

    # out = filters.unsharp_mask(x, 3, 1.0)###sharp
    # low = 0.10  # Pixels with intensity smaller than this will be black
    # high = 0.90  # Pixels with intensity larger than this will be white
    # out = exposure.rescale_intensity(x, in_range=(low, high))
    out = exposure.equalize_hist(x)
    # out = exposure.equalize_adapthist(x, clip_limit=0.01)

    return out
def save_compare(image, hist_img, target, pred0, pred1, loader, output_path, palette):
    # Saves the image, the model output and the results after the post processing
    val_img = []
    restore_transform = transforms.Compose([
        local_transforms.DeNormalize(loader.dataset.MEAN, loader.dataset.STD),
        transforms.ToPILImage()])
    viz_transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor()])

    image = image[0].data.cpu()
    image = restore_transform(image)
    hist_img = hist_img[0].data.cpu()
    hist_img = restore_transform(hist_img)
    d, t, o = image.convert('RGB'), image.convert('RGB'), hist_img.convert('RGB')
    [d, t, o] = [viz_transform(x) for x in [d, t, o]]
    val_img.extend([d, t, o])

    target = target.cpu().numpy()[0]
    # print('check pred size ', pred0.size(), pred1.size(), 'image shape ', image.shape)
    pred0 = pred0.data.max(1)[1]
    # print('check pred size ', pred0.size())
    pred0 = pred0.cpu().numpy()[0]  ###to 2d mask, w*h
    pred1 = pred1.data.max(1)[1].cpu().numpy()[0]

    d = colorize_mask(target, palette)  ##restore_transform(image)
    t, o = colorize_mask(pred0, palette), colorize_mask(pred1, palette)
    d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
    [d, t, o] = [viz_transform(x) for x in [d, t, o]]
    val_img.extend([d, t, o])

    val_img = torch.stack(val_img, 0)
    val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
    ndarr = val_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(output_path)
####不稳定性曲线
def vis_unstable(loader, args):
    num_classes = args.classes
    palette = loader.dataset.palette
    image_files = loader.dataset.files
    restore_transform = transforms.Compose([
        local_transforms.DeNormalize(loader.dataset.MEAN, loader.dataset.STD),
        transforms.ToPILImage()])
    input_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(loader.dataset.MEAN, loader.dataset.STD)])
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

    encmodel = resnet50()
    model = getattr(models, 'PSPNet')(num_classes, encmodel, args)
    model_path = 'saved/PSPNet/'+args.loss+'best_model.pth'
    checkpoint = torch.load(model_path, map_location=device)# Load checkpoint
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']

    model = torch.nn.DataParallel(model)# load#during training, we used data parallel
    model.load_state_dict(checkpoint)

    model.eval()
    opals = []
    oiouls = []
    mpals = []###multi scale
    miouls = []
    hpals = []###hist equal
    hiouls = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            ####data取值为[1.0,-1.0],不同于cv图像取值
            target = target.to(device)
            if target.size()[-1] == 3:
                target, backgrd = target[:, :, :, 0].clone(), target[:, :, :, 1].clone()
                backgrd[backgrd == 255] = 1  ###binary image, 防止loss过大
            input_size = (data.size(2), data.size(3))
            upsample = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            ###原pixelacc miou
            prediction0 = model(data)###[0]  ###
            # print('check szie ', prediction0.size(), target.size())
            correct, labeled, inter, union0 = eval_metrics(prediction0, target, num_classes)####ignore background
            pixAcc0 = 1.0 * correct / (np.spacing(1) + labeled)
            IoU = 1.0 * inter / (np.spacing(1) + union0)
            mIoU0 = IoU.mean()
            ####multiscale pixelacc miou
            image = data.cpu().numpy()
            scaled_img = ndimage.zoom(image, (1.0, 1.0, float(1.5), float(1.5)), order=1, prefilter=False)
            new_img = torch.from_numpy(scaled_img).to(device)
            prediction1 = upsample(model(new_img).cpu()).cuda()
            print('check szie ', prediction1.size())
            correct, labeled, inter, union1 = eval_metrics(prediction1, target, num_classes)
            pixAcc1 = 1.0 * correct / (np.spacing(1) + labeled)
            IoU = 1.0 * inter / (np.spacing(1) + union1)
            mIoU1 = IoU.mean()
            ###白平衡 pixelacc miou
            out = data.clone()  ###避免覆盖
            out = np.array(restore_transform(out.squeeze())) / 255.0  ###原图
            # print('check image max ', out.max(), out.min())
            new_img = equalize(out)  ###直方图均衡、其他预处理
            new_img = input_transform(new_img).float()  ###转normaled image
            new_img = new_img.unsqueeze(dim=0).cuda()  ##转tensor

            prediction1 = model(new_img)
            correct, labeled, inter, union1 = eval_metrics(prediction1, target, num_classes)
            pixAcc2 = 1.0 * correct / (np.spacing(1) + labeled)
            IoU = 1.0 * inter / (np.spacing(1) + union1)
            mIoU2 = IoU.mean()

            opals.append(pixAcc0)
            oiouls.append(mIoU0)
            mpals.append(pixAcc1)  ###multi scale
            miouls.append(mIoU1)
            hpals.append(pixAcc2)  ###hist equal
            hiouls.append(mIoU2)

            mpae = [x - y for x, y in zip(opals, mpals)]
            mioue = [x - y for x,y in zip(oiouls, miouls)]
            hpae = [x - y for x, y in zip(opals, hpals)]
            hioue = [x - y for x, y in zip(oiouls, hiouls)]

            ###plt/save
            save_time = datetime.datetime.now().strftime('%m-%d')
            df = pd.DataFrame()
            df['mpae'] = mpae
            df['mioue'] = mioue
            df['hpae'] = hpae
            df['hioue'] = hioue
            df.to_csv(args.loss + save_time + 'pae.csv', index=False)
###画出对比曲线
import seaborn as sns
import matplotlib.pyplot as plt
from utils.palette import ADE20K_palette
from matplotlib import patches as mpatches
def drow_lin(ori_path, com_path, outname='cesce'):
    df = pd.read_csv(ori_path)
    scale_s = 50.0
    orimpae = sorted(df['mpae']*scale_s)
    orimioue = sorted(df['mioue']*scale_s)
    orihpae = sorted(df['hpae']*scale_s)
    orihioue = sorted(df['hioue']*scale_s)

    df1 = pd.read_csv(com_path)
    commpae = sorted(df1['mpae']*scale_s)
    commioue = sorted(df1['mioue']*scale_s)
    comhpae = sorted(df1['hpae']*scale_s)
    comhioue = sorted(df1['hioue']*scale_s)
    # sorted(orimpae), sorted(orimioue), sorted(orihpae), sorted(orihioue), sorted(commpae), sorted(commioue), sorted(comhpae), sorted(comhioue)
    palette = ADE20K_palette
    palette = list(np.array(palette) / 255.0)
    colormap = []
    for i in range(8):
        colormap.append(tuple(palette[i * 3:i * 3 + 3]))
    X = range(len(orimpae))
    # fig,(ax0, ax1, ax2, ax3) = plt.subplots(2, 2)
    fig = plt.figure(figsize=(12, 8))
    y0 = [0 for x in X]
    ax0 = plt.subplot(221)
    ax1 = plt.subplot(222)
    ax2 = plt.subplot(223)
    ax3 = plt.subplot(224)
    ax0.plot(X, orimpae, label='orimpae')
    ax0.plot(X, commpae, label='commpae')
    ax0.plot(X, y0, label='commpae')
    ax0.legend(loc="upper left", bbox_to_anchor=[0, 1], shadow=True)
    ax0.set_ylim((-5, 5))

    ax1.plot(X, orihpae, label='orihpae')
    ax1.plot(X, comhpae, label='comhpae')
    ax1.legend(loc="upper left", bbox_to_anchor=[0, 1], shadow=True)
    ax1.set_ylim((-5, 5))

    ax2.plot(X, orimioue, label='orimioue')
    ax2.plot(X, commioue, label='commioue')
    ax2.legend(loc="upper left", bbox_to_anchor=[0, 1], shadow=True)
    ax2.set_ylim((-5, 5))

    ax3.plot(X, orihioue, label='orihioue')
    ax3.plot(X, comhioue, label='comhioue')
    ax3.legend(loc="upper left", bbox_to_anchor=[0, 1], shadow=True)
    ax3.set_ylim((-5, 5))
    # sns.lineplot(x=X, y=orimpae, color=colormap[0])
    # sns.lineplot(x=X, y=orimioue, color=colormap[1])
    # sns.lineplot(x=X, y=orihpae, color=colormap[2])
    # sns.lineplot(x=X, y=orihioue, color=colormap[3])
    # sns.lineplot(x=X, y=commpae, color=colormap[4])
    # sns.lineplot(x=X, y=commioue, color=colormap[5])
    # sns.lineplot(x=X, y=comhpae, color=colormap[6])
    # sns.lineplot(x=X, y=comhioue, color=colormap[7])
    # plt.legend(handles=[mpatches.Patch(color=c) for c in colormap],
    #            labels=['orimpae', 'orimioue', 'orihpae', 'orihioue', 'commpae', 'commioue', 'comhpae', 'comhioue'])
    # plt.xlabel('image_indx')
    # plt.ylabel('errors')

    plt.savefig(outname+'.png')
    plt.close()  ###关闭当前

def main(args):
    # args.addhsv = True
    if args.tensorboard:
        train_logger = Logger()
    else:
        train_logger = None
    if args.backbone == 'resnet50':
        encmodel = resnet50(pretrained=True, root=args.backbone_path)
    else:
        encmodel = resnet50()###从scratch开始
    # model = getattr(models, 'PSPNet')(args.classes, encmodel, args)
    # encmodel = resnet50()
    model = getattr(models, 'PSPNet')(args.classes, encmodel, args)
    print('check args ', args)
    train_loader = ADE20K_AGB(args.data_dir, args.batch_size, 'train', crop_size=args.crop_size,
                             base_size=args.base_size, scale_min=args.scale_min, scale_max=args.scale_max,
                             shuffle=True, color=args.color, blur=args.blur, augment=args.augment)
    test_loader = ADE20K_AGB(args.data_dir, 1, 'valid', shuffle=False)  ###image size different set to batch to 1
    if args.loss in ['Lovsz', 'lovsz']:
        loss = LovaszSoftmax()
    elif args.loss in ['nrce', 'NRCE']:
        loss = NCEandRCE()
    elif args.loss in ['nau', 'NAU']:
        loss = NCEandAUE()
    elif args.loss in ['gce', 'GCE']:
        loss = GCELoss()
    elif args.loss in ['Focal', 'focal']:
        loss = FocalLoss()##gamma幂指数 alpha系数
    elif args.loss in ['sce', 'SCE']:
        loss = SCELoss(0.8, 0.2)
    elif args.loss in ['region', 'Region']:
        loss = region_wei_loss()##1
    elif args.loss in ['sym', 'Sym']:
        loss = symmetric_focal_loss()##delta系数 gamma幂指数
    elif args.loss in ['aue', 'AUE']:
        loss = AUELoss()  ##delta系数 gamma幂指数
    else:
        loss = CrossEntropyLoss2d()  ###params in args

    trainer = Trainer(
        model=model,
        loss=loss,
        resume=args.resume,
        config=args,
        train_loader=train_loader,
        val_loader=test_loader,
        train_logger=train_logger)

    trainer.train()
    ###显示不稳定性
    # vis_unstable(test_loader, args)

if __name__ == "__main__":
    args.resume = False
    args.background = True
    # args.augment = 'None'
    main(args)
    # drow_lin('cescratch_pae.csv', 'CEpretrin_pae.csv', outname='wn_pretin')