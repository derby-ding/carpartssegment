###简化版seg代码，包括encode decode，用于理解语义分割
####本代码仅支持图像+seg图像形式的数据集。
import argparse
from dataloaders import ADE20K, ADE20K_AG, ADE20K_AGB
from utils.losses import CrossEntropyLoss2d, FocalLoss, LovaszSoftmax, SCELoss, GCELoss, symmetric_focal_loss, AUELoss, region_wei_loss, NCELoss, NCEandAUE, NCEandRCE
# from models import PSPNet
import models
from torch.utils.data import DataLoader
from trainer import Trainer
from utils import Logger
from models.resnet import resnet50, resnet101
import cv2
parser = argparse.ArgumentParser()
parser.add_argument('--arch', default="PSPNet")
parser.add_argument('--n_gpu', default=1, type=int)
parser.add_argument('--classes', default=14, type=int)

parser.add_argument('--loss', default='CE')
parser.add_argument('--optimizer', default="SGD")
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

parser.add_argument('--data_dir', default="../datasets/CarPartsSegment")
parser.add_argument('--background', action='store_true', help='if background is used for guided loss')
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--backbone_path', default='pretrained/resnet50s-a75c83cf.pth')
parser.add_argument('--base_size', default=512, type=int)
parser.add_argument('--crop_size', default=480, type=int)

parser.add_argument('--augment', default='10,15')
parser.add_argument('--scale_max', default=1.2)
parser.add_argument('--scale_min', default=0.8)
parser.add_argument('--blur', action='store_true', help='blur noise')
parser.add_argument('--color', action='store_true', help='color noise')
parser.add_argument('--split', default="train")
parser.add_argument('--num_workers', default=1, type=int)

parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--save_dir', default="saved/")
parser.add_argument('--save_period', default=50)
parser.add_argument('--monitor', default="max Mean_IoU")
parser.add_argument('--early_stop', default=50, type=int)

parser.add_argument('--tensorboard', action='store_true', help='summery writer')
parser.add_argument('--log_dir', default="saved/runs")
parser.add_argument('--log_per_iter', default=20, type=int)
parser.add_argument('--val', default=True)
parser.add_argument('--val_per_epochs', default=10, type=int)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()
def main(args):
    print('check args ', args)
    # args.data_dir = '../datasets/CarPartsSegment'
    test_loader = ADE20K_AGB(args.data_dir, 1, 'valid', shuffle=False, num_workers=1)  ##批大小为1，支持图像大小不均
    # for img, labim in test_loader:
    #     print(type(labim[:, :, 1]))
    #     # cv2.imshow(labim[:, :, 1])
    #     # cv2.waitKey(1000)
    #     # cv2.destroyAllWindows()
    # exit()
    # train_loader = ADE20K(args.data_dir, args.batch_size, 'train', crop_size=args.crop_size, base_size=args.base_size, shuffle=True, augment=args.augment)
    # test_loader = ADE20K(args.data_dir, 1, 'valid', crop_size=args.crop_size, shuffle=False, augment=False)###image size different set to batch to 1
    # if not args.background:
        # train_loader = ADE20K_AG(args.data_dir, args.batch_size, 'train', crop_size=args.crop_size,
        #                           base_size=args.base_size, scale_min=args.scale_min, scale_max=args.scale_max,
        #                           shuffle=True, color=args.color, blur=args.blur, augment=args.augment)
        # test_loader = ADE20K_AG(args.data_dir, 1, 'valid', shuffle=False)  ###image size different set to batch to 1
    from dataloaders.ade20kaug import ADE20K_aug
    kwargs = {'root': args.data_dir, 'filename': 'train',###train or val，图像子集名
        'augment': args.augment, 'crop_size': args.crop_size, 'base_size': args.base_size, 'scale_min': args.scale_min, 'scale_max': args.scale_max, 'blur': args.blur, 'color': args.color,}
    dataset = ADE20K_aug(**kwargs)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, prefetch_factor=2)
    kwargs['filename'] = 'valid'
    testd = ADE20K_aug(**kwargs)
    test_loader = DataLoader(testd, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # else:
    #     train_loader = ADE20K_AGB(args.data_dir, args.batch_size, 'train', crop_size=args.crop_size, base_size=args.base_size, scale_min=args.scale_min, scale_max=args.scale_max, shuffle=True,
    #                              color=args.color, blur=args.blur, augment=args.augment)
    #     test_loader = ADE20K_AGB(args.data_dir, 1, 'valid', shuffle=False)  ##批大小为1，支持图像大小不均

    if args.tensorboard:
        train_logger = Logger()
    else:
        train_logger = None
    ###change for new backbone
    encoder_model = resnet50(pretrained=args.resume, root=args.backbone_path)
    model = getattr(models, args.arch)(args.classes, encoder_model, args)

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


if __name__ == "__main__":
    main(args)