###简化版seg代码，包括encode decode，用于理解语义分割
####本代码仅支持图像+seg图像形式的数据集。
import argparse
from dataloaders import ADE20K, ADE20K_AG, ADE20K_AGB
import models
from utils import transforms as local_transforms
from torchvision.utils import make_grid
from torchvision import transforms
from utils.helpers import colorize_mask
from models.resnet import resnet50, resnet101
import cv2, os
import torch
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default="outputs")
parser.add_argument('--backbone', default="resnet50")
parser.add_argument('--data_dir', default="../datasets/CarPartsSegment")
parser.add_argument('--backbone_path', default='pretrained/resnet50s-a75c83cf.pth')
parser.add_argument('--split', default="valid")
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--resume_path', default='saved/PSPNet/10-26/regionbest_model.pth', help='resume from checkpoint')
args = parser.parse_args()

def val_(model, data_loader, config):
    model.eval()
    restore_transform = transforms.Compose([local_transforms.DeNormalize(data_loader.dataset.MEAN, data_loader.dataset.STD),
        transforms.ToPILImage()])
    viz_transform = transforms.Compose([transforms.Resize((400, 400)), transforms.ToTensor()])
    with torch.no_grad():
        val_visual = []
        ct = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            # data, target = data.to(self.device), target.to(self.device)
            if target.size()[-1] == 3:
                target, backgrd = target[:, :, :, 0].clone(), target[:, :, :, 1].clone()
                backgrd[backgrd == 255] = 1  ###binary image, 防止loss过大

            output = model(data)
            # LIST OF IMAGE TO VIZ (15 images)
            if len(val_visual) < 10:
                target_np = target.data.cpu().numpy()
                output_np = output.data.max(1)[1].cpu().numpy()
                val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])
            else:###每10图保存
                # WRTING & VISUALIZING THE MASKS
                val_img = []
                palette = data_loader.dataset.palette
                for d, t, o in val_visual:
                    d = restore_transform(d)
                    t, o = colorize_mask(t, palette), colorize_mask(o, palette)
                    d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
                    [d, t, o] = [viz_transform(x) for x in [d, t, o]]
                    val_img.extend([d, t, o])
                val_img = torch.stack(val_img, 0)
                val_img = make_grid(val_img.cpu(), nrow=3, padding=5)

                ndarr = val_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                im = Image.fromarray(ndarr)
                im.save(os.path.join(config.outdir, config.split + str(ct) + '.png'))
                ct += 1
                val_visual = []##重置


def main(args):
    print('check args ', args)
    data_loader = ADE20K_AG(args.data_dir, 1, args.split, shuffle=False, num_workers=2)  ###image size different set to batch to 1

    ###change for new backbone
    encoder_model = resnet50(pretrained=True, root=args.backbone_path)
    model = getattr(models, 'PSPNet')(14, encoder_model, args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model, device_ids=[0])##cuda 0
    model.to(device)

    checkpoint = torch.load(args.resume_path)
    # Load last run info, the model params, the optimizer and the loggers
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['state_dict'])
    val_(model, data_loader, args)
main(args)