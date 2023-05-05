import numpy as np
import matplotlib.pyplot as plt
# from torch.utils.data import Dataset
# from easydict import EasyDict
import torch, os
from glob import glob
from PIL import Image
import torch.nn.functional as F
import json
from torchvision import transforms
from utils.palette import ADE20K_palette
from models.resnet import resnet50
from models import PSPNet
from collections import OrderedDict
from utils.helpers import colorize_mask
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# A = torch.rand((3,4,4))
# B = torch.rand((4,4))
# print('check a b ', A*B)
# exit()
def check_seed():
    for i in range(100):
        work_seed = torch.initial_seed()
        print('check seed', work_seed)
        if work_seed > 2**32:
            work_seed = work_seed % 2**32
        print('check seed', work_seed)
        np.random.seed(work_seed)

import skimage.measure as sm

def opt_conn(lab_image):
    labim = sm.label(lab_image)
    areals = [r.area for r in sm.regionprops(labim)]
    if len(areals)>1:
        maxaa = max(areals)
        print('check opt_conn max area ', maxaa)
        for reg in sm.regionprops(labim):
            if reg.area != maxaa:###非最大连通区值为背景
                for coordinates in reg.coords:
                    lab_image[coordinates[0], coordinates[1]] = 0
    else:
        return lab_image
    return lab_image
#预处理程序，将非车体部分label赋值为0，车体赋值为其他，保存为png，训练时加权使用。
def remask(imgdir='../datasets/CarPartsSegment_fus/train/JPEGImages', outdir='../datasets/CarPartsSegment_fus/train/background2'):

    to_tensor = transforms.ToTensor()
    MEAN = [0.48897059, 0.46548275, 0.4294]
    STD = [0.22861765, 0.22948039, 0.24054667]
    normalize = transforms.Normalize(MEAN, STD)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    # Model
    encoder_model = resnet50(pretrained=False)
    config = {}
    model = PSPNet(21, encoder_model, config)
    # model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

    # Load checkpoint
    checkpoint = torch.load('../PSPnet.pth', map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']

    # If during training, we used data parallel
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        # for gpu inference, use data parallel
        if "cuda" in device.type:
            model = torch.nn.DataParallel(model)
        else:
            # for cpu inference, remove module
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]
                new_state_dict[name] = v
            checkpoint = new_state_dict
    # load
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    image_files = sorted(glob(os.path.join(imgdir, '*.jpg')))
    print('check image files', image_files[:5])
    with torch.no_grad():
        for img_file in image_files:
            sttime = datetime.now()
            image = Image.open(img_file).convert('RGB')
            input = normalize(to_tensor(image)).unsqueeze(0)

            prediction = model(input.to(device))
            prediction = prediction.squeeze(0).cpu().numpy()
            prediction = F.softmax(torch.from_numpy(prediction), dim=0).cpu().numpy()[7, :, :]###7is car
            # print('check prediction shape', prediction.shape, np.amax(prediction))
            plab = 255*(prediction > 0.3)
            # print('time ', datetime.now()-sttime)
            plab = opt_conn(np.array(plab))###二值化
            image_file = os.path.basename(img_file).split('.')[0]
            new_mask = Image.fromarray(plab.astype(np.uint8)).convert('P')
            new_mask.save(os.path.join(outdir, image_file + '.png'))
# remask()
# exit()
#####可视化增强方法
from dataloaders import ADE20K_AGB
from utils import transforms as local_transforms
from torchvision.utils import make_grid
def show_aug_image():
    # train_loader = ADE20K_AGB('../datasets/CarPartsSegment_fus/', 6, 'train', crop_size=512, base_size=480, scale_min=0.8, scale_max=1.2, shuffle=True, color=False, blur=True, augment='10,15')
    train_loader = ADE20K_AGB('../datasets/CarPartsSegment_fus/', 6, 'train', crop_size=512, base_size=480,
                              scale_min=0.8, scale_max=1.2, shuffle=True, color=False, blur=False, augment='2,15')
    # TRANSORMS FOR VISUALIZATION
    palette = train_loader.dataset.palette
    print('check palette ', palette[:9])
    MEAN = [0.48897059, 0.46548275, 0.4294]
    STD = [0.22861765, 0.22948039, 0.24054667]
    # restore_transform = transforms.Compose([local_transforms.DeNormalize(train_loader.dataset.MEAN, train_loader.dataset.STD), transforms.ToPILImage()])
    restore_transform = transforms.Compose([local_transforms.DeNormalize(MEAN, STD), transforms.ToPILImage()])
    viz_transform = transforms.Compose([transforms.Resize((400, 400)), transforms.ToTensor()])
    val_visual = []
    for batch_idx, (data, target) in enumerate(train_loader):
        # print('check load ', target.size(), data.size())
        if batch_idx < 5:  ###显示图像，验证
            if target.size()[-1]==3:
                target, backgrd = target[:, :, :, 0].clone(), target[:, :, :, 1].clone()###三通道色图

            if len(val_visual) < 10:
                target_np = target.data.cpu().numpy()
                # output_np = output.data.max(1)[1].cpu().numpy()
                val_visual.append([data[0].data.cpu(), target_np[0]])
                # print('check target_np shape', type(target_np), target_np.shape)
    val_img = []
    for d, t in val_visual:
        d = restore_transform(d)
        t = colorize_mask(t, palette)
        # m = colorize_mask(t, palette)
        d, t = d.convert('RGB'), t.convert('RGB')
        [d, t] = [viz_transform(x) for x in [d, t]]
        val_img.extend([d, t])
    val_img = make_grid(val_img, nrow=2, padding=5)
    ndarr = val_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save('trainsamples.png')

show_aug_image()
exit()
import matplotlib
print('check matplotlib', matplotlib.matplotlib_fname())
y = np.array([18.9, 73.3, 5.3, 1.2, 0.5, 0.8])
import seaborn as sns
import matplotlib.pyplot as plt
Y = [1.9, 3.3, 2.3, 1.2, 0.5, 0.8]
x = list(range(6))
sns.lineplot(x=x, y=Y)
idx = np.argmax(Y)
plt.text(idx, max(Y)+0.01, str(Y[idx]))
plt.show()
exit()
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体：解决plot不能显示中文问题
patches, ltxt = plt.pie(y,
        labels=['多车碰撞', '碰撞', '未知原因', '水淹', '玻璃破碎', '着火等'], # 设置饼图标签
        colors=['green', 'wheat', 'thistle', 'tomato', 'turquoise', 'violet'], # 设置饼图颜色
        explode=[0.0, 0.0, 0.0, 0.05, 0.1, 0.15]
       )
for l in ltxt:
    l.set_size(12)
plt.title("车损类型统计分析") # 设置标题
plt.show()
exit()

plt.rc('axes', unicode_minus='False')
label_path = '../datasets/CarPartsSegment/train/annotation/train6.jpg'
# img_src = cv2.imread()
# label = np.asarray(Image.open(label_path), dtype=np.int32) - 1

# print(set(list(np.max(label, axis=1))))
# b,g,r = cv2.split(img_src)
kwargs = {'root': '../datasets/CarPartsSegment/',
    'filename': 'split',  ###train or val，图像子集名
}

# class ADE20K_aug(Dataset):
#     def __init__(self, **kwargs):
#         print('check parse', kwargs)
#         self.root = kwargs['root']
# mm = ADE20K_aug(**kwargs)
# print(mm.root)

#显示图像
# fig, ax = plt.subplots(2, 2)
# histSize = 256
# histRange = (0, histSize)#统计的范围和histSize保持一致时可覆盖所有取值
# b_hist = cv2.calcHist([label], [0], None, [histSize], histRange)
# ax[0, 0].set_title('b hist')
# # ax[0, 0].hist(img_src.ravel(), bins=256)
# ax[0, 0].plot(b_hist)
# ax[0,1].set_title('src')
# ax[0,1].imshow(cv2.cvtColor(img_src,cv2.COLOR_BGR2RGB))
#ax[0,0].axis('off');ax[0,1].axis('off');ax[1,0].axis('off');
# ax[1,1].axis('off')#关闭坐标轴显示
# plt.show()