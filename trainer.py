import torch
import time, os
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
from utils import transforms as local_transforms
from base import BaseTrainer, DataPrefetcher
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
def rgb2yuv(x):
    '''convert batched rgb tensor to yuv'''
    out = x.clone()
    out[:, 0, :, :] = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
    out[:, 1, :, :] = -0.168736 * x[:, 0, :, :] - 0.331264 * x[:, 1, :, :] + 0.5 * x[:, 2, :, :]
    out[:, 2, :, :] = 0.5 * x[:, 0, :, :] - 0.418688 * x[:, 1, :, :] - 0.081312 * x[:, 2, :, :]
    return out
class Trainer(BaseTrainer):
    def __init__(self, model, loss, resume, config, train_loader, val_loader=None, train_logger=None, prefetch=True):
        super(Trainer, self).__init__(model, loss, resume, config, train_loader, val_loader, train_logger)
        
        self.wrt_mode, self.wrt_step = 'train_', 0
        # self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.train_loader.batch_size)))
        self.log_step = config.log_per_iter
        if config.log_per_iter: self.log_step = int(self.log_step / self.train_loader.batch_size) + 1

        self.num_classes = config.classes
        # self.lossn = config.loss
        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(self.train_loader.dataset.MEAN, self.train_loader.dataset.STD),##self.train_loader.MEAN, self.train_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])
        
        if self.device == torch.device('cpu'): prefetch = False
        if prefetch:
            self.train_loader = DataPrefetcher(train_loader, device=self.device)
            self.val_loader = DataPrefetcher(val_loader, device=self.device)

        torch.backends.cudnn.benchmark = True

    def _train_epoch(self, epoch):
        # self.logger.info('\n')

        self.model.train()
        if self.config.freeze_bn:
            if isinstance(self.model, torch.nn.DataParallel): self.model.module.freeze_bn()
            else: self.model.freeze_bn()
        self.wrt_mode = 'train'
        palette = self.train_loader.dataset.palette
        tic = time.time()
        self._reset_metrics()
        # tbar = tqdm(self.train_loader, ncols=130)
        val_visual = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.data_time.update(time.time() - tic)
            data, target = data.to(self.device), target.to(self.device)
            # print('check load train loader', target.size())
            if target.size()[-1]==3:
                target, backgrd = target[:, :, :, 0].clone(), target[:, :, :, 1].clone()###三通道色图
                backgrd[backgrd == 255] = 1###binarize image, 防止loss过大
            else:
                backgrd = None
            # print('check backgrd ', backgrd)
            self.lr_scheduler.step(epoch=epoch-1)
            if self.config.background:####外框式使用前背景
                bbkg = backgrd.unsqueeze(dim=1).repeat(1, 3, 1, 1)
                print('check size ', data.size(), backgrd.size(), bbkg.size())
                data = data*bbkg
            # LOSS & OPTIMIZE
            self.optimizer.zero_grad()
            if self.config.addhsv and self.config.resume:####稳定学习增强，废止
                data1 = rgb2yuv(data)
                output = self.model(data1)
            else:
                output = self.model(data)
            # print('check label max ', torch.max(torch.max(target[0])))
            # print('check model input output', output[0].size(), data[0].size(), target.size())
            if self.config.arch[:3] == 'PSP':
                assert output[0].size()[2:] == target.size()[1:]
                assert output[0].size()[1] == self.num_classes
                loss = self.loss(output[0], target, backgrd)
                loss = loss + self.loss(output[1], target, backgrd) * 0.4###多层loss
                output = output[0]
            else:
                assert output.size()[2:] == target.size()[1:]
                assert output.size()[1] == self.num_classes 
                loss = self.loss(output, target, backgrd)


            if isinstance(self.loss, torch.nn.DataParallel):
                loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            self.total_loss.update(loss.item())

            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # FOR EVAL
            seg_metrics = eval_metrics(output, target, self.num_classes)
            self._update_seg_metrics(*seg_metrics)
            pixAcc, mIoU, _ = self._get_seg_metrics().values()
            if batch_idx % int(len(self.train_loader) - 1) == 0 and batch_idx > 0 and epoch % 5 == 0:
                print('TRAIN ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f} | B {:.2f} D {:.2f} |'.format(
                    epoch, self.total_loss.average, pixAcc, mIoU, self.batch_time.average, self.data_time.average))
            # LOGGING & TENSORBOARD
            if self.config.tensorboard:
                if batch_idx % int(len(self.train_loader) - 1) == 0 and batch_idx > 0 and epoch % 5 == 0:
                    self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
                    self.writer.add_scalar(f'{self.wrt_mode}/loss', loss.item(), self.wrt_step)
                    # print('TRAIN ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f} | B {:.2f} D {:.2f} |'.format(
                    #     epoch, self.total_loss.average, pixAcc, mIoU, self.batch_time.average, self.data_time.average))

            # LIST OF IMAGE TO VIZ (15 images)
            if len(val_visual) < 10:###如果记录文件大，减小此值
                target_np = target.data.cpu().numpy()
                output_np = output.data.max(1)[1].cpu().numpy()
                val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

        # METRICS TO TENSORBOARD
        seg_metrics = self._get_seg_metrics()
        if self.config.tensorboard:
            for k, v in list(seg_metrics.items())[:-1]:
                self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)
            for i, opt_group in enumerate(self.optimizer.param_groups):
                self.writer.add_scalar(f'{self.wrt_mode}/Learning_rate_{i}', opt_group['lr'], self.wrt_step)
                #self.writer.add_scalar(f'{self.wrt_mode}/Momentum_{k}', opt_group['momentum'], self.wrt_step)
            # WRTING & VISUALIZING THE MASKS
        val_img = []
        for d, t, o in val_visual:
            d = self.restore_transform(d)
            t, o = colorize_mask(t, palette), colorize_mask(o, palette)
            d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
            [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
            val_img.extend([d, t, o])
        val_img = torch.stack(val_img, 0)
        val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
        ####保存图像
        if epoch > 270 and epoch % 50 == 0:
            # self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)
            ndarr = val_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            im.save(os.path.join(self.config.log_dir, str(epoch) + self.config.loss + 'train.png'))
        # RETURN LOSS & METRICS
        log = {'loss': self.total_loss.average, **seg_metrics}

        #if self.lr_scheduler is not None: self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}

        self.model.eval()
        self.wrt_mode = 'val'

        self._reset_metrics()
        # tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            val_visual = []
            for batch_idx, (data, target) in enumerate(self.val_loader):
                #data, target = data.to(self.device), target.to(self.device)

                if target.size()[-1] == 3:
                    target, backgrd = target[:, :, :, 0].clone(), target[:, :, :, 1].clone()
                    backgrd[backgrd == 255] = 1  ###binary image, 防止loss过大
                else:
                    backgrd = None
                if self.config.background:

                    data = data * backgrd
                # LOSS
                output = self.model(data)
                loss = self.loss(output, target, backgrd)
                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
                self.total_loss.update(loss.item())

                seg_metrics = eval_metrics(output, target, self.num_classes)

                self._update_seg_metrics(*seg_metrics)

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 10:
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                pixAcc, mIoU, _ = self._get_seg_metrics().values()
                if batch_idx % int(len(self.val_loader) -1) == 0 and batch_idx > 0 and epoch % 5 == 0:
                    print('EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format(epoch,
                                                self.total_loss.average, pixAcc, mIoU))

            seg_metrics = self._get_seg_metrics()

            # WRTING & VISUALIZING THE MASKS
            val_img = []
            palette = self.train_loader.dataset.palette
            for d, t, o in val_visual:
                d = self.restore_transform(d)
                t, o = colorize_mask(t, palette), colorize_mask(o, palette)
                d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
                [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
                val_img.extend([d, t, o])
            val_img = torch.stack(val_img, 0)
            val_img = make_grid(val_img.cpu(), nrow=3, padding=5)

            if self.config.tensorboard:
                self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)
                # METRICS TO TENSORBOARD
                self.wrt_step = (epoch) * len(self.val_loader)
                self.writer.add_scalar(f'{self.wrt_mode}/loss', self.total_loss.average, self.wrt_step)

                for k, v in list(seg_metrics.items())[:-1]:
                    self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            ####保存图像
            if epoch > 270 and epoch % 50 == 0:
            # if epoch % 20 == 0:
                ndarr = val_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                im = Image.fromarray(ndarr)
                im.save(os.path.join(self.config.log_dir, str(epoch) + self.config.loss + 'val.png'))
                ###画confusion
                # confusion_matrix(y_true.view(-1).cpu().numpy(), pred.view(-1).cpu().numpy(), labels=range(num_classes),
                #                  normalize='pred')

            log = {'loss': self.total_loss.average, **seg_metrics}

        return log

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }