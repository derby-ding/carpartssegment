import os
import logging
import json
import math
import torch
import datetime
from torch.utils import tensorboard
from utils import helpers
import pandas as pd
import utils.lr_scheduler
from utils.sync_batchnorm import convert_model
from utils.sync_batchnorm import DataParallelWithCallback
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as mpatches
from utils.palette import ADE20K_palette
####保证尽量复现
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

class BaseTrainer:
    def __init__(self, model, loss, resume, config, train_loader, val_loader=None, train_logger=None):
        self.model = model
        self.loss = loss
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_logger = train_logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.do_validation = self.config.val
        self.start_epoch = 1
        self.improved = False

        # SETTING THE DEVICE
        self.device, availble_gpus = self._get_available_devices(self.config.n_gpu)
        if config.use_synch_bn:
            self.model = convert_model(self.model)
            self.model = DataParallelWithCallback(self.model, device_ids=availble_gpus)
        else:
            self.model = torch.nn.DataParallel(self.model, device_ids=availble_gpus)
        self.model.to(self.device)

        # CONFIGS
        # cfg_trainer = self.config['trainer']
        self.epochs = self.config.epochs
        self.save_period = self.config.save_period

        # OPTIMIZER
        if self.config.differential_lr:
            if isinstance(self.model, torch.nn.DataParallel):
                trainable_params = [{'params': filter(lambda p:p.requires_grad, self.model.module.get_decoder_params())},
                                    {'params': filter(lambda p:p.requires_grad, self.model.module.get_backbone_params()), 
                                    'lr': config.lr / 10}]
            else:
                trainable_params = [{'params': filter(lambda p:p.requires_grad, self.model.get_decoder_params())},
                                    {'params': filter(lambda p:p.requires_grad, self.model.get_backbone_params()), 
                                    'lr': config.lr / 10}]
        else:
            trainable_params = filter(lambda p:p.requires_grad, self.model.parameters())
        if config.optimizer=='SGD':
            self.optimizer = getattr(torch.optim, config.optimizer)(trainable_params, lr=config.lr, weight_decay=config.weight_decay, momentum=config.momentum)
        else:
            self.optimizer = getattr(torch.optim, config.optimizer)(trainable_params, lr=config.lr,
                                                                    weight_decay=config.weight_decay)
        self.lr_scheduler = getattr(utils.lr_scheduler, config.lr_scheduler)(self.optimizer, self.epochs, len(train_loader))

        # MONITORING
        self.mnt_metric = 'Pixel_Accuracy'
        self.mnt_best = -math.inf #if self.mnt_mode == 'max' else math.inf
        self.early_stoping = self.config.early_stop

        # CHECKPOINTS & TENSOBOARD
        start_time = datetime.datetime.now().strftime('%m-%d')
        self.checkpoint_dir = os.path.join(self.config.save_dir, self.config.arch)##
        helpers.dir_exists(self.checkpoint_dir)

        if not os.path.exists(self.config.log_dir):
            os.mkdir(self.config.log_dir)
        self.writer_dir = os.path.join(self.config.log_dir, start_time)
        self.writer = tensorboard.SummaryWriter(self.writer_dir)
        checkpointpath = os.path.join(self.config.save_dir, self.config.arch, self.config.loss+'best_model.pth')##
        if resume: self._resume_checkpoint(checkpointpath)

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            self.logger.warning('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            self.logger.warning(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu
            
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        self.logger.info(f'Detected GPUs: {sys_gpu} Requested: {n_gpu}')
        available_gpus = list(range(n_gpu))
        return device, available_gpus
    
    def train(self):
        num_classes = self.config.classes
        if self.config.fix_seed:
            set_seed(3047)####3047 is all you need
        trloss = []##trloss
        tracc = []
        triou = []
        trIOUs = [[] for _ in range(num_classes)]
        vlloss = []
        vlacc = []
        vliou = []
        vlIOUs = [[] for _ in range(num_classes)]
        LR = []
        glepoch = 0
        for epoch in range(self.start_epoch, self.epochs+1):
            # RUN TRAIN (AND VAL)
            results = self._train_epoch(epoch)

            if self.do_validation and epoch % self.config.val_per_epochs == 0:
                ###同时保存train val，保证维度相同print('check results ', results, type(results))
                LR.append(self.lr_scheduler.get_last_lr()[0] * 10)  ####
                trloss.append(results['loss'])
                tracc.append(results['Pixel_Accuracy'])
                triou.append(results['Mean_IoU'])
                for i, iou in enumerate(trIOUs):
                    iou.append(results['Class_IoU'][i])
                results = self._valid_epoch(epoch)
                vlloss.append(results['loss'])
                vlacc.append(results['Pixel_Accuracy'])
                vliou.append(results['Mean_IoU'])
                for i, iou in enumerate(vlIOUs):
                    iou.append(results['Class_IoU'][i])

                log = {'epoch': epoch, **results}
                try:
                    self.improved = (log[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    print(f'The metrics being tracked ({self.mnt_metric}) has not been calculated. Training stops.')
                    break
                print('compare best ', self.mnt_metric, 'val best ', log[self.mnt_metric], 'record best', self.mnt_best)
                if self.improved:
                    self.mnt_best = log[self.mnt_metric]
                    self._save_checkpoint(epoch, save_best=True)####保存最优模型
                    self.not_improved_count = 0
                else:
                    self.not_improved_count += 1

                if self.not_improved_count > self.early_stoping:
                    print(f'\nPerformance didn\'t improve for {self.early_stoping} epochs')
                    print('Training Stoped')
                    break
            glepoch = epoch
        ####修改palette，与masktojson一致。
        ##########################
        if not self.config.tensorboard:
            palette = ADE20K_palette
            palette = list(np.array(palette) / 255.0)
            affix = self.config.loss + str(glepoch)
            if self.config.background:
                affix = 'bk' + affix


            colormap = []
            for i in range(7):
                colormap.append(tuple(palette[i * 3:i * 3 + 3]))
            X = range(len(trloss))
            sns.lineplot(x=X, y=trloss, color=colormap[0])
            sns.lineplot(x=X, y=tracc, color=colormap[1])
            sns.lineplot(x=X, y=triou, color=colormap[2])
            sns.lineplot(x=X, y=vlloss, color=colormap[3])
            sns.lineplot(x=X, y=vlacc, color=colormap[4])
            sns.lineplot(x=X, y=vliou, color=colormap[5])
            sns.lineplot(x=X, y=LR, color=colormap[6])
            idx = np.argmax(triou)
            plt.text(idx, max(triou) + 0.01, 'triou'+str(triou[idx]))
            idx = np.argmax(vliou)
            plt.text(idx, max(vliou) + 0.01, 'vliou'+str(vliou[idx]))
            idx = np.argmax(tracc)
            plt.text(idx, max(tracc) + 0.01, 'tracc'+str(tracc[idx]))
            idx = np.argmax(vlacc)
            plt.text(idx, max(vlacc) + 0.01, 'vlacc'+str(vlacc[idx]))
            #plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题
            plt.legend(handles=[mpatches.Patch(color=c) for c in colormap], labels=['train_loss', 'train_pixacc', 'train_miou', 'val_loss', 'val_pixacc', 'val_miou', 'lr_rate*10'])
            plt.xlabel('epochs')
            plt.ylabel('scores')
            plt.ylim((0, 2))
            plt.savefig(os.path.join(self.writer_dir, affix+'train_log.png'))
            plt.close()###关闭当前
            print('check val miou \n', max(vliou), ' val acc ', max(vlacc), '\n train miou', max(triou), 'train acc ', max(tracc))
            clsname = ['background', 'backbumper', 'backglass', 'backdoor', 'backlight', 'frontbumper', 'frontglass', 'frontdoor', 'frontlight', 'fronthood', 'mirror', 'backhood', 'trunk', 'wheel']
            colormap = []
            for i in range(num_classes):
                colormap.append(tuple(palette[i * 3:i * 3 + 3]))
            trlen = len(trIOUs[0])
            # print('check plot ', trIOUs[0])
            for i in range(num_classes):
                sns.lineplot(x=range(trlen), y=trIOUs[i], color=colormap[i])
            plt.legend(handles=[mpatches.Patch(color=c) for c in colormap],
                       labels=clsname)
            plt.xlabel('epochs')
            plt.ylabel('scores')
            plt.savefig(os.path.join(self.writer_dir, affix+'train_ious.png'))
            plt.close()

            vllen = len(vlIOUs[0])
            for i in range(num_classes):
                sns.lineplot(x=range(vllen), y=vlIOUs[i], color=colormap[i])
            plt.legend(handles=[mpatches.Patch(color=c) for c in colormap],
                       labels=clsname)
            plt.xlabel('epochs')
            plt.ylabel('scores')
            plt.savefig(os.path.join(self.writer_dir, affix+'val_ious.png'))
            plt.close()
            ####保存df
            save_time = datetime.datetime.now().strftime('%m-%d')
            df = pd.DataFrame()
            for i in range(num_classes):
                df[clsname[i]] = vlIOUs[i]
            df['valacc'] = vlacc
            df['valiou'] = vliou
            df['triou'] = triou
            df['tracc'] = tracc
            df.to_csv(affix+save_time+'metrics.csv', index=False)

        return self.model
    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        # filename = os.path.join(self.checkpoint_dir, f'checkpoint-epoch{epoch}.pth')
        # self.logger.info(f'\nSaving a checkpoint: {filename} ...')
        # torch.save(state, filename)

        if save_best:
            filename = os.path.join(self.checkpoint_dir, self.config.loss+'best_model.pth')
            torch.save(state, filename)
            self.logger.info("Saving current best: best_model.pth")

    def _resume_checkpoint(self, resume_path):
        self.logger.info(f'Loading checkpoint : {resume_path}')
        checkpoint = torch.load(resume_path)

        # Load last run info, the model params, the optimizer and the loggers
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.not_improved_count = 0

        # if checkpoint['config']['arch'] != self.config['arch']:
        #     self.logger.warning({'Warning! Current model is not the same as the one in the checkpoint'})
        self.model.load_state_dict(checkpoint['state_dict'])

        # if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
        #     self.logger.warning({'Warning! Current optimizer is not the same as the one in the checkpoint'})
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info(f'Checkpoint <{resume_path}> (epoch {self.start_epoch}) was loaded')

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _valid_epoch(self, epoch):
        raise NotImplementedError

    def _eval_metrics(self, output, target):
        raise NotImplementedError

    
