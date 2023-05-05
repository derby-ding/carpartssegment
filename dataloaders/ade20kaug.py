from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import torch
from PIL import Image
from glob import glob
import torchvision.transforms as standard_transforms
import dataloaders.joint_transforms as joint_transforms
import dataloaders.transforms as extended_transforms
from torch.utils.data import DataLoader, Dataset
from dataloaders.randaugment import RandAugment

class ADE20K_AG(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=100, base_size=100, scale_min=1.0, scale_max=1.0, num_workers=6,
                    shuffle=False, color=False, blur=False, augment=None, val_split=None):
        ####augment参数为随机投影增强，5表示选择5种变换，共10种，20表示强度，最高30
        kwargs = {
            'root': data_dir,
            'filename': split,###train or val，图像子集名
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale_min': scale_min,
            'scale_max': scale_max,
            'blur': blur,
            'color': color,
        }

        self.dataset = ADE20K_aug(**kwargs)
        super(ADE20K_AG, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
        # self.shuffle = shuffle
        #
        # self.init_kwargs = {
        #     'dataset': self.dataset,
        #     'batch_size': batch_size,
        #     'shuffle': self.shuffle,
        #     'num_workers': num_workers,
        #     # 'worker_init_fn': seed_worker,
        #     'pin_memory': True
        # }
        # super(BaseDataLoader, self).__init__(sampler=self.train_sampler, **self.init_kwargs)


class ADE20K_aug(Dataset):
    def __init__(self, **kwargs):
        self.root = kwargs['root']
        self.MEAN = [0.48897059, 0.46548275, 0.4294]
        self.STD = [0.22861765, 0.22948039, 0.24054667]
        self.scale_min = kwargs['scale_min']
        self.scale_max = kwargs['scale_max']
        self.crop_size = kwargs['crop_size']
        self.augment = kwargs['augment']
        self.base_size = kwargs['base_size']
        self.color = kwargs['color']
        self.blur = kwargs['blur']
        self.palette = palette.ADE20K_palette
        self.split = kwargs['filename']
        self.files = []
        self._set_files()
        # super(ADE20K_aug, self).__init__(**kwargs)
        # crop_size = int(self.crop_size)
        train_joint_transform_list = [joint_transforms.RandomSizeAndCrop(self.crop_size, False, scale_min=self.scale_min, scale_max=self.scale_max, full_size=False, pre_size=self.base_size)]
        train_joint_transform_list.append(
            joint_transforms.RandomHorizontallyFlip())

        if self.augment is not None and not self.augment in ['none', 'None', 'null']:
            N, M = [int(i) for i in self.augment.split(',')]
            assert isinstance(N, int) and isinstance(M, int), \
                f'Either N {N} or M {M} not integer'
            train_joint_transform_list.append(RandAugment(N, M))

        ######################################################################
        # Image only augmentations
        ######################################################################
        train_input_transform = []

        if self.color:
            train_input_transform += [extended_transforms.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25)]
        if self.blur:
        #     train_input_transform += [extended_transforms.RandomBilateralBlur()]
        # elif args.gblur:
            train_input_transform += [extended_transforms.RandomGaussianBlur()]

        mean_std = (self.MEAN, self.STD)
        train_input_transform += [standard_transforms.ToTensor(),
                                  standard_transforms.Normalize(*mean_std)]
        train_input_transform = standard_transforms.Compose(train_input_transform)

        val_input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])

        target_transform = extended_transforms.MaskToTensor()

        target_train_transform = extended_transforms.MaskToTensor()
        # self.dataset = ADE20KDataset(**kwargs)
        # super(ADE20K, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
        self.palette = palette.ADE20K_palette

        val_joint_transform_list = None
        if self.split == 'valid':###测试集
            self.joint_transform_list = val_joint_transform_list
            self.img_transform = val_input_transform
            self.label_transform = target_train_transform
        else:
            self.joint_transform_list = train_joint_transform_list
            self.img_transform = train_input_transform
            self.label_transform = target_transform

    def _set_files(self):

        ####new, more general, support voc
        if self.split in ["train", "valid"]:
            # self.image_dir = os.path.join(self.root, 'JPEGImages', self.split)
            # self.label_dir = os.path.join(self.root, 'annotation', self.split)
            self.image_dir = os.path.join(self.root, self.split, 'JPEGImages')
            self.label_dir = os.path.join(self.root, self.split, 'annotation')
            self.files = [os.path.basename(path).split('.')[0] for path in glob(self.image_dir + '/*.jpg')]
            self.files.sort()  ###排序
            print('check files', len(self.files), self.files[:5])
        else:
            raise ValueError(f"Invalid split name {self.split}")
    def do_transforms(self, img, mask, centroid=None):
        """
        Do transformations to image and mask

        :returns: image, mask
        """

        if self.joint_transform_list is not None:
            for idx, xform in enumerate(self.joint_transform_list):
                if idx == 0 and centroid is not None:
                    # HACK! Assume the first transform accepts a centroid
                    outputs = xform(img, mask, centroid)
                else:
                    outputs = xform(img, mask)

                if len(outputs) == 3:
                    img, mask, scale_float = outputs
                else:
                    img, mask = outputs

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            mask = self.label_transform(mask)

        return img, mask

    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        label_path = os.path.join(self.label_dir, image_id + '.png')
        # label_path = os.path.join(self.label_dir, image_id + '.jpg')
        image, label = self.do_transforms(Image.open(image_path).convert('RGB'), Image.open(label_path))
        image = np.asarray(image, dtype=np.float32)
        label = np.asarray(label, dtype=np.int32)  # from -1 to 149
        return image, label, image_id
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, label, image_id = self._load_data(index)
        # if self.val:
        #     image, label = self._val_augmentation(image, label)
        # elif self.augment:
        #     image, label = self._augmentation(image, label)
        #
        # label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        # image = Image.fromarray(np.uint8(image))
        # if self.return_id:
        #     return  self.normalize(self.to_tensor(image)), label, image_id
        return image, label
