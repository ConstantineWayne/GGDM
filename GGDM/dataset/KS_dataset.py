import csv
import math
import os
import random
import copy
import numpy as np
import torch
import torch.nn.functional
import torchaudio
from PIL import Image
from scipy import signal
from torch.utils.data import Dataset
from torchvision import transforms


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def inv_norm_tensor(img):
    inv_norm = UnNormalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    img[:, :, :] = inv_norm(img)
    return img


class KS_dataset(Dataset):

    def __init__(self, config,mode, select_ratio=1, v_norm=True, a_norm=False, name="KS"):
        self.data = []
        self.label = []
        self.config = config
        if mode == 'train':
            csv_path = 'ks_train_overlap.txt'
            self.audio_path = 'k400/train_spec'
            self.visual_path = 'Image-01-FPS-SE'

        elif mode == 'val':
            csv_path = 'ks_test_overlap.txt'
            self.audio_path = './data/test_spec'
            self.visual_path = './data/val-frames-1fps/test'

        else:
            csv_path = 'ks_test_overlap.txt'
            self.audio_path = '/k400/val_spec'
            self.visual_path = 'Image-01-FPS-SE'

        with open(csv_path) as f:
            for line in f:
                item = line.strip().split(" ")
                name = item[0]

                audio_file = os.path.join(self.audio_path, name + '.npy')
                visual_dir = os.path.join(self.visual_path, name)

                # 如果 audio 或 visual 缺失，则跳过
                if not os.path.exists(audio_file) or not os.path.isdir(visual_dir):
                    continue

                files_list = os.listdir(visual_dir)
                if len(files_list) > 3:
                    self.data.append(name)
                    self.label.append(int(item[-1]))

        print('Data load finish')
        self.normalize = v_norm
        self.mode = mode
        self._init_atransform()

        if mode == 'train' and select_ratio < 1:
            self._random_choice(select_ratio)
        print('# of files = %d ' % len(self.data))

    def _random_choice(self, select_ratio=0.40):
        num_data = len(self.data)
        selected_id = np.random.choice(np.arange(num_data), int(num_data * select_ratio))
        selected_data = [self.data[idx] for idx in selected_id]
        selected_label = [self.label[idx] for idx in selected_id]
        self.data = selected_data
        self.label = selected_label

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        av_file = self.data[idx]

        # 读取音频
        spectrogram = np.load(self.audio_path + '/' + av_file + '.npy')
        spectrogram = np.expand_dims(spectrogram, axis=0)

        # 读取视觉数据
        path = os.path.join(self.visual_path, av_file)
        files_list = sorted([fn for fn in os.listdir(path) if fn.endswith("jpg")])
        file_num = len(files_list)

        if self.mode == 'train':
            transf = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            transf = [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
            ]
        if self.normalize:
            transf.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        transf = transforms.Compose(transf)

        pick_num = 3  # 采样图片数
        if file_num < pick_num:
            # 如果图片不足 pick_num，则循环重复取样
            sampled_files = list(cycle(files_list))[:pick_num]
        else:
            # **随机采样 `pick_num` 张图片**
            sampled_files = random.sample(files_list, pick_num)

        # 读取并转换图像
        image_arr = []
        for img_file in sampled_files:
            img_path = os.path.join(path, img_file)
            image = Image.open(img_path).convert('RGB')
            image_tensor = transf(image).unsqueeze(1).float()
            image_arr.append(image_tensor)

        image_n = torch.cat(image_arr, dim=1)  # 拼接成 (C, T, H, W)

        # label = self.label[idx]
        one_hot = np.eye(self.config["setting"]["num_class"])
        one_hot_label = one_hot[self.label[idx]]
        label = torch.FloatTensor(one_hot_label)

        sample = {
            'audio': spectrogram,
            'clip': image_n,
            'target': label
        }
        return sample
