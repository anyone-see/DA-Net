import os.path
import re
import time
import librosa
import random
import math
import numpy as np
import glob
import torch
import pickle
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import torch.utils.data as data
from scipy.io.wavfile import read as read_wav
from scipy.signal import stft


class MinMaxNorm(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        assert isinstance(min, (float, tuple)) and isinstance(max, (float, tuple))
        self.min = torch.tensor(min)
        self.max = torch.tensor(max)

    def forward(self, tensor):
        if tensor.shape[0] == 2:
            norm_tensor_c0 = (tensor[0, ...] - self.min[0]) / (self.max[0] - self.min[0])
            norm_tensor_c1 = (tensor[1, ...] - self.min[1]) / (self.max[1] - self.min[1])
            norm_tensor = torch.concatenate([norm_tensor_c0.unsqueeze(0), norm_tensor_c1.unsqueeze(0)], dim=0)

        else:
            norm_tensor = (tensor - self.min) / (self.max - self.min)

        return norm_tensor


def get_scene_list(path):
    with open(path) as f:
        scenes_test = f.readlines()
    scenes_test = [x.strip() for x in scenes_test]
    return scenes_test


def normalize(samples, desired_rms=0.1, eps=1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples ** 2)))
    samples = samples * (desired_rms / rms)
    return samples


def generate_spectrogram(audioL, audioR, winl=32):
    channel_1_spec = librosa.stft(audioL, n_fft=512, win_length=winl)
    channel_2_spec = librosa.stft(audioR, n_fft=512, win_length=winl)

    spectro_two_channel = np.concatenate(
        (np.expand_dims(np.abs(channel_1_spec), axis=0), np.expand_dims(np.abs(channel_2_spec), axis=0)), axis=0)

    return spectro_two_channel


def process_image(rgb, augment):
    if augment:

        enhancer = ImageEnhance.Brightness(rgb)
        rgb = enhancer.enhance(random.random() * 0.6 + 0.7)
        enhancer = ImageEnhance.Color(rgb)
        rgb = enhancer.enhance(random.random() * 0.6 + 0.7)
        enhancer = ImageEnhance.Contrast(rgb)
        rgb = enhancer.enhance(random.random() * 0.6 + 0.7)
    return rgb


def get_file_list(path):
    file_name_prefix = []
    for file in os.listdir(path):
        file_name_prefix.append(file.split('.')[0])
    file_name_prefix = np.unique(file_name_prefix)
    return file_name_prefix


def add_to_list(index_list, file_path, data):
    for index in index_list:
        rgb = os.path.join(file_path, index) + '.png'
        audio = os.path.join(file_path, index) + '.wav'
        depth = os.path.join(file_path, index) + '.npy'
        data.append([rgb, audio, depth])


def load_from_csv(filename):
    csv_data = np.loadtxt(filename, delimiter=',', dtype=str, skiprows=1)
    return csv_data.tolist()


class AudioVisualDataset(data.Dataset):
    def __init__(self, dataset_name, mode, config):
        super(AudioVisualDataset, self).__init__()
        self.train_data = []
        self.val_data = []
        self.test_data = []
        replica_dataset_path = config.replica_dataset_path
        mp3d_dataset_path = config.mp3d_dataset_path
        metadata_path = config.metadata_path
        self.audio_resize = config.audio_resize
        self.audio_normalization = config.audio_normalization
        if dataset_name == 'mp3d':
            self.win_length = 32
            self.audio_sampling_rate = 16000
            self.audio_length = 0.060
            # train,val,test scenes
            train_scenes_file = os.path.join(metadata_path, 'mp3d_scenes_train.txt')
            val_scenes_file = os.path.join(metadata_path, 'mp3d_scenes_val.txt')
            test_scenes_file = os.path.join(metadata_path, 'mp3d_scenes_test.txt')
            train_scenes = get_scene_list(train_scenes_file)
            val_scenes = get_scene_list(val_scenes_file)
            test_scenes = get_scene_list(test_scenes_file)
            for scene in os.listdir(mp3d_dataset_path):
                if scene in train_scenes:
                    for orn in os.listdir(os.path.join(mp3d_dataset_path, scene)):
                        file_name_prefix = get_file_list(os.path.join(mp3d_dataset_path, scene, orn))
                        add_to_list(file_name_prefix, os.path.join(mp3d_dataset_path, scene, orn), self.train_data)
                elif scene in val_scenes:
                    for orn in os.listdir(os.path.join(mp3d_dataset_path, scene)):
                        file_name_prefix = get_file_list(os.path.join(mp3d_dataset_path, scene, orn))
                        add_to_list(file_name_prefix, os.path.join(mp3d_dataset_path, scene, orn), self.val_data)
                elif scene in test_scenes:
                    for orn in os.listdir(os.path.join(mp3d_dataset_path, scene)):
                        file_name_prefix = get_file_list(os.path.join(mp3d_dataset_path, scene, orn))
                        add_to_list(file_name_prefix, os.path.join(mp3d_dataset_path, scene, orn), self.test_data)
        if dataset_name == 'replica':
            self.win_length = 64
            self.audio_sampling_rate = 44100
            self.audio_length = 0.060
            self.train_data = load_from_csv(os.path.join(metadata_path, 'train.csv'))
            self.val_data = load_from_csv(os.path.join(metadata_path, 'val.csv'))
            self.test_data = load_from_csv(os.path.join(metadata_path, 'test.csv'))


        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        vision_transform_list = [transforms.ToTensor(), normalize]
        self.vision_transform = transforms.Compose(vision_transform_list)
        self.mode = mode
        self.dataset = dataset_name

    def __getitem__(self, index):
        # rgb, audio, depth
        if self.mode == 'train':
            data_ = self.train_data[index]
        elif self.mode == 'val':
            data_ = self.val_data[index]
        elif self.mode == 'test':
            data_ = self.test_data[index]
        # print(data_)
        rgb_path, audi_path, depth_path = data_[0], data_[1], data_[2]
        rgb = Image.open(rgb_path).convert('RGB')

        if self.mode == 'train':
            rgb = process_image(rgb, True)
        rgb = self.vision_transform(rgb)
        audio, audio_rate = librosa.load(audi_path, sr=self.audio_sampling_rate, mono=False, duration=self.audio_length)
        if self.audio_normalization:
            audio = normalize(audio)
        audio_spec_both = torch.FloatTensor(generate_spectrogram(audio[0, :], audio[1, :], self.win_length))
        if self.audio_resize:
            audio_spec_both = transforms.Resize((128, 128), antialias=True)(audio_spec_both)
        depth = torch.FloatTensor(np.load(depth_path))
        depth = depth.unsqueeze(0)
        return {'img': rgb, 'audio': audio_spec_both, 'depth': depth}

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        elif self.mode == 'val':
            return len(self.val_data)
        elif self.mode == 'test':
            return len(self.test_data)


def get_data_loader(dataset_name, mode, shuffle, config):
    if dataset_name == 'BV2':
        dataset = BatvisionV2Dataset(mode, config)
    elif dataset_name == 'BV1':
        dataset = BatvisionV1Dataset(mode, config)
    else:
        dataset = AudioVisualDataset(dataset_name, mode, config)

    if mode == 'val':
        return data.DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers,
                               drop_last=True)
    return data.DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers)


class BatvisionV2Dataset(data.Dataset):
    def __init__(self, mode, config):
        super(BatvisionV2Dataset, self).__init__()

        self.dataset_path = config.bv2_dataset_path

        self.train_data = []
        self.val_data = []
        self.test_data = []
        for scene in os.listdir(self.dataset_path):
            _file = np.loadtxt(os.path.join(self.dataset_path, scene, 'train.csv'), dtype='str', delimiter=',',
                               skiprows=1)
            for item in _file:
                rgb_path = os.path.join(self.dataset_path, item[7], item[8])
                audi_path = os.path.join(self.dataset_path, item[4], item[5])
                depth_path = os.path.join(self.dataset_path, item[10], item[11])
                self.train_data.append((rgb_path, audi_path, depth_path))
            _file = np.loadtxt(os.path.join(self.dataset_path, scene, 'val.csv'), dtype='str', delimiter=',',
                               skiprows=1)
            for item in _file:
                rgb_path = os.path.join(self.dataset_path, item[7], item[8])
                audi_path = os.path.join(self.dataset_path, item[4], item[5])
                depth_path = os.path.join(self.dataset_path, item[10], item[11])
                self.val_data.append((rgb_path, audi_path, depth_path))
            _file = np.loadtxt(os.path.join(self.dataset_path, scene, 'test.csv'), dtype='str', delimiter=',',
                               skiprows=1)
            for item in _file:
                rgb_path = os.path.join(self.dataset_path, item[7], item[8])
                audi_path = os.path.join(self.dataset_path, item[4], item[5])
                depth_path = os.path.join(self.dataset_path, item[10], item[11])
                self.test_data.append((rgb_path, audi_path, depth_path))

        self.audio_resize = config.audio_resize
        self.audio_normalization = config.audio_normalization
        self.max_depth = config.max_depth

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        vision_transform_list = [transforms.ToTensor(), transforms.Resize((256, 256), antialias=False), normalize]
        self.vision_transform = transforms.Compose(vision_transform_list)
        # self.audio_transform = transforms.Resize((256, 256))
        self.depth_transform = transforms.Compose([transforms.Resize((256, 256), antialias=False)])
        self.mode = mode

    def __getitem__(self, index):
        # rgb, audio, depth
        if self.mode == 'train':
            data_ = self.train_data[index]
        elif self.mode == 'val':
            data_ = self.val_data[index]
        elif self.mode == 'test':
            data_ = self.test_data[index]

        rgb_path, audio_path, depth_path = data_[0], data_[1], data_[2]
        rgb = Image.open(rgb_path).convert('RGB')

        if self.mode == 'train':
            rgb = process_image(rgb, True)
        rgb = self.vision_transform(rgb)
        audio, audio_rate = librosa.load(audio_path, sr=None, mono=False, duration=(2 * self.max_depth / 340))
        if self.audio_normalization:
            audio = normalize(audio)
        audio_spec_both = torch.FloatTensor(generate_spectrogram(audio[0, :], audio[1, :], 64))

        # audio_spec_both = self.audio_transform(audio_spec_both)
        depth = np.load(depth_path).astype(np.float32)
        depth = depth / 1000  # to go from mm to m
        depth[depth > self.max_depth] = self.max_depth

        depth = torch.FloatTensor(depth)
        depth = depth.unsqueeze(0)
        depth = self.depth_transform(depth)
        return {'img': rgb, 'audio': audio_spec_both, 'depth': depth}

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        elif self.mode == 'val':
            return len(self.val_data)
        elif self.mode == 'test':
            return len(self.test_data)


class BatvisionV1Dataset(data.Dataset):

    def __init__(self, mode, config):

        self.dataset_path = config.bv1_dataset_path
        self.mode = mode

        self.train_data = []
        self.val_data = []
        self.test_data = []

        self.data_path_list = []

        if mode == 'train':
            csv_file = np.loadtxt(os.path.join(self.dataset_path, 'train.csv'), dtype='str', delimiter=',',
                                  skiprows=1)
        elif mode == 'val':
            csv_file = np.loadtxt(os.path.join(self.dataset_path, 'val.csv'), dtype='str', delimiter=',',
                                  skiprows=1)
        elif mode == 'test':
            csv_file = np.loadtxt(os.path.join(self.dataset_path, 'test.csv'), dtype='str', delimiter=',',
                                  skiprows=1)

        for item in csv_file:
            rgb_path = os.path.join(self.dataset_path, item[5])
            audi_left_path = os.path.join(self.dataset_path, item[2])
            audi_right_path = os.path.join(self.dataset_path, item[3])
            depth_path = os.path.join(self.dataset_path, item[7])
            self.data_path_list.append((rgb_path, audi_left_path, audi_right_path, depth_path))

        self.audio_resize = config.audio_resize
        self.audio_normalization = config.audio_normalization
        self.max_depth = config.max_depth

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        vision_transform_list = [transforms.ToTensor(), transforms.Resize((256, 256), antialias=False), normalize]
        self.vision_transform = transforms.Compose(vision_transform_list)
        self.depth_transform = transforms.Compose([transforms.Resize((256, 256), antialias=False)])


    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, idx):
        rgb_path, audio_left_path, audio_right_path, depth_path = self.data_path_list[idx]

        rgb = Image.open(rgb_path).convert('RGB')

        if self.mode == 'train':
            rgb = process_image(rgb, True)
        rgb = self.vision_transform(rgb)

        audio_left = np.load(audio_left_path).astype(np.float32)
        audio_right = np.load(audio_right_path).astype(np.float32)
        audio = np.stack((audio_left, audio_right))
        if self.audio_normalization:
            audio = normalize(audio)
        audio = torch.FloatTensor(generate_spectrogram(audio[0, :], audio[1, :], 64))

        depth = np.load(depth_path)
        # Set nan value to 0
        depth = np.nan_to_num(depth)
        depth[depth == -np.inf] = 0
        depth[depth == np.inf] = self.max_depth

        depth = depth / 1000  # to go from mm to m
        depth[depth > self.max_depth] = self.max_depth
        depth[depth < 0.0] = 0.0

        depth = torch.FloatTensor(depth)
        depth = depth.unsqueeze(0)

        depth = self.depth_transform(depth)
        return {'img': rgb, 'audio': audio, 'depth': depth}