import os

import numpy as np
import torch
import yaml
from torch import nn


class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

    def forward(self, preds, targets, weight=None):
        if isinstance(preds, list):
            N = len(preds)
            if weight is None:
                weight = preds[0].new_ones(1)

            errs = [self._forward(preds[n], targets[n], weight[n])
                    for n in range(N)]
            err = torch.mean(torch.stack(errs))

        elif isinstance(preds, torch.Tensor):
            if weight is None:
                weight = preds.new_ones(1)
            err = self._forward(preds, targets, weight)

        return err


class LogDepthLoss(BaseLoss):
    def __init__(self):
        super(LogDepthLoss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(torch.log(torch.abs(pred - target) + 1))


class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


# def compute_errors(gt, pred):
#     """Computation of error metrics between predicted and ground truth depths
#     """
#     # select only the values that are greater than zero
#     mask = gt > 0
#     pred = pred[mask]
#     gt = gt[mask]
#
#     thresh = np.maximum((gt / pred), (pred / gt))
#     a1 = (thresh < 1.25).mean()
#     a2 = (thresh < 1.25 ** 2).mean()
#     a3 = (thresh < 1.25 ** 3).mean()
#
#     rmse = (gt - pred) ** 2
#     rmse = np.sqrt(rmse.mean())
#     if rmse != rmse:
#         rmse = 0.0
#     if a1 != a1:
#         a1 = 0.0
#     if a2 != a2:
#         a2 = 0.0
#     if a3 != a3:
#         a3 = 0.0
#
#     abs_rel = np.mean(np.abs(gt - pred) / gt)
#     log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
#     mae = (np.abs(gt - pred)).mean()
#     if abs_rel != abs_rel:
#         abs_rel = 0.0
#     if log_10 != log_10:
#         log_10 = 0.0
#     if mae != mae:
#         mae = 0.0
#
#     return abs_rel, rmse, a1, a2, a3, log_10, mae


def compute_errors(gt: torch.Tensor, pred: torch.Tensor):
    """
    @param gt: 真实深度，大小为[batch_size, 1, height, width]
    @param pred: 预测深度，大小为[batch_size, 1, height, width]
    @return:
        [abs_rel: 绝对相对误差,
        rmse: 均方根误差,
        a1: 阈值1,
        a2: 阈值2,
        a3: 阈值3,
        log_10: 对数误差,
        mae: 平均绝对误差]
        计算预测深度与地面真实深度之间的误差指标
    """
    """Computation of error metrics between predicted and ground truth depths
        """
    gt = gt.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()

    mask = gt > 0
    pred = pred[mask]
    gt = gt[mask]

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    if rmse != rmse:
        rmse = 0.0
    if a1 != a1:
        a1 = 0.0
    if a2 != a2:
        a2 = 0.0
    if a3 != a3:
        a3 = 0.0

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    mae = (np.abs(gt - pred)).mean()
    if abs_rel != abs_rel:
        abs_rel = 0.0
    if log_10 != log_10:
        log_10 = 0.0
    if mae != mae:
        mae = 0.0

    return abs_rel, rmse, a1, a2, a3, log_10, mae


class TextWrite(object):
    ''' Wrting the values to a text file
    '''

    def __init__(self, filename):
        self.filename = filename  # 初始化类，将文件名字符串赋给类的filename属性
        self.file = open(self.filename, "w+")  # 以读写模式打开文件，如果文件不存在就创建它
        self.file.close()  # 关闭文件

    def write_line(self, data_list: list):
        self.file = open(self.filename, "a")  # 以追加模式打开文件
        str_write = ""
        for item in data_list:
            if isinstance(item, int):
                str_write += "{:03d}".format(item)
            if isinstance(item, str):
                str_write += item
            if isinstance(item, float):
                str_write += "{:.6f}".format(item)
            str_write += ","
        # 去除最后的逗号
        str_write = str_write[:-1]
        str_write += "\n"
        self.file.write(str_write)  # 将str_write的内容写入文件
        self.file.close()  # 关闭文件


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):  # 如果paths是列表而不是字符串
        for path in paths:  # 迭代列表中的每一个项
            mkdir(path)  # 创建对应的目录
    else:
        mkdir(paths)  # 如果paths不是列表，就直接创建目录


def mkdir(path):
    if not os.path.exists(path):  # 如果指定的路径不存在
        os.makedirs(path)  # 就创建目录


def save_config_to_yaml(config, file_path):
    # Convert the Config instance to a dictionary
    config_dict = {attr: getattr(config, attr) for attr in dir(config) if
                   not callable(getattr(config, attr)) and not attr.startswith("__")}
    # Write the dictionary to a YAML file
    with open(file_path, "w") as yaml_file:
        yaml.dump(config_dict, yaml_file, default_flow_style=False)


def load_config_from_yaml(file_path):
    with open(file_path, 'r') as yaml_file:
        config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return config_dict


def update_config_from_dict(config, config_dict):
    for key, value in config_dict.items():
        setattr(config, key, value)
