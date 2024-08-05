import os
import torch
import numpy as np
from tqdm import tqdm

from data_loader.my_dataset import get_data_loader
from networks.builder import ModelBuilder
from util.util import compute_errors, load_config_from_yaml, update_config_from_dict, mkdirs
import argparse

from config import Config
from torch import nn
from util.util import silog_loss, LogDepthLoss

def creat_loss(opt):
    if opt.loss == 'LogDepthLoss':
        return LogDepthLoss()
    elif opt.loss == 'L1Loss':
        return nn.L1Loss()
    elif opt.loss == 'silog_loss':
        return silog_loss(variance_focus=opt.variance_focus)

def main(config_yaml_path):
    config = Config()
    # Set your desired file path for the YAML file

    # Load settings and update the config object
    config_dict = load_config_from_yaml(os.path.join(config_yaml_path, 'config.yml'))
    update_config_from_dict(config, config_dict)

    config.expr_dir=config_yaml_path
    loss_criterion = creat_loss(config)
    config.device = torch.device("cuda")

    config.model['weights'] = config.expr_dir
    builder = ModelBuilder()

    model = builder.build(config.model)

    model.to(config.device)
    model.eval()
    config.mode = 'test'

    dataloader_val = get_data_loader(config.dataset, config.mode, False, config)
    dataset_size_val = len(dataloader_val)


    losses, errs = [], []
    with torch.no_grad():
        pbar = tqdm(total=dataset_size_val * config.batch_size, desc="Processing")  # Initialize tqdm with the total number of items
        for i, val_data in enumerate(dataloader_val):
            val_data['audio'] = val_data['audio'].to(config.device)
            val_data['img'] = val_data['img'].to(config.device)
            val_data['depth'] = val_data['depth'].to(config.device)
            output = model.forward(val_data)
            depth_gt = val_data['depth']
            depth_predicted = output['depth_predicted']

            loss = loss_criterion(depth_predicted, depth_gt, depth_gt != 0)
            losses.append(loss.item())
            for j in range(depth_gt.shape[0]):
                errs.append(compute_errors(depth_gt[j], depth_predicted[j]))
                pbar.update(1)  # Update the progress bar after each item
        pbar.close()  # Close the progress bar after the loop

    mean_loss = sum(losses) / len(losses)
    mean_errs = np.array(errs).mean(0)

    print('Loss: {:.3f}, RMSE: {:.3f}'.format(mean_loss, mean_errs[1]))

    errors = {}
    errors['ABS_REL'], errors['RMSE'], errors['LOG10'] = mean_errs[0], mean_errs[1], mean_errs[5]
    errors['DELTA1'], errors['DELTA2'], errors['DELTA3'] = mean_errs[2], mean_errs[3], mean_errs[4]
    errors['MAE'] = mean_errs[6]

    print('ABS_REL:{:.3f}, LOG10:{:.3f}, MAE:{:.3f}'.format(errors['ABS_REL'], errors['LOG10'], errors['MAE']))
    print('DELTA1:{:.3f}, DELTA2:{:.3f}, DELTA3:{:.3f}'.format(errors['DELTA1'], errors['DELTA2'], errors['DELTA3']))
    print('===' * 25)

    # Save the results to a file
    results_file = os.path.join(config.expr_dir, f'results_{errors["RMSE"]:.4f}.txt')

    with open(results_file, "w+") as f:
        f.write('===' * 25 + '\n')
        f.write(config.dataset + '\t' + config.model['name'] + '\t' + config.loss + '\t' + config.optimizer + '\t' +
                str(config.lr) + '\t' + str(config.batch_size) + '\t' + str(config.epochs) + '\n')
        f.write('Loss: {:.3f}, RMSE: {:.3f}\n'.format(mean_loss, mean_errs[1]))
        f.write('ABS_REL:{:.3f}, LOG10:{:.3f}, MAE:{:.3f}\n'.format(errors['ABS_REL'], errors['LOG10'], errors['MAE']))
        f.write(
            'DELTA1:{:.3f}, DELTA2:{:.3f}, DELTA3:{:.3f}\n'.format(errors['DELTA1'], errors['DELTA2'], errors['DELTA3']))
        f.write('===' * 25 + '\n')
        f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('config_yaml_path', type=str, help='Path to the configuration YAML file')
    args = parser.parse_args()
    main(args.config_yaml_path)