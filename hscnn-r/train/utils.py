from __future__ import division

import math

import torch
import torch.nn as nn
from torch.autograd import Variable

import logging
import numpy as np
import os
import hdf5storage


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_logger(file_dir):
    """Print the results in the log file."""
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger


def save_checkpoint(model_path, epoch, iteration, model, optimizer, layer, input_chan, output_chan, str, loss):
    """Save the checkpoint."""
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    # d is dropout percentage, loss is test loss percentage
    torch.save(state, os.path.join(model_path, 'layer%d_%d-to-%d_%s+loss+%2.2f_%d.pkl'
                                   % (layer, input_chan, output_chan, str, loss * 100, epoch)))


def save_matv73(mat_name, var_name, var):
    hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)


def record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, test_loss))
    loss_csv.flush()
    loss_csv.close


def get_reconstruction(input, num_split, dimension, model):
    """As the limited GPU memory split the input."""
    input_split = torch.split(input, int(input.shape[3] / num_split), dim=dimension)
    output_split = []
    for i in range(num_split):
        with torch.no_grad():
            var_input = Variable(input_split[i].cuda())
            var_output = model(var_input)
            output_split.append(var_output.data)
            if i == 0:
                output = output_split[i]
            else:
                output = torch.cat((output, output_split[i]), dim=dimension)

    return output


def reconstruction(rgb, model):
    """Output the final reconstructed hyperspectral images."""
    img_res = get_reconstruction(torch.from_numpy(rgb).float(), 1, 3, model)
    img_res = img_res.cpu().numpy()
    img_res = np.transpose(np.squeeze(img_res))
    img_res_limits = img_res
    # img_res_limits = np.minimum(img_res, 1)
    # img_res_limits = np.maximum(img_res_limits, 0)
    return img_res_limits


    # rrmse = np.mean((np.sqrt(np.power(error, 2))))
    # chan = img_gt.shape[0]
    # flat_error = error.clone().detach().view([chan, -1])
    # flat_outputs = img_res.clone().detach().view([chan, -1])
    # flat_label = img_gt.clone().detach().view([chan, -1])
    # y1 = lambda a, b: (flat_outputs[a] - flat_outputs[b])
    # y2 = lambda a, b: (flat_label[a] - flat_label[b])
    # flat_rrmse = rrmse.clone().detach()
    # for i in range(chan):
    #     lf = i - 1 if i > 0 else 0
    #     r = i + 1 if i < chan - 1 else chan - 1
    #
    #     flat_error[i] = (
    #             torch.arccos(torch.round(
    #                 torch.sqrt(((flat_rrmse ** 2 + y1(r, i) * y2(r, i)) ** 2)
    #                            / (flat_rrmse ** 2 + y1(r, i) ** 2) / (flat_rrmse ** 2 + y2(r, i) ** 2))
    #                 , decimals=3))
    #             +
    #             torch.arccos(torch.round(
    #                 torch.sqrt(((flat_rrmse ** 2 + y1(i, lf) * y2(i, lf)) ** 2)
    #                            / (flat_rrmse ** 2 + y1(i, lf) ** 2) / (flat_rrmse ** 2 + y2(i, lf) ** 2))
    #                 , decimals=3))
    #     )
        # flat_error[i] = (
        #     torch.abs((flat_outputs[r] - flat_outputs[i]) - (flat_label[r] - flat_label[i])) +
        #     torch.abs((flat_outputs[i] - flat_outputs[lf]) - (flat_label[i] - flat_label[lf]))
        #     )
    # g_error = torch.mean(flat_error.view(-1))
