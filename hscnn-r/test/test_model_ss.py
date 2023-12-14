from __future__ import division

import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import os
import numpy as np
# import matplotlib.pyplot as plt
# from imageio import imread

from resblock import resblock, conv_relu_res_relu_block
from utils import save_matv73, reconstruction, rrmse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

model_path = './models/layer14_20-to-100_val-1-7-17+loss+4.25_105.pkl'
input_chan = 20
model_label = model_path.split('/')[2][:-4]
chan_label = 'chann+{}'.format(input_chan)

img_path = './ss_results/'
var_name = 'reconstruct'

save_point = torch.load(model_path)
model_param = save_point['state_dict']
drop = 0
model = resblock(conv_relu_res_relu_block, 14, input_chan, 100, drop)
# model = nn.DataParallel(model)
model.load_state_dict(model_param)

model = model.cuda()
model.eval()

for img_name in sorted(os.listdir(img_path)):
    if 'input' in img_name and chan_label+'.mat' in img_name:
        print(img_name)
        img_path_name = os.path.join(img_path, img_name)
        # rgb = plt.imread(img_path_name)
        data = h5py.File(img_path_name)
        rgb = data.get('F_color_chart')
        print(rgb)
        # plt.imshow(np.transpose(rgb, [2, 1, 0]).sum(axis=2))
        # plt.colorbar()
        # plt.show()
        # print(rgb.shape)
        # exit()
        # rgb = rgb / 255
        rgb = np.expand_dims(np.transpose(rgb, [0, 1, 2]), axis=0).copy()
        print(rgb.shape)

        img_res1 = reconstruction(rgb, model)
        img_res2 = np.flip(reconstruction(np.flip(rgb, 2).copy(), model), 1)
        img_res3 = (img_res1 + img_res2) / 2

        mat_name = 'recon_' + model_label + img_name.replace('input', '')
        mat_dir = os.path.join(img_path, mat_name)

        save_matv73(mat_dir, var_name, img_res3)
