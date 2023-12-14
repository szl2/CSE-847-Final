from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import os
import time
import scipy.io as sio
import math
from datetime import datetime

from dataset import DatasetFromHdf5
from resblock import resblock, conv_relu_res_relu_block
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss
from loss import sam_mae, sam_mrae, sam, mrae


'''loss function 선택'''
def choose_loss(name):
    if name == 'mrae':
        return mrae
    elif name == 'sam':
        return sam
    elif name == 'sam+mrae':
        return sam_mrae
    elif name == 'sam+mae':
        return sam_mae


def main():
    # iterate whole model train process
    tot_train_num = 1
    for i in range(tot_train_num):
        print('Model Train {}th'.format(i + 1))
        model_train()


def model_train():
    cudnn.benchmark = True

    '''Change parameter'''
    # model & data set parameter
    layer = 14
    input_channel = 20
    output_channel = 100
    loss_func = 'sam+mae'
    headers = ['offset_black_val-1-2-3-4', 'offset_black_val-5-6-7-8',
               'offset_black_val-9-10-11-12', 'offset_black_val-13-14-15-16',
               'offset_black_val-17-18-19-20', 'offset_black_val-21-22-23-b']
    k = len(headers)
    data_path = '../data/hdf5_data/'
    train_set_list = [f'{data_path}train_{h}_input+chann+{input_channel}.h5' for h in headers]
    valid_set_list = [f'{data_path}valid_{h}_input+chann+{input_channel}.h5' for h in headers]
    headers = [f'{h}_{loss_func}' for h in headers]
    drop = 0  # if 0, there is no dropout, in this model don't apply

    # Dataset
    train_data = []
    valid_data = []
    for i in range(k):
        train_data.append(DatasetFromHdf5(train_set_list[i]))
        print(len(train_data[i]))
        valid_data.append(DatasetFromHdf5(valid_set_list[i]))
        print(len(valid_data[i]))

    # device count & verify cuda run
    # print(torch.cuda.device_count())
    # print(torch.cuda.is_available())

    # Data Loader (Input Pipeline)
    per_iter_time = len(train_data[0])
    train_data_loaders = []
    val_data_loaders = []
    for i in range(k):
        train_data_loaders.append(
            DataLoader(dataset=train_data[i], num_workers=3, batch_size=64, shuffle=True, pin_memory=True)
        )
        val_data_loaders.append(
            DataLoader(dataset=valid_data[i], num_workers=0, batch_size=1, shuffle=False, pin_memory=True)
        )

    """ ERROR : 특정 시점 이후 loss nan 발생
    원인 함수 찾기 : torch.autograd 함수 중에 NaN loss가 발생했을 경우 원인을 찾아주는 함수
    https://ocxanc.tistory.com/54
    """
    # torch.autograd.set_detect_anomaly(True)
    model_dir = './models/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for i in range(k):
        # train parameter
        start_epoch = 0
        end_epoch = 300
        init_lr = 0.0002
        iteration = 0
        record_test_loss = 1000
        criterion = choose_loss(loss_func)  # from loss.py - sam_mae, sam_mrae, sam, mrae
        test_criterion = criterion

        # Model
        model = resblock(conv_relu_res_relu_block, layer, input_channel, output_channel, drop)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        if torch.cuda.is_available():
            model.cuda()

        # Parameters, Loss and Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-09, weight_decay=0)

        # Resume
        resume_file = ''
        if resume_file:
            if os.path.isfile(resume_file):
                print("=> loading checkpoint '{}'".format(resume_file))
                checkpoint = torch.load(resume_file)
                start_epoch = checkpoint['epoch']
                iteration = checkpoint['iter']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])

        header = headers[i]
        model_path = model_dir + header + datetime.now().strftime('_%y%m%d_%H-%M-%S/')
        os.makedirs(model_path)

        loss_csv = open(os.path.join(model_path, 'loss.csv'), 'w+')
        loss_csv.write('epoch,iteration,epoch_time,lr,train_loss,test_loss\n')

        log_dir = os.path.join(model_path, 'train.log')
        logger = initialize_logger(log_dir)

        train_data_loader = train_data_loaders[i]
        val_data_loader = val_data_loaders[i]
        print("<K-Fold %d>" % i)

        for epoch in range(start_epoch + 1, end_epoch):
            start_time = time.time()
            train_loss, iteration, lr = train(train_data_loader, model, criterion, optimizer, iteration, init_lr
                                              , epoch, end_epoch, max_iter=per_iter_time * end_epoch)

            test_loss = validate(val_data_loader, model, test_criterion)

            # Save model
            if test_loss < record_test_loss or epoch == end_epoch or epoch == end_epoch / 2:
                save_checkpoint(model_path, epoch, iteration, model, optimizer, layer
                                , input_channel, output_channel, header, test_loss)
                if test_loss < record_test_loss:
                    record_test_loss = test_loss

            # print loss
            end_time = time.time()
            epoch_time = end_time - start_time
            print("Epoch [%d], Iter[%d], Time:%.9f, Train Loss: %.9f Test Loss: %.9f, learning rate:"
                  % (epoch, iteration, epoch_time, train_loss, test_loss), lr)

            # save loss
            record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss)
            logger.info("Epoch [%d], Iter[%d], Time:%.9f, Train Loss: %.9f Test Loss: %.9f, learning rate:"
                        % (epoch, iteration, epoch_time, train_loss, test_loss) + ' {}'.format(lr))


# Training
def train(train_data_loader, model, criterion, optimizer, iteration, init_lr, epoch, end_epoch, max_iter=1e8):
    losses = AverageMeter()
    for i, (images, labels) in enumerate(train_data_loader.dataset):
        labels = labels.cuda(non_blocking=True)
        images = images.cuda(non_blocking=True)
        images = Variable(images)
        labels = Variable(labels)

        # Decaying Learning Rate

        lr = poly_lr_scheduler(optimizer, init_lr, iteration, epoch, end_epoch, max_iter=max_iter)
        iteration = iteration + 1

        # Forward + Backward + Optimize       
        output = model(images)

        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()

        #  record loss
        losses.update(loss.data)

    return losses.avg, iteration, lr


# Validate
def validate(val_loader, model, criterion):
    model.eval()
    losses = AverageMeter()

    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            input_var = Variable(input)
            target_var = Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            #  record loss
            losses.update(loss.data)

    return losses.avg


# Learning rate
def poly_lr_scheduler(optimizer, init_lr, iteraion, epoch, end_epoch,
                      lr_decay_iter=1, max_iter=100):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer

    lr = init_lr * (1 + math.cos(epoch * math.pi / end_epoch)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


if __name__ == '__main__':
    main()
