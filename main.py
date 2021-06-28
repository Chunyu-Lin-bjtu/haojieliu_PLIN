import argparse
import os
import time
import math
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim
import torch.utils.data

from dataloaders.kitti_loader import load_calib, oheight, owidth, input_options, KittiDepth
import modelresnetunet as modelnet
from modelresnetunet import DepthCompletionNet
import sys

from metrics import AverageMeter, Result
import criteria
import helper
from inverse_warp import Intrinsics, homography_from
from matplotlib import pyplot
import cv2
from tensorboardX import SummaryWriter
from liteflownet.run import Network
import flowtoimage as fl
import matplotlib.pyplot as plt
import numpy as np
import flow_util

parser = argparse.ArgumentParser(description='Sparse-to-Dense')
parser.add_argument('-w', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: 11)')
parser.add_argument('--epochsample', default=40000, type=int, metavar='N',
                    help='number of total epochs to run (default: 11)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-c', '--criterion', metavar='LOSS', default='l2',
                    choices=criteria.loss_names,
                    help='loss function: | '.join(criteria.loss_names) + ' (default: l2)')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate (default 1e-5)')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-i','--input', type=str, default='gd',
                    choices=input_options, help='input: | '.join(input_options))
parser.add_argument('-l','--layers', type=int, default=34,
                    help='use 16 for sparse_conv; use 18 or 34 for resnet')
parser.add_argument('--pretrained', action="store_true",
                    help='use ImageNet pre-trained weights')
# parser.add_argument('--val', type=str, default="select",
#                     choices= ["select","full"], help='full or select validation set')
parser.add_argument('--val', type=str, default="full",
                    choices= ["select","full"], help='full or select validation set')
parser.add_argument('--jitter', type=float, default=0.1,
                    help = 'color jitter for images')
parser.add_argument('--rank-metric', type=str, default='rmse',
                    choices=[m for m in dir(Result()) if not m.startswith('_')],
                    help = 'metrics for which best result is sbatch_datacted')
parser.add_argument('-m', '--train-mode', type=str, default="dense",
                    choices = ["dense", "sparse", "photo", "sparse+photo", "dense+photo"],
                    help = 'dense | sparse | photo | sparse+photo | dense+photo')
parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()
args.use_pose = ("photo" in args.train_mode)
# args.pretrained = not args.no_pretrained
args.result = os.path.join('.', 'results_experiment_all')
# args.use_rgb = ('rgb' in args.input) or args.use_pose

args.use_rgb = True
args.use_d = 'd' in args.input
args.use_g = 'g' in args.input
if args.use_pose:
    args.w1, args.w2 = 0.1, 0.1
else:
    args.w1, args.w2 = 0, 0
print(args)

cont =0
writer = SummaryWriter('logs')
# define loss functions
depth_criterion = criteria.MaskedMSELoss() if (args.criterion == 'l2') else criteria.MaskedL1Loss()

if args.use_pose:
    # hard-coded KITTI camera intrinsics
    K = load_calib()
    fu, fv = float(K[0,0]), float(K[1,1])
    cu, cv = float(K[0,2]), float(K[1,2])
    kitti_intrinsics = Intrinsics(owidth, oheight, fu, fv, cu, cv).cuda()


moduleNetwork = Network().cuda().eval()
FlowBackWarp = modelnet.backWarp(1216, 256, device)
# moduleNetwork = Network().to(device).eval()
def flownet(tensorPreprocessedFirst, tensorPreprocessedSecond,mode):
    # torch.set_grad_enabled(False)
    intWidth = 1216
    intHeight = 256

    # tensorPreprocessedFirst = tensorFirst.cuda().view(1, 3, intHeight, intWidth)
    # tensorPreprocessedSecond = tensorSecond.cuda().view(1, 3, intHeight, intWidth)
    with torch.no_grad():
        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

        tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        # nn.Upsample
        
        tensorFlow = torch.nn.functional.interpolate(input=moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

        tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)
        # if mode == 'train':
        #     torch.set_grad_enabled(True)
    return tensorFlow[:, :, :, :]


def train(mode, args, loader, model, optimizer, logger, epoch):
    global cont
    block_average_meter = AverageMeter()
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]

    # switch to appropriate mode
    assert mode in ["train", "val", "eval", "test_prediction", "test_completion"], \
        "unsupported mode: {}".format(mode)

    model.train()
    lr = helper.adjust_learning_rate(args.lr, optimizer, epoch)
    print('len(loader)=',len(loader))

    step = 0
    for i, batch_data in enumerate(loader):
        start = time.time()
        # batch_data = {key:val.cuda() for key,val in batch_data.items() if val is not None}
        batch_data = {key:val.to(device) for key,val in batch_data.items() if val is not None}
        if len(loader)-step <5:
            break
        if step == args.epochsample:
            break 

        rgb1 = batch_data['rgb1']/255.0
        rgb = batch_data['rgb']/255.0
        rgb3 = batch_data['rgb3']/255.0

        #warp flow
        F_t_0= flownet(rgb, rgb1,mode) #1 2 240 1216
        F_t_1= flownet(rgb, rgb3,mode) #1 2 240 1216

        g_I0_F_t_0 = FlowBackWarp(batch_data['d'], F_t_0)
        g_I1_F_t_1 = FlowBackWarp(batch_data['d3'], F_t_1)
        #warp end

        gt = batch_data['gt']
        data_time = time.time() - start
        start = time.time()
        
        pred , coarse = model(batch_data['d'],batch_data['d3'],F_t_0, F_t_1, g_I1_F_t_1, g_I0_F_t_0,batch_data['rgb'])  

        depth_loss= 0
        if mode == 'train':
            # Loss 1: the direct depth supervision from ground truth label
            # mask=1 indicates that a pixel does not ground truth labels
            if 'sparse' in args.train_mode:
                depth_loss = depth_criterion(pred, batch_data['d2'])
                mask = (batch_data['d2'] < 1e-3).float()
            elif 'dense' in args.train_mode:
                depth_loss = depth_criterion(pred, gt)
                depth_loss_coarse = depth_criterion(coarse, gt)
                mask = (gt < 1e-3).float()

            if step % 100 == 0:
                print('1',depth_loss)
                print('2',depth_loss_coarse)
            # backprop
            loss = depth_loss +depth_loss_coarse*0.1 

            writer.add_scalar('loss', loss, cont)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            cont=cont+1

        gpu_time = time.time() - start
        # measure accuracy and record loss
        with torch.no_grad():
            mini_batch_size = next(iter(batch_data.values())).size(0)
            result = Result()
            if mode != 'test_prediction' and mode != 'test_completion':
                result.evaluate(pred.data, gt.data)
           
            [m.update(result, gpu_time, data_time, mini_batch_size) for m in meters]
            logger.conditional_print(mode, i, epoch, lr, len(loader), block_average_meter, average_meter)
            # logger.conditional_save_img_comparison(mode, i, batch_data, pred, epoch)
        step=step+1  

    avg = logger.conditional_save_info(mode, average_meter, epoch)
    is_best = logger.rank_conditional_save_best(mode, avg, epoch)
    if is_best and not (mode == "train"):
        logger.save_img_comparison_as_best(mode, epoch)
    logger.conditional_summarize(mode, avg, is_best)
    writer.close()
    print('-------here------')


def val(mode, args, loader, model, logger, epoch):

    block_average_meter = AverageMeter()
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]
    model.eval()       
    lr = 0
    print('len(loader)=',len(loader))
    step = 0 
    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            start = time.time()
            # batch_data = {key:val.cuda() for key,val in batch_data.items() if val is not None}
            batch_data = {key:val.to(device) for key,val in batch_data.items() if val is not None}
            if len(loader)-step <5:
                break
            if step == args.epochsample:
                break 
            rgb1 = batch_data['rgb1']/255.0
            rgb  = batch_data['rgb']/255.0
            rgb3 = batch_data['rgb3']/255.0
            # pred_flow = flownet(rgb1, rgb3,mode) #1 2 240 1216
           
            #warp flow
            F_t_0= flownet(rgb, rgb1,mode) #1 2 240 1216
            F_t_1= flownet(rgb, rgb3,mode) #1 2 240 1216

            g_I0_F_t_0 = FlowBackWarp(batch_data['d'], F_t_0)
            g_I1_F_t_1 = FlowBackWarp(batch_data['d3'], F_t_1)
            #warp end
            gt = batch_data['gt']
            data_time = time.time() - start
            start = time.time()
            pred , coarse = model(batch_data['d'],batch_data['d3'],F_t_0, F_t_1, g_I1_F_t_1, g_I0_F_t_0,batch_data['rgb'])  
            gpu_time = time.time() - start
            # measure accuracy and record loss
            step=step+1 
           
            mini_batch_size = next(iter(batch_data.values())).size(0)
            result = Result()       
            result.evaluate(pred.data, gt.data)        
            [m.update(result, data_time, mini_batch_size) for m in meters]
            logger.conditional_print(mode, i, epoch, lr, len(loader), block_average_meter, average_meter)
            logger.conditional_save_img_comparison(mode, i, batch_data, pred, epoch)
           

        avg = logger.conditional_save_info(mode, average_meter, epoch)
        is_best = logger.rank_conditional_save_best(mode, avg, epoch)
        if is_best and not (mode == "train"):
            logger.save_img_comparison_as_best(mode, epoch)
        logger.conditional_summarize(mode, avg, is_best)
        
        return avg, is_best


def main():
    global args
    checkpoint = None
    is_eval = False
    if args.evaluate:
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}'".format(args.evaluate))
            checkpoint = torch.load(args.evaluate)
            args = checkpoint['args']
            is_eval = True
            print("=> checkpoint loaded.")
        else:
            print("=> no model found at '{}'".format(args.evaluate))
            return
    elif args.resume: # optionally resume from a checkpoint
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']+1
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # # print("=> creating model and optimizer...")
  
    # model = DepthCompletionNet(args)
    #    #lhj one gpu
    # if torch.cuda.device_count()>1:
    #     model = torch.nn.DataParallel(model)
    # model = DepthCompletionNet(args).to(device)

    print("=> creating model and optimizer...")
    device_ids = [0]
    model = DepthCompletionNet(args).to(device)
    print("=> model transferred to multi-GPU.")
    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model,device_ids=device_ids)

    model_named_params = [p for _,p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay)

    print("=> model and optimizer created.")
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> checkpoint state loaded.")

    # Data loading code
    print("=> creating data loaders ...")
    if not is_eval:
        print("KittiDepth('train', args)")
        train_dataset = KittiDepth('train', args)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers, pin_memory=True, sampler=None)
    
    val_dataset = KittiDepth('val', args)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True) # set batch size to be 1 for validation
    print("=> data loaders created.")

    # create backups and results folder
    logger = helper.logger(args)
    if checkpoint is not None:
        logger.best_result = checkpoint['best_result']
    print("=> logger created.")

    if is_eval:
        result, is_best = val("val", args, val_loader, model, logger, checkpoint['epoch'])
        return
    # main loop
    for epoch in range(args.start_epoch, args.epochs):
        print("=> starting training epoch {} ..".format(epoch))
        train("train", args, train_loader, model, optimizer, logger, epoch) # train for one epoch
    
        result, is_best = val("val", args, val_loader, model, logger, epoch) # evaluate on validation set

        helper.save_checkpoint({ # save checkpoint
            'epoch': epoch,
            # 'model': model.module.state_dict(),
            'model': model.state_dict(),
            'best_result': logger.best_result,
            'optimizer' : optimizer.state_dict(),
            'args' : args,
        }, is_best, epoch, logger.output_directory)

if __name__ == '__main__':
    main()

