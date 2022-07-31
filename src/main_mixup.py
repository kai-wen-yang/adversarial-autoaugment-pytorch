import argparse
import pdb

import numpy as np
import torch

import sys
from tqdm import tqdm
from dataloader.dataloader import get_dataloader
from torch.cuda.amp import GradScaler, autocast
from torch.autograd import Variable
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR,CosineAnnealingLR
from models import *
from utils import *
import wandb
from torch import nn


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def parse_args():
    parser = argparse.ArgumentParser(description='Unofficial Implementation of Adversarial Autoaugment')
    parser.add_argument('--load_conf', type = str)
    parser.add_argument('--logdir', type = str)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--local_rank', type = int, default = -1)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--batch_size',
                        default=128,
                        type=int)
    parser.add_argument('--mixup', action='store_true')
    args = parser.parse_args()
    return args


def init_ddp(local_rank):
    if local_rank !=-1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl',init_method='env://')


if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)

    conf = load_yaml(args.load_conf)
    name = args.load_conf.split('/')[-1].split('.')[0] + "_%d"%args.seed+'_normal_'
    if args.mixup:
        name = name + 'mixup'
    logger = Logger(os.path.join(args.logdir, name))
    num_gpus = torch.cuda.device_count()

    wandb.init(name=name, config=args)

    ## DDP set print_option + initialization
    if args.local_rank > 0:
        sys.stdout = open(os.devnull, 'w')
    init_ddp(args.local_rank)
    print("EXPERIMENT:", args.load_conf.split('/')[-1].split('.')[0])
    
    train_sampler, train_loader, valid_loader, test_loader = get_dataloader(conf, dataroot = '../../data/', split = 0, split_idx = 0, multinode = (args.local_rank!=-1), batch_size=args.batch_size)

    model = get_model(conf, args.local_rank)
    optimizer = SGD(model.parameters(), lr=conf['lr'], momentum=0.9, nesterov=True, weight_decay=conf['weight_decay'])
    scheduler = MultiStepLR(optimizer, [60, 120, 160], gamma=0.2, last_epoch=-1, verbose=False)
    criterion = nn.CrossEntropyLoss()

    if args.amp:
        scaler = GradScaler()
    
    step = 0
    for epoch in range(conf['epoch']):
        if args.local_rank >=0:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0
        train_top1 = 0
        train_top5 = 0
        
        progress_bar = tqdm(train_loader)
        for idx, (x, label, _) in enumerate(progress_bar):
            optimizer.zero_grad()
            x = x.cuda()
            label = label.cuda()

            if args.mixup:
                x, label_a, label_b, lam = mixup_data(x, label)
                x, label_a, label_b = map(Variable, (x, label_a, label_b))

            with autocast(enabled=args.amp):
                pred = model(x)
                loss = mixup_criterion(criterion, pred, label_a, label_b, lam) if args.mixup else criterion(pred, label)
            
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            top1, top5 = accuracy(pred, label, (1, 5))

            train_loss += reduced_metric(loss.detach(), num_gpus, args.local_rank !=-1) / len(train_loader)
            train_top1 += reduced_metric(top1.detach(), num_gpus, args.local_rank !=-1) / len(train_loader)
            train_top5 += reduced_metric(top5.detach(), num_gpus, args.local_rank !=-1) / len(train_loader)
            
            progress_bar.set_description('Step: {}. LR : {:.5f}. Epoch: {}/{}. Iteration: {}/{}. Train_Loss : {:.5f}'.format(step, optimizer.param_groups[0]['lr'], epoch, conf['epoch'], idx + 1, len(train_loader), loss.item()))
            step += 1

        model.eval()
        scheduler.step()
        
        valid_loss = 0.
        valid_top1 = 0.
        valid_top5 = 0.
        cnt = 0.
        with torch.no_grad():
            for idx, (data,label) in enumerate(tqdm(test_loader)):
                b = data.size(0)
                data = data.cuda()
                label = label.cuda()
                
                pred = model(data)
                loss = criterion(pred,label)

                top1, top5 = accuracy(pred, label, (1, 5))
                valid_loss += reduced_metric(loss.detach(), num_gpus, args.local_rank !=-1) *b 
                valid_top1 += reduced_metric(top1.detach(), num_gpus, args.local_rank !=-1) *b
                valid_top5 += reduced_metric(top5.detach(), num_gpus, args.local_rank !=-1) *b 
                cnt += b
            
            valid_loss = valid_loss / cnt
            valid_top1 = valid_top1 / cnt
            valid_top5 = valid_top5 / cnt
            
        logger.add_dict(
            {
                'train_loss' : train_loss,
                'train_top1' : train_top1,
                'train_top5' : train_top5,
                'valid_loss' : valid_loss,
                'valid_top1' : valid_top1,
                'valid_top5' : valid_top5,
            }
        )

        wandb.log({'train/acc': train_top1,
                   'test/acc': valid_top1,
                   'train/loss_ori': train_loss,
                   })

        if args.local_rank <= 0:
            logger.save_model(model,epoch)
        logger.info(epoch,['train_loss','train_top1','train_top5','valid_loss','valid_top1','valid_top5',])
    
        logger.save_logs()