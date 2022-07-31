import argparse
import pdb

import numpy as np
import torch

import sys
from tqdm import tqdm
from dataloader.dataloader import get_dataloader
from torch.cuda.amp import GradScaler, autocast
from dataloader.transform import clamp, mu_cifar, std_cifar
from torch.autograd import Variable
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from models import *
from utils import *
import wandb
from typing import List, Optional, Tuple, Union, cast
import torch.nn.functional as F
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


def normalize_flatten_features(
    features: Tuple[torch.Tensor, ...],
    eps=1e-10,
) -> torch.Tensor:

    normalized_features: List[torch.Tensor] = []
    for feature_layer in features:
        norm_factor = torch.sqrt(
            torch.sum(feature_layer ** 2, dim=1, keepdim=True)) + eps
        normalized_features.append(
            (feature_layer / (norm_factor *
                              np.sqrt(feature_layer.size()[2] *
                                      feature_layer.size()[3])))
            .view(feature_layer.size()[0], -1)
        )
    return torch.cat(normalized_features, dim=1)


def get_attack(model, inputs, targets_u, y_ori, flat_feat_ori, args):
    upper_limit = ((1 - mu_cifar) / std_cifar).cuda()
    lower_limit = ((0 - mu_cifar) / std_cifar).cuda()

    perturbations = torch.zeros_like(inputs)
    perturbations.uniform_(-0.01, 0.01)
    perturbations.data = clamp(perturbations, lower_limit - inputs, upper_limit - inputs)
    perturbations.requires_grad = True
    for attack_iter in range(args.num_iterations):
        # Decay step size, but increase lambda over time.
        step_size = \
            args.bound * 0.1 ** (attack_iter / args.num_iterations)
        lam = \
            args.lam * 0.1 ** (1 - attack_iter / args.num_iterations)

        if perturbations.grad is not None:
            perturbations.grad.data.zero_()

        inputs_adv = inputs + perturbations

        with autocast(enabled=args.amp):
            logits_adv, feat_adv = model(inputs_adv, adv=True, return_feature=True)
            prob_adv = torch.softmax(logits_adv, dim=-1)
            y_adv = torch.log(torch.gather(prob_adv, 1, targets_u.view(-1, 1)).squeeze(dim=1))

            pip = (normalize_flatten_features(feat_adv) - \
            flat_feat_ori).norm(dim=1).mean()
            constraint = y_ori - y_adv
            loss = -pip + lam * F.relu(constraint - args.bound).mean()

        loss.backward()

        with autocast(enabled=args.amp):
            grad = perturbations.grad.data
            grad_normed = grad / \
                          (grad.reshape(grad.size()[0], -1).norm(dim=1)
                           [:, None, None, None] + 1e-8)
            with torch.no_grad():
                y_after = torch.log(torch.gather(torch.softmax(
                             model(inputs + perturbations - grad_normed * 0.1, adv=True), dim=1),
                             1, targets_u.view(-1, 1)).squeeze(dim=1))
                dist_grads = torch.abs( y_adv - y_after
                             ) / 0.1
                norm = step_size / (dist_grads + 1e-4)
            perturbation_updates = -grad_normed * norm[:, None, None, None]

            perturbations.data = clamp(perturbations + perturbation_updates,
                                       lower_limit - inputs, upper_limit - inputs).detach()

    inputs_adv = (inputs + perturbations).detach()
    return inputs_adv


def parse_args():
    parser = argparse.ArgumentParser(description='Unofficial Implementation of Adversarial Autoaugment')
    parser.add_argument('--load_conf', type=str)
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--batch_size',
                        default=128,
                        type=int)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--bound', default=0.002, type=float, help='bound for adversarial')
    parser.add_argument('--num_iterations', default=5, type=int, help='eps for adversarial')
    parser.add_argument('--lam', default=1, type=float, help='bound for adversarial')
    parser.add_argument('--warmup_adv', default=5, type=int, help='warm up epoch')
    parser.add_argument('--portion', default=0.5, type=float, help='portion for adv')
    parser.add_argument('--alpha', default=0.5, type=float, help='portion for adv')
    args = parser.parse_args()
    return args


def init_ddp(local_rank):
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')


if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)

    conf = load_yaml(args.load_conf)
    name = args.load_conf.split('/')[-1].split('.')[0] + "_%d" % args.seed + '_lpa3_portion{}'.format(args.portion)
    if args.mixup:
        name = name + '_mixup'
    logger = Logger(os.path.join(args.logdir, name))
    num_gpus = torch.cuda.device_count()

    wandb.init(name=name, config=args)

    ## DDP set print_option + initialization
    if args.local_rank > 0:
        sys.stdout = open(os.devnull, 'w')
    init_ddp(args.local_rank)
    print("EXPERIMENT:", args.load_conf.split('/')[-1].split('.')[0])

    train_sampler, train_loader, valid_loader, test_loader = get_dataloader(conf, dataroot='../../data/', split=0,
                                                                            split_idx=0,
                                                                            multinode=(args.local_rank != -1),
                                                                            batch_size=args.batch_size)

    mem_logits = Variable(torch.zeros([len(train_loader.dataset), num_class(conf['dataset'])], dtype=torch.int64, requires_grad=False).cuda() + 1/num_class(conf['dataset']))
    mem_tc = Variable(torch.zeros(len(train_loader.dataset), requires_grad=False).cuda())
    threshold = 1

    model = get_model(conf, args.local_rank, bn_adv_flag=True, bn_adv_momentum=0.01)
    optimizer = SGD(model.parameters(), lr=conf['lr'], momentum=0.9, nesterov=True, weight_decay=conf['weight_decay'])
    scheduler = MultiStepLR(optimizer, [60, 120, 160], gamma=0.2, last_epoch=-1, verbose=False)
    criterion = nn.CrossEntropyLoss()

    if args.amp:
        scaler = GradScaler()

    step = 0
    for epoch in range(conf['epoch']):
        if args.local_rank >= 0:
            train_sampler.set_epoch(epoch)

        model.train()

        train_loss_adv_ori = 0
        train_loss_adv = 0
        train_loss = 0
        train_top1 = 0
        train_top5 = 0

        progress_bar = tqdm(train_loader)
        for idx, (x, label, index) in enumerate(progress_bar):
            optimizer.zero_grad()
            x = x.cuda()
            index = index.cuda()
            label = label.cuda()

            with autocast(enabled=args.amp):
                with torch.no_grad():
                    logits_ori, feat_ori = model(x, adv=True, return_feature=True)
                    _, targets_uadv = torch.max(logits_ori, 1)
                    flat_feat_ori = normalize_flatten_features(feat_ori)
                    prob = torch.softmax(logits_ori, dim=-1)
                    y_ori = torch.log(torch.gather(prob, 1, targets_uadv.view(-1, 1)).squeeze(dim=1))
                    at = F.kl_div(mem_logits[index].log(), prob, reduction='none').mean(dim=1)

            x_adv = get_attack(model, x, targets_uadv, y_ori, flat_feat_ori, args)
            with torch.no_grad():
                l2norm = (x_adv-x).reshape(x.shape[0], -1).norm(dim=1)
            optimizer.zero_grad()

            if args.mixup:
                x, label_a, label_b, lam = mixup_data(x, label)
                x, label_a, label_b = map(Variable, (x, label_a, label_b))

            with autocast(enabled=args.amp):
                pred = model(x)
                loss_ori = mixup_criterion(criterion, pred, label_a, label_b, lam) if args.mixup else criterion(pred, label)

                pred_adv,  feat_adv = model(x_adv, adv=True, return_feature=True)
                mask = ((mem_tc[index]).lt(threshold))
                loss_adv_ori = F.cross_entropy(pred_adv, label)
                loss_adv = (F.cross_entropy(pred_adv, label, reduction='none')*mask).mean()

                mem_tc[index] = 0.01 * mem_tc[index] - 0.99 * at #update memory of time consistency
                mem_logits[index] = prob

                if epoch >= args.warmup_adv:
                    loss = loss_ori + args.alpha * loss_adv
                else:
                    loss = loss_ori

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            top1, top5 = accuracy(pred, label, (1, 5))

            train_loss_adv_ori += reduced_metric(loss_adv_ori.detach(), num_gpus, args.local_rank != -1) / len(train_loader)
            train_loss_adv += reduced_metric(loss_adv.detach(), num_gpus, args.local_rank != -1) / len(train_loader)
            train_loss += reduced_metric(loss_ori.detach(), num_gpus, args.local_rank != -1) / len(train_loader)
            train_top1 += reduced_metric(top1.detach(), num_gpus, args.local_rank != -1) / len(train_loader)
            train_top5 += reduced_metric(top5.detach(), num_gpus, args.local_rank != -1) / len(train_loader)

            progress_bar.set_description(
                'Step: {}. LR : {:.5f}. Epoch: {}/{}. Iteration: {}/{}. Train_Loss : {:.5f}'.format(step,optimizer.param_groups[0]['lr'], epoch,conf['epoch'],idx + 1,len(train_loader),loss.item()))
            step += 1

            y_adv = (torch.gather(pred_adv, 1, label.view(-1, 1)).squeeze(dim=1))
            y_w = (torch.gather(pred, 1, label.view(-1, 1)).squeeze(dim=1))
            l2norm = torch.where(torch.isnan(l2norm), torch.full_like(l2norm, 0), l2norm)

            wandb.log({'y_adv': y_adv.mean().cpu().detach().numpy(),
                       'y_w': y_w.mean().cpu().detach().numpy(),
                       'y_w_select': y_w[mask].mean().cpu().detach().numpy(),
                       'y_adv_select': y_adv[mask].mean().cpu().detach().numpy(),
                       'num': mask.sum().cpu().detach().numpy(),
                       'his_y_adv': wandb.Histogram(y_adv[mask].cpu().detach().numpy(), num_bins=512),
                       'his_y_w': wandb.Histogram(y_w[mask].cpu().detach().numpy(), num_bins=512),
                       'l2_norm': torch.mean(l2norm[mask].cpu().detach()),
                       'l2_norm_his': wandb.Histogram(l2norm[mask].cpu().detach().numpy(), num_bins=512),
                       }, commit=False)

        model.eval()
        scheduler.step()

        _, indices = torch.sort(mem_tc, descending=True)
        kt = (1-args.portion) * len(train_loader.dataset)
        mem_tc_copy = copy.deepcopy(mem_tc)
        threshold = mem_tc_copy[indices[int(kt)]]

        valid_loss = 0.
        valid_top1 = 0.
        valid_top5 = 0.
        cnt = 0.
        with torch.no_grad():
            for idx, (data, label) in enumerate(tqdm(test_loader)):
                b = data.size(0)
                data = data.cuda()
                label = label.cuda()

                pred = model(data)
                loss = criterion(pred, label)

                top1, top5 = accuracy(pred, label, (1, 5))
                valid_loss += reduced_metric(loss.detach(), num_gpus, args.local_rank != -1) * b
                valid_top1 += reduced_metric(top1.detach(), num_gpus, args.local_rank != -1) * b
                valid_top5 += reduced_metric(top5.detach(), num_gpus, args.local_rank != -1) * b
                cnt += b

            valid_loss = valid_loss / cnt
            valid_top1 = valid_top1 / cnt
            valid_top5 = valid_top5 / cnt

        logger.add_dict(
            {
                'train_loss': train_loss,
                'train_top1': train_top1,
                'train_top5': train_top5,
                'valid_loss': valid_loss,
                'valid_top1': valid_top1,
                'valid_top5': valid_top5,
            }
        )

        wandb.log({'train/acc': train_top1,
                   'test/acc': valid_top1,
                   'train/loss_ori': train_loss,
                   'train/loss_adv': train_loss_adv,
                   'train/loss_adv_ori': train_loss_adv_ori,
                   })

        if args.local_rank <= 0:
            logger.save_model(model, epoch)
        logger.info(epoch, ['train_loss', 'train_top1', 'train_top5', 'valid_loss', 'valid_top1', 'valid_top5', ])

        logger.save_logs()