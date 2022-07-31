import argparse
import pdb

import numpy as np
import torch

import sys
from tqdm import tqdm
from dataloader.dataloader import get_dataloader
from torch.cuda.amp import GradScaler, autocast
from dataloader.transform import parse_policies, MultiAugmentation_WithOrigin, clamp, mu_cifar, std_cifar
from optimizer_scheduler import get_optimizer_scheduler
from models import *
from utils import *
import wandb
from typing import List, Optional, Tuple, Union, cast
import torch.nn.functional as F
from torch.autograd import Variable


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
    parser.add_argument('--load_conf', type = str)
    parser.add_argument('--logdir', type = str)
    parser.add_argument('--M', type = int)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--local_rank', type = int, default = -1)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--bound', default=0.002, type=float, help='bound for adversarial')
    parser.add_argument('--num_iterations', default=5, type=int, help='eps for adversarial')
    parser.add_argument('--lam', default=1, type=float, help='bound for adversarial')
    parser.add_argument('--warmup_adv', default=5, type=int, help='warm up epoch')
    parser.add_argument('--portion', default=0.5, type=float, help='portion for adv')
    parser.add_argument('--batch_size',
                        default=128,
                        type=int)
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
    logger = Logger(os.path.join(args.logdir,args.load_conf.split('/')[-1].split('.')[0] +
                                 "_%d"%args.seed+'portion{}'.format(args.portion)+'lpa3'))
    num_gpus = torch.cuda.device_count()

    wandb.init(name=args.load_conf.split('/')[-1].split('.')[0] + "_%d"%args.seed+'portion{}'.format(args.portion)+'lpa3', config=args)

    ## DDP set print_option + initialization
    if args.local_rank > 0:
        sys.stdout = open(os.devnull, 'w')
    init_ddp(args.local_rank)
    print("EXPERIMENT:",args.load_conf.split('/')[-1].split('.')[0]+ "_%d"%args.seed+'lpa3')
    
    train_sampler, train_loader, valid_loader, test_loader = get_dataloader(conf, dataroot = '../../data/', split = 0, split_idx = 0, multinode = (args.local_rank!=-1), batch_size=args.batch_size)

    mem_logits = Variable(torch.zeros([len(train_loader.dataset.dataset), num_class(conf['dataset'])], dtype=torch.int64, requires_grad=False).cuda() + 1/num_class(conf['dataset']))
    mem_tc = Variable(torch.zeros(len(train_loader.dataset.dataset), requires_grad=False).cuda())
    threshold = 1

    controller = get_controller(conf, args.local_rank)
    model = get_model(conf, args.local_rank, bn_adv_flag=True, bn_adv_momentum=0.01)
    (optimizer, scheduler), controller_optimizer = get_optimizer_scheduler(controller, model, conf)
    criterion = CrossEntropyLabelSmooth(num_classes = num_class(conf['dataset']))

    if args.amp:
        scaler = GradScaler()
    
    step = 0
    for epoch in range(conf['epoch']):
        if args.local_rank >=0:
            train_sampler.set_epoch(epoch)
        
        Lm = torch.zeros(args.M).cuda()
        Lm.requires_grad = False

        model.train()
        controller.train()
        policies, log_probs, entropies = controller(args.M) # (M,2*2*5) (M,) (M,) 
        policies = policies.cpu().detach().numpy()
        parsed_policies = parse_policies(policies)
        
        trfs_list = train_loader.dataset.dataset.transform.transforms 
        trfs_list[2] = MultiAugmentation_WithOrigin(parsed_policies)## replace augmentation into new one

        train_loss_adv_ori = 0
        train_loss_adv = 0
        train_loss = 0
        train_top1 = 0
        train_top5 = 0

        progress_bar = tqdm(train_loader)
        for idx, (x,label,index) in enumerate(progress_bar):
            optimizer.zero_grad()
            x = x.cuda()
            index = index.cuda()
            sx = torch.cat([x[i::args.M+1, ...] for i in range(args.M)])
            wx = x[args.M::args.M+1, ...]
            label = label.cuda()

            with autocast(enabled=args.amp):
                with torch.no_grad():
                    logits_ori, feat_ori = model(wx, adv=True, return_feature=True)
                    _, targets_uadv = torch.max(logits_ori, 1)
                    flat_feat_ori = normalize_flatten_features(feat_ori)
                    prob = torch.softmax(logits_ori, dim=-1)
                    y_w = torch.log(torch.gather(prob, 1, targets_uadv.view(-1, 1)).squeeze(dim=1))
                    at = F.kl_div(mem_logits[index].log(), prob, reduction='none').mean(dim=1)

            x_adv = get_attack(model, wx, targets_uadv, y_w, flat_feat_ori, args)
            optimizer.zero_grad()
            with autocast(enabled=args.amp):
                pred = model(sx)
                pred_adv,  feat_adv = model(x_adv, adv=True, return_feature=True)
                losses = [criterion(pred[i*args.batch_size: (i+1)*args.batch_size], label) for i in range(args.M)]
                loss_ori = torch.mean(torch.stack(losses))

                mask = ((mem_tc[index]).lt(threshold))
                loss_adv_ori = F.cross_entropy(pred_adv, label)
                loss_adv = (F.cross_entropy(pred_adv, label, reduction='none')*mask).mean()

                mem_tc[index] = 0.01 * mem_tc[index] - 0.99 * at #update memory of time consistency
                mem_logits[index] = prob

                if epoch >= args.warmup_adv:
                    loss = loss_ori + loss_adv
                else:
                    loss = loss_ori
            
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            for i,_loss in enumerate(losses):
                Lm[i] += reduced_metric(_loss.detach(), num_gpus, args.local_rank !=-1) / len(train_loader)
            
            top1 = None
            top5 = None
            for i in range(args.M):
                _top1, _top5 = accuracy(pred[i*args.batch_size:(i+1)*args.batch_size], label, (1, 5))
                top1 = top1 + _top1/args.M if top1 is not None else _top1/args.M
                top5 = top5 + _top5/args.M if top5 is not None else _top5/args.M

            train_loss_adv_ori += reduced_metric(loss_adv_ori.detach(), num_gpus, args.local_rank != -1) / len(train_loader)
            train_loss_adv += reduced_metric(loss_adv.detach(), num_gpus, args.local_rank != -1) / len(train_loader)
            train_loss += reduced_metric(loss_ori.detach(), num_gpus, args.local_rank !=-1) / len(train_loader)
            train_top1 += reduced_metric(top1.detach(), num_gpus, args.local_rank !=-1) / len(train_loader)
            train_top5 += reduced_metric(top5.detach(), num_gpus, args.local_rank !=-1) / len(train_loader)
            
            progress_bar.set_description('Step: {}. LR : {:.5f}. Epoch: {}/{}. Iteration: {}/{}. Train_Loss : {:.5f}'.format(step, optimizer.param_groups[0]['lr'], epoch, conf['epoch'], idx + 1, len(train_loader), loss.item()))
            step += 1

            l2norm = (x_adv - wx).reshape(
                wx.shape[0], -1).norm(dim=1)

            wandb.log({'num': mask.sum().cpu().detach().numpy(),
                       'l2_norm': torch.mean(l2norm[mask].cpu().detach()),
                        'l2_norm_his': wandb.Histogram(l2norm[mask].cpu().detach().numpy(),num_bins=512),
                       }, commit=False)

        model.eval()
        controller.train()
        controller_optimizer.zero_grad()
        
        normalized_Lm = (Lm - torch.mean(Lm))/(torch.std(Lm) + 1e-5)
        score_loss = torch.mean(-log_probs * normalized_Lm) # - derivative of Score function
        entropy_penalty = torch.mean(entropies) # Entropy penalty
        controller_loss = score_loss - conf['entropy_penalty'] * entropy_penalty
        
        controller_loss.backward()
        controller_optimizer.step()
        scheduler.step()

        _, indices = torch.sort(mem_tc, descending=True)
        kt = (1-args.portion) * len(train_loader.dataset.dataset)
        mem_tc_copy = copy.deepcopy(mem_tc)
        threshold = mem_tc_copy[indices[int(kt)]]

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
                'controller_loss' : controller_loss.item(),
                'score_loss' : score_loss.item(),
                'entropy_penalty' : entropy_penalty.item(),
                'valid_loss' : valid_loss,
                'valid_top1' : valid_top1,
                'valid_top5' : valid_top5,
                'policies' : parsed_policies,
            }
        )

        wandb.log({'train/acc': train_top1,
                   'test/acc': valid_top1,
                   'train/loss_ori': train_loss,
                   'train/loss_adv': train_loss_adv,
                   'train/loss_adv_ori': train_loss_adv_ori,
                   })

        if args.local_rank <= 0:
            logger.save_model(model,epoch)
        logger.info(epoch,['train_loss','train_top1','train_top5','valid_loss','valid_top1','valid_top5','controller_loss'])
    
        logger.save_logs()