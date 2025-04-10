#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import functional as F
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
import json
import numpy as np
import argparse
import random
from sklearn.metrics import f1_score, average_precision_score
from data.template import config

from utils.utils import (
    create_logger,
    Averager,
    deep_update_dict,
)
from model.model import Discrim
from utils.compute_volume import volume_computation3
from dataset.KS_dataset import KS_dataset
def compute_mAP(outputs, labels):
    y_true = labels.cpu().detach().numpy()
    y_pred = outputs.cpu().detach().numpy()
    AP = []
    for i in range(y_true.shape[1]):
        AP.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return np.mean(AP)
def calculate_belief(alpha, n_classes):


    S = torch.sum(alpha, dim=1, keepdim=True)


    E = alpha - 1


    b = E / S.expand(E.shape)


    u = n_classes / S

    return 10*torch.mean(b,dim=0).detach()


def Alignment(logits_a, logits_v,m=None, beta=0.1):

    p_a = F.softmax(logits_a, dim=-1)
    p_v = F.softmax(logits_v, dim=-1)


    if m is None:
        m = 0.5 * (p_a + p_v)
    else:
        m = F.softmax(m,dim=-1)


    kl_a_m = F.kl_div(p_a.log(), m, reduction='batchmean')
    kl_v_m = F.kl_div(p_v.log(), m, reduction='batchmean')
    js_loss = 0.5 * (kl_a_m + kl_v_m)


    kl_a = F.kl_div(p_a.log(), p_a, reduction='batchmean')
    kl_v = F.kl_div(p_v.log(), p_v, reduction='batchmean')

    return js_loss + beta * (kl_a + kl_v)



def train_discrim(model, train_loader, optimizer, epoch,audio_dim=512,video_dim=512,embed_dim=10,num_labels=28,layers=3,p_y=None):
    tanh = nn.Tanh()
    tl_loss = Averager()
    tl_loss_a = Averager()
    tl_loss_v = Averager()
    tl_pen = Averager()
    criterion = nn.CrossEntropyLoss(reduction='mean').cuda()


    record_names_audio = []
    record_names_visual = []
    for name, param in model.named_parameters():
        if 'mlp' in name:
            continue
        if ('audio' in name):
            record_names_audio.append((name, param))
            continue
        if ('video' in name):
            record_names_visual.append((name, param))
            continue

    for step, sample in enumerate(tqdm(train_loader)):
        image = sample['clip']
        spectrogram = sample['audio']
        y = sample['target']
        image_train = image.float().cuda()
        y = y.cuda()
        y = torch.argmax(y,dim=1)

        spectrogram_train = spectrogram.float().cuda()
        optimizer.zero_grad()
        model.train()
        a_feature, v_feature, av_feature, out_a, out_v, out_av= model([spectrogram_train, image_train])

        loss_a = criterion(out_a, y)
        loss_v = criterion(out_v, y)

        loss_alignment = Alignment(out_a,out_v)

        loss_cls = criterion(0.5*out_a+0.5*out_v,y)





        coef_a = 1
        coef_b = 1




        alpha_audio = F.softplus(out_a) + 1
        alpha_video = F.softplus(out_v) + 1
        alpha_av = F.softplus(out_av) + 1
        belief_audio = calculate_belief(alpha_audio,num_labels)
        belief_video = calculate_belief(alpha_video,num_labels)
        belief_av = calculate_belief(alpha_av,num_labels)

        grads_audio = {}
        grads_visual = {}

        losses = [loss_cls,loss_alignment]
        all_loss = ['supervise', 'unsupervise']
        for idx,loss_type in enumerate(all_loss):
            loss = losses[idx]
            loss.backward(retain_graph=True)
            if (loss_type == 'unsupervise'):
                for tensor_name, param in record_names_visual:
                    if loss_type not in grads_visual.keys():
                        grads_visual[loss_type] = {}
                    grads_visual[loss_type][tensor_name] = param.grad.data.clone()
                    # grads_visual[loss_type][tensor_name] = param.grad
                grads_visual[loss_type]["concat"] = torch.cat(
                    [grads_visual[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_visual])

                for tensor_name, param in record_names_audio:
                    if loss_type not in grads_audio.keys():
                        grads_audio[loss_type] = {}
                    grads_audio[loss_type][tensor_name] = param.grad.data.clone()
                    # grads_audio[loss_type][tensor_name] = param.grad
                grads_audio[loss_type]["concat"] = torch.cat(
                    [grads_audio[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_audio])

            else:
                for tensor_name, param in record_names_audio:
                    if loss_type not in grads_audio.keys():
                        grads_audio[loss_type] = {}
                    grads_audio[loss_type][tensor_name] = param.grad.data.clone()
                    # grads_audio[loss_type][tensor_name] = param.grad
                grads_audio[loss_type]["concat"] = torch.cat([grads_audio[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_audio])
                for tensor_name, param in record_names_visual:
                    if loss_type not in grads_visual.keys():
                        grads_visual[loss_type] = {}
                    grads_visual[loss_type][tensor_name] = param.grad.data.clone()
                    # grads_visual[loss_type][tensor_name] = param.grad
                grads_visual[loss_type]["concat"] = torch.cat([grads_visual[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_visual])

            optimizer.zero_grad()

        volume_a = volume_computation3(grads_visual['unsupervise']['concat'].view(1,-1),grads_visual['supervise']['concat'].view(1,-1),grads_audio['supervise']['concat'].view( 1,-1))
        volume_v = volume_computation3(grads_audio['unsupervise']['concat'].view(1,-1),grads_audio['supervise']['concat'].view(1,-1),grads_visual['supervise']['concat'].view(1,-1))


        mean_volume_a = volume_a.mean()
        mean_volume_v = volume_v.mean()


        loss_total =  loss_cls + loss_alignment
        loss_total.backward()

        tanh = nn.Tanh()
        relu = nn.ReLU()

        ratio_a = mean_volume_a / (mean_volume_a+mean_volume_v)
        ratio_v = 1-ratio_a
        ratio_a = 1-tanh(relu(ratio_a))
        ratio_v = 1-tanh(relu(ratio_v))
        if mean_volume_a < 0.3:
            ratio_a = 1
        if mean_volume_v < 0.3:
            ratio_v = 1
        belief_audio[belief_audio < 0.1] = 0.5
        belief_video[belief_video < 0.1] = 0.5
        belief_av[belief_av<0.1] = 0.5

        for name, param in model.named_parameters():
            if param.grad is not None:

                layer = name

                if ('mlp' in layer):
                    if 0<epoch <= 80:
                        if ('audio' in layer):
                            if len(param.grad.size()) > 1:
                                param.grad *= belief_audio.unsqueeze(-1)
                            else:
                                param.grad *= belief_audio
                        elif ('video' in layer):

                            if len(param.grad.size()) > 1:
                                param.grad *= belief_video.unsqueeze(-1)
                            else:
                                param.grad *= belief_video

                    continue
                if epoch < 100:
                    if ('audio' in layer):

                        new_grad = ratio_a * grads_audio['unsupervise'][name] + grads_audio['supervise'][name]


                        param.grad = new_grad *4

                    if ('video' in layer):
                        new_grad = ratio_v * 2* grads_visual['unsupervise'][
                            name] + grads_visual['supervise'][name]



                        param.grad = new_grad * 4



        optimizer.step()
        tl_loss.add(loss_cls.item())
        loss_a = torch.FloatTensor([0]).cuda()
        loss_v = torch.FloatTensor([0]).cuda()
        tl_loss_a.add(loss_a.item())
        tl_loss_v.add(loss_v.item())
    losses_av = tl_loss.item()
    losses_a = tl_loss_a.item()
    losses_v = tl_loss_v.item()
    return model, losses_a, losses_v, losses_av




def val(epoch,val_loader,models,logger):
    models.eval()
    pred_list = []
    pred_list_a = []
    pred_list_v = []
    label_list = []
    soft_pred = []
    soft_pred_a = []
    soft_pred_v = []
    one_hot_label = []
    score_a = 0.0
    score_v = 0.0
    with torch.no_grad():
        for step,sample in enumerate(tqdm(val_loader)):
            y = sample['target']
            image = sample['clip']
            spectrogram = sample['audio']
            label_list = label_list + torch.argmax(y, dim=1).tolist()
            one_hot_label = one_hot_label + y.tolist()
            image = image.cuda()
            y = y.cuda()

            spectrogram = spectrogram.float().cuda()

            a_feature, v_feature, av_feature, a_logits, v_logits, av_logits = models([spectrogram, image])

            soft_pred_a = soft_pred_a + (F.softmax(a_logits, dim=1)).tolist()
            soft_pred_v = soft_pred_v + (F.softmax(v_logits, dim=1)).tolist()

            soft_pred = soft_pred + (F.softmax(0.5*a_logits+0.5*v_logits, dim=-1)).tolist()
            pred_a = (F.softmax(a_logits, dim=1)).argmax(dim=1)
            pred_v = (F.softmax(v_logits, dim=1)).argmax(dim=1)

            pred = (F.softmax(0.5*a_logits+0.5*v_logits, dim=1)).argmax(dim=1)
            pred_list = pred_list + pred.tolist()
            pred_list_a = pred_list_a + pred_a.tolist()
            pred_list_v = pred_list_v + pred_v.tolist()
            f1 = f1_score(label_list, pred_list, average='macro')
            f1_a = f1_score(label_list, pred_list_a, average='macro')
            f1_v = f1_score(label_list, pred_list_v, average='macro')
            correct = sum(1 for x, y in zip(label_list, pred_list) if x == y)
            correct_a = sum(1 for x, y in zip(label_list, pred_list_a) if x == y)
            correct_v = sum(1 for x, y in zip(label_list, pred_list_v) if x == y)
            acc = correct / len(label_list)
            acc_a = correct_a / len(label_list)
            acc_v = correct_v / len(label_list)
            mAP = compute_mAP(torch.Tensor(soft_pred), torch.Tensor(one_hot_label))
            mAP_a = compute_mAP(torch.Tensor(soft_pred_a), torch.Tensor(one_hot_label))
            mAP_v = compute_mAP(torch.Tensor(soft_pred_v), torch.Tensor(one_hot_label))

        logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logger.info((
                        'Epoch {epoch:d}: f1:{f1:.4f},acc:{acc:.4f},mAP:{mAP:.4f},f1_a:{f1_a:.4f},acc_a:{acc_a:.4f},mAP_a:{mAP_a:.4f},f1_v:{f1_v:.4f},acc_v:{acc_v:.4f},mAP_v:{mAP_v:.4f}').format(
            epoch=epoch, f1=f1, acc=acc, mAP=mAP,
            f1_a=f1_a, acc_a=acc_a, mAP_a=mAP_a,
            f1_v=f1_v, acc_v=acc_v, mAP_v=mAP_v))
        return acc, score_a, score_v

if __name__ == '__main__':
    # ----- LOAD PARAM -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str,default='./data/ks.json')
    parser.add_argument('--dataset',default='KS')
    args = parser.parse_args()
    cfg = config

    with open(args.config, "r") as f:
        exp_params = json.load(f)

    cfg = deep_update_dict(exp_params, cfg)

    # ----- SET SEED -----
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpu_id']
    # ----- SET LOGGER -----
    local_rank = cfg['train']['local_rank']
    logger, log_file, exp_id = create_logger(cfg, local_rank)

    # ----- SET DATALOADER -----
    train_dataset = KS_dataset(cfg,mode='train')
    test_dataset = KS_dataset(cfg,mode='test')


    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['train']['num_workers'], pin_memory=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg['test']['batch_size'], shuffle=False,
                             num_workers=cfg['test']['num_workers'], pin_memory=True)

    val_batch = next(iter(train_loader))

# 计算类别先验分布

    # ----- MODEL -----
    input_dim=512
    audio_dim = video_dim = input_dim
    hidden_dim=512
    embed_dim = 10
    layers = 3
    num_labels = cfg['setting']['num_class']
    total_samples = 0

    discrim = Discrim(input_dim,hidden_dim,num_labels,cfg).cuda()



    lr_adjust = config['train']['optimizer']['lr']

    optimizer_discrim = optim.SGD(discrim.parameters(), lr=lr_adjust,
                                  momentum=config['train']['optimizer']['momentum'],
                                  weight_decay=config['train']['optimizer']['wc'])

    scheduler = optim.lr_scheduler.StepLR(optimizer_discrim, config['train']['lr_scheduler']['patience'], 0.1)
    acc_best = 0
    for epoch in range(cfg['train']['epoch_dict']):
        logger.info(('Epoch {epoch:d} is pending...').format(epoch=epoch))

        scheduler.step()
        discrim,losses_a,losses_v,losses_av = train_discrim(discrim,train_loader,optimizer_discrim,epoch,p_y=None,num_labels=num_labels)
        logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logger.info(('Epoch:{epoch:d},loss_a:{loss_a:.3f},loss_v:{loss_v:.3f},loss_av:{loss_av:.3f},penalty:{penalty:.3f}').format(epoch=epoch, loss_a=losses_a, loss_v=losses_v, loss_av=losses_av,penalty=penalty))

        acc, v_a, v_v = val(epoch, test_loader, discrim, logger)

