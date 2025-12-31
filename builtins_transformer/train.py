# -*- coding: utf-8 -*-
'''
File:train.py                                         
Time:2025/7/7 20:32                                    
IDE:PyCharm                                     
Author:Barranzi An                                        
email:awc19930818@outlook.com                           
github:https://github.com/La0bALanG                     
Barranzi's Blog:私聊巴郎子要文档链接                  
requirement:(Please describle your requirement here) -->
'''

import os
import torch.optim as optim
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from config import *
from model import *
from utils import *
from dataloaders import *

# 记录日志
logger = SummaryWriter('./logs')

# seed init: Ensure Reproducible Result
# 固定全局随机种子
torch.manual_seed(HP.seed)
torch.cuda.manual_seed(HP.seed)
random.seed(HP.seed)
np.random.seed(HP.seed)


# 验证集函数
def evaluate(model_, devloader, crit):
    model_.eval()  # set evaluation flag
    sum_loss = 0.
    with torch.no_grad():
        for batch in devloader:
            words_idxs, word_len, phoneme_seqs_idxs, phoneme_len = batch
            output_post, attention = model_(words_idxs.to(HP.device), phoneme_seqs_idxs[:, :-1].to(HP.device))
            out = output_post.view(-1, output_post.size(-1))  # [N*seq_len, phoneme_size]
            trg = phoneme_seqs_idxs[:, 1:]
            trg = trg.contiguous().view(-1)  # [N*seq_len, ]
            loss = crit(out.to(HP.device), trg.to(HP.device))
            sum_loss += loss.item()
    model_.train()  # back to training mode
    return sum_loss / len(devloader)


# 模型保存函数
def save_checkpoint(model_, epoch_, optm, checkpoint_path):
    save_dict = {
        'epoch': epoch_,
        'model_state_dict': model_.state_dict(),
        'optimizer_state_dict': optm.state_dict()
    }
    torch.save(save_dict, checkpoint_path)


def train():
    # print('+++++')
    # new model instance
    # 实例化模型对象，加载至cuda
    model = Transformer()
    model = model.to(HP.device)

    # 构建loss，这里直接选用多分类交叉熵loss
    criterion = nn.CrossEntropyLoss(ignore_index=HP.DECODER_PAD_IDX)  # ignore PAD index

    # 优化器采用Adam自适应矩估计
    opt = optim.Adam(model.parameters(), lr=HP.init_lr)

    # 加载训练集数据和验证集数据
    # train dataloader
    trainset = G2PDataset(HP.train_dataset_path)
    train_loader = DataLoader(trainset, batch_size=HP.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)

    # dev datalader(evaluation)
    devset = G2PDataset(HP.val_dataset_path)
    dev_loader = DataLoader(devset, batch_size=HP.batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)

    # 初始化训练轮数和步数
    start_epoch, step = 0, 0

    # 模型进入训练态
    model.train()  # set training flag

    # main loop
    for epoch in range(start_epoch, HP.epochs):
        # print(f'测试：当前epoch:{epoch}')

        # 按batch加载数据
        for batch in train_loader:

            # 拿到一个batch的数据
            words_idxs, word_len, phoneme_seqs_idxs, phoneme_len = batch

            # 梯度归零
            opt.zero_grad()  # gradient clean

            # model进行一次forward，获得预测结果
            output_post, attention = model(words_idxs.to(HP.device), phoneme_seqs_idxs[:, :-1].to(HP.device))
            out = output_post.view(-1, output_post.size(-1))  # [N*seq_len, phoneme_size]

            # 准备训练集中的output真实结果
            trg = phoneme_seqs_idxs[:, 1:]
            trg = trg.contiguous().view(-1)  # [N*seq_len, ]

            # 构建loss
            loss = criterion(out.to(HP.device), trg.to(HP.device))

            # backward反向传播
            loss.backward()  # backward process

            # 约束梯度值的下降
            torch.nn.utils.clip_grad_norm_(model.parameters(), HP.grad_clip_thresh)

            # 优化器更新参数
            opt.step()

            # 一次训练完毕，开始记录日志信息
            logger.add_scalar('Loss/Train', loss, step)

            # 达到迭代步进行一次验证集验证
            if not step % HP.verbose_step:  # evaluate log print
                eval_loss = evaluate(model, dev_loader, criterion)
                logger.add_scalar('Loss/Dev', eval_loss, step)

            # 达到迭代步进行模型保存
            if not step % HP.save_step:  # model save
                model_path = f'model_{epoch}_{step}.pth'
                save_checkpoint(model, epoch, opt, os.path.join('./model_save', model_path))

            step += 1
            logger.flush()
            print(f'Epoch: [{epoch}/{HP.epochs}], step: {step} Train Loss: {loss.item()}, Dev Loss: {eval_loss}')
    logger.close()

train()
























