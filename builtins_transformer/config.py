# -*- coding: utf-8 -*-
'''
File:config.py                                         
Time:2025/7/7 20:32
IDE:PyCharm                                     
Author:Barranzi An                                        
email:awc19930818@outlook.com                           
github:https://github.com/La0bALanG                     
Barranzi's Blog:私聊巴郎子要文档链接                  
requirement:(Please describle your requirement here) -->
'''
import torch
from utils import *


class HyperParameters(object):
    '''
    全局超参数配置
    '''

    # data部分相关的超参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 数据存放的根目录
    data_dir = './data/'

    # 三个样本子集的加载路径
    train_dataset_path = './data/data_train.json'
    val_dataset_path = './data/data_val.json'
    test_dataset_path = './data/data_test.json'

    # 固定一个全局的随机种子
    seed = 123

    # 模型相关的超参

    # encoder相关的参数
    encoder_layer = 6 # 表示encoder块中神经网络层的层数
    encoder_dim = 128 # 这里为了方便计算和更快速的训练收敛，我们直接保持token embedding的dim、Transformer hiddem layer的dim、decoder的dim以及位置编码positional encoding的dim一致，即128
    # 其实这个维度，在原始Transformer论文中保持的维度是512，但其实咱们写代码的时候没必要，为啥？太大了，不太好，算力资源不足，搞大了训练时容易OOM

    encoder_drop_prob = 0.1 # 表示encoder块中dropout的正则化系数

    # 字素序列的长度
    graphemes_size = len(graphemes_char2id)

    # encoder的输入最大长度
    encoder_max_len = 30

    # 多头自注意力的注意力头的个数
    n_head = 4 # 原文中其实是8，但这里同样没必要，咱就是一个代码复现的需求，搞清楚Transformer怎么实现的就行了，没必要搞那么复杂

    # feedforward layer的dim维度
    encoder_feed_forward_dim = 1024
    decoder_feed_forward_dim = 1024

    # feedforward layer中加入的dropout正则化系数
    feed_forward_drop_prob = 0.3

    # decoder部分的参数
    decoder_layers = 6 # 表示decoder块中神经网络层的层数

    # decoder dim
    decoder_dim = encoder_dim

    decoder_drop_prob = 0.1 # 表示decoder块中dropout的正则化系数

    # 音素序列的长度
    phonemes_size = len(phonemes_char2id)

    # decoder解码输出的最大步数
    MAX_DECODE_STEP = 50

    # encoder的起始、结束、填充token
    ENCODER_SOS_IDX = graphemes_char2id['<s>']
    ENCODER_EOS_IDX = graphemes_char2id['</s>']
    ENCODER_PAD_IDX = graphemes_char2id['<pad>']

    # decoder的起始、结束、填充token
    DECODER_SOS_IDX = phonemes_char2id['<s>']
    DECODER_EOS_IDX = phonemes_char2id['</s>']
    DECODER_PAD_IDX = phonemes_char2id['<pad>']

    # 训练相关的参数
    batch_size = 128
    init_lr = 1e-4
    epochs = 5
    verbose_step = 100 # 日志记录的步数
    save_step = 500 # 模型保存的步数

    # 梯度grad值的约束，最大为1
    grad_clip_thresh = 1.

HP = HyperParameters()
























