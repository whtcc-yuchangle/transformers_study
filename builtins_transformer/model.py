# -*- coding: utf-8 -*-
'''
File:model.py                                         
Time:2025/7/7 20:32                                    
IDE:PyCharm                                     
Author:Barranzi An                                        
email:awc19930818@outlook.com                           
github:https://github.com/La0bALanG                     
Barranzi's Blog:私聊巴郎子要文档链接                  
requirement:(Please describle your requirement here) -->
'''
import torch
from modelscope.models.cv.table_recognition.modules.lore_processor import attention_score
from sympy.physics.units import energy
from torch import nn as nn

from config import HP
from utils import *
import torch.nn.functional as F
import math


# 1.先构建最简单的绝对位置编码类
class PositionalEncoding(nn.Module):
    '''
    绝对位置编码类
    '''
    def __init__(self, d_model, max_len=10000):
        '''
        初始化函数
        :param d_model:token embedding的dim
        :param max_len: 输入数据的token id序列的最大长度
        '''

        super().__init__()

        #                    torch.arange是生成一个一维张量tensor，给定序列的起始值，结束值和步长，这里的步长为2，统一以偶数位为步长，奇数位2i + 1
        #                                             .float是将张量中的值统一转换为float类型
        # 这里就相当于是e的{2i * 1 / d_model * log 10000}
        # ...这特么不就是脱裤子放屁——多此一举么？其实依据指数的相关变换过程来表达这个式子，在数学上是比较符合理论的。当然了。我也想装个比。
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.) / d_model))

        # 下一步，准备pos这个量，即位置的索引值
        position = torch.arange(max_len).unsqueeze(1) # 这里的unsqueeze(1)是将pos这个张量的维度从一维扩展到二维，即[max_len] -> [max_len,1]

        # 初始化pe向量
        pe = torch.zeros(1,max_len,d_model) # 这里的pe向量的维度为[1,max_len,d_model]，即[batch_size,max_len,d_model]

        # 偶数位的计算
        pe[0,:,0::2] = torch.sin(position * div_term)
        # 奇数位的计算
        pe[0,:,1::2] = torch.cos(position * div_term)

        # 将pe注册为全局变量
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        前向传播函数
        :param x:
        :return:
        '''
        # 这里的x的维度为[batch_size,max_len,d_model]
        # 这里的pe的维度为[1,max_len,d_model]
        # 这里的pe的维度与x的维度是一致的，因此可以直接相加
        # 这里就是让输入的embedding向量与计算得出的绝对位置编码向量进行逐元素相加即可
        x = x + self.pe[:,:x.size(1),:]
        return x

# 实现多头注意力模块；注意：这里是类型的统一封装过程，实则在实例化时需要创建为不同类型的注意力层
class MutiHeadAttentionLayer(nn.Module):
    '''
    多头注意力模块
    '''
    def __init__(self, hid_dim, nhead):
        '''
        初始化方法
        :param hid_dim: encoder或decoder的维度
        :param nhead: 注意力头的头数
        '''

        super().__init__()

        # 1.先指定encoder或decoder的维度
        self.hid_dim = hid_dim

        # 2.指定注意力头的头数
        self.nhead = nhead # 注意：这里的头数虽然是任意的，但要求必须是hid_dim的整数倍，否则无法分出整数个头
        # e.g. 如果hid_dim = 128,则头数只能给为偶数，否则如果给一个奇数，那就除不尽了

        # 所以这里为了确保万无一失，最好添加一个断言
        assert not self.hid_dim % self.nhead # 这个断言就表示：这里必须返回一个True，才继续进行后续操作，否则直接报错

        # 3.结合隐藏层的维度以及注意力头的数量，最终计算出每一个注意力头的输入维度
        self.head_dim = hid_dim // self.nhead

        # 4.按照论文所示，QKV使用三个相同的linear来封装
        self.fc_q = nn.Linear(self.hid_dim,self.hid_dim)
        self.fc_k = nn.Linear(self.hid_dim,self.hid_dim)
        self.fc_v = nn.Linear(self.hid_dim,self.hid_dim)

        # 5.因为你做的是多头注意力，还需要最后一个用于投影缩放的投影层
        self.fc_o = nn.Linear(self.hid_dim,self.hid_dim)

        # 6.注册变量 - 缩放因子：hid_dim其实就是那个d_model，这里就相当于是先准备好根号下的d_model，方便后续的使用
        self.register_buffer('scale',torch.sqrt(torch.tensor(self.hid_dim).float()))

    def forward(self,query,key,value,inputs_mask=None):
        '''
        多头注意力的计算过程实现
        :param query: query向量
        :param key: key向量
        :param value: value向量
        :param inputs_mask:可选是否mask掉目标的输入信息；这里其实就是为了decoder中那个带有masked操作的attention准备的，因为我们希望的是一个attention的类可以适配所有的attention实例
        :return:
        '''

        # 1.先获取batch_size的大小
        bn = query.size(0)

        # 2.生成QKV矩阵
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # 3.开始划分多头，这里其实要先对Q矩阵进行形状的变换，这里其实会新增一个维度：按照头数来进行划分，每个头的维度要表达出来
        # 这里的Q.view(bn,-1,self.nhead,self.head_dim)其实就是将Q矩阵的维度进行重新排列，即[batch_size,seq_len,hid_dim] -> [batch_size,nhead,seq_len,head_dim]
        Q = Q.view(bn,-1,self.nhead,self.head_dim).permute((0,2,1,3))
        K = K.view(bn,-1,self.nhead,self.head_dim).permute((0,2,1,3))
        V = V.view(bn,-1,self.nhead,self.head_dim).permute((0,2,1,3))

        # 4.计算QK^T矩阵的乘积过程，同时完成根号的d_model的缩放
        energy = torch.matmul(Q,K.permute(0,1,3,2)) / self.scale

        # 5.这里有一个特殊情况需要判断：如果input_mask不为空，则表示需要有mask操作，即mask掉目标的输入信息
        if inputs_mask is not None:

            # 那就mask一下
            # 在Transformer的注意力机制中，mask的维度需要与attention
            # scores（energy）的维度对齐。这里的逻辑是：
            #
            # inputs_mask的构造规则：
            # 当值为0时表示需要被屏蔽的位置（如padding位置或未来时刻的位置）
            # 值为1表示有效位置
            # masked_fill的工作机制：
            # 当条件inputs_mask == 0
            # 成立时（即需要屏蔽的位置）
            # 用极大负数 - 1e10
            # 填充，这样经过softmax后会趋近于0
            energy.masked_fill(inputs_mask == 0,-1.e10) # 这里的-1.e10其实就是一个很小的负数，用于mask掉目标的输入信息

        # 6.再进行softmax归一化操作
        attention_score = F.softmax(energy,dim=-1)

        # 7.继续在跟v做乘积
        out = torch.matmul(attention_score,V)

        # 8.这个输出的维度还需要进行变换，即[batch_size,nhead,seq_len,head_dim] -> [batch_size,seq_len,hid_dim]
        # 这里再次注意：这里指的是将输出的形状重新变回原来的三维状态，那么每个头的输出，会按照头的维度以及头的个数直接完成了拼接，即所有输出的z_i按某维度进行拼接
        out = out.permute((0,2,1,3)).contiguous()
        out = out.view((bn,-1,self.hid_dim)) # 重新转回三维

        # 9.最后再进投影线性层o，即将多个头的输出压缩为一个输出
        out = self.fc_o(out)

        return out,attention_score


# 继续实现feedforward层
class PointWiseFeedForwardLayer(nn.Module):

    def __init__(self, hid_dim, pff_dim, pff_drop_out):
        '''
        实现基于位置编码信息的fc全连接前馈神经网络层
        :param hid_dim: 隐藏层的维度
        :param pff_dim: feedforward层的维度
        :param pff_drop_out: 该层上应用dropout正则化的正则化系数
        '''

        super(PointWiseFeedForwardLayer, self).__init__()

        # 隐藏层的维度
        self.hid_dim = hid_dim

        # feedforward层的维度
        self.pff_dim = pff_dim

        # 正则化系数
        self.pff_drop_out = pff_drop_out

        # 按照论文所描述，准备两个fc全连接
        self.fc1 = nn.Linear(self.hid_dim, self.pff_dim)
        self.fc2 = nn.Linear(self.pff_dim, self.hid_dim)

        # dropout正则化
        self.dropout = nn.Dropout(self.pff_drop_out)

    def forward(self, x):

        return self.fc2(self.dropout(F.relu(self.fc1(x))))

# 继续实现encoderlayer
class EncoderLayer(nn.Module):

    def __init__(self):

        super(EncoderLayer, self).__init__()

        # 1.先实现Encoder中多头注意力层之后的LN层归一化
        self.self_att_layer_norm = nn.LayerNorm(HP.encoder_dim)

        # 2.再实现Encoder中feedforward层之后的ln层归一化
        self.pff_layer_norm = nn.LayerNorm(HP.encoder_dim)

        # 3.再基于MutiHeadAttentionLayer创建多头注意力层
        self.self_att = MutiHeadAttentionLayer(HP.encoder_dim,HP.n_head)

        # 4.再基于PointWiseFeedForwardLayer创建feedforward层
        self.pff = PointWiseFeedForwardLayer(HP.encoder_dim,HP.encoder_feed_forward_dim,HP.feed_forward_drop_prob)

        # 5.再实现dropout正则化
        self.dropout = nn.Dropout(HP.encoder_drop_prob)

    def forward(self,inputs,inputs_mask):

        # 1.输入先进多头注意力层
        _inputs,att_score = self.self_att(inputs, inputs, inputs,inputs_mask)

        # 2.再实现残差连接，之后整体ln层归一化
        inputs = self.self_att_layer_norm(inputs + self.dropout(_inputs))

        # 3.继续进入feedforward层
        _inputs = self.pff(inputs)

        # 4.再实现残差连接，之后整体ln层归一化
        inputs = self.pff_layer_norm(inputs + self.dropout(_inputs))
        return inputs

# 现在总体实现Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # 1.输入的token先embedding词嵌入
        self.token_embedding = nn.Embedding(HP.graphemes_size,HP.encoder_dim)

        # 2.准备位置编码
        self.pe = PositionalEncoding(d_model=HP.encoder_dim,max_len=HP.encoder_max_len)

        # 3.准备6个encoderlayer的完整Encoder结构
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(HP.encoder_layer)])

        # 4.再实现dropout正则化
        self.dropout = nn.Dropout(HP.encoder_drop_prob)

        # 5.注册变量
        self.register_buffer('scale',torch.sqrt(torch.tensor(HP.encoder_dim).float()))

    def forward(self,inputs,inputs_mask):

        # 1.先token做词嵌入
        token_emb = self.token_embedding(inputs)

        # 2.计算位置编码矩阵
        inputs = self.pe(token_emb * self.scale)

        # 3.正则化
        inputs = self.dropout(inputs)

        # 4.就是6个encoderlayer逐层进行信号传递了
        for idx, layer in enumerate(self.layers):
            inputs = layer(inputs,inputs_mask)

        return inputs

# 继续实现Decoderlayer
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()

        # 1.准备Masked多头注意力层
        self.mask_self_att = MutiHeadAttentionLayer(HP.decoder_dim,HP.n_head)

        # 2.准备decoder中masked多头注意力层之后的LN层归一化
        self.mask_self_norm = nn.LayerNorm(HP.decoder_dim)

        # 3.准备decoderlayer中的第二个多头注意力层，注意：该层不是多头自注意力
        self.mha = MutiHeadAttentionLayer(HP.decoder_dim,HP.n_head)

        # 4.准备decoderlayer中第二个多头注意力层之后的LN层归一化
        self.mha_norm = nn.LayerNorm(HP.decoder_dim)

        # 5.准备feedforward层
        self.pff = PointWiseFeedForwardLayer(HP.decoder_dim,HP.decoder_feed_forward_dim,HP.feed_forward_drop_prob)

        # 6.准备feedforward层之后的LN层归一化
        self.pff_norm = nn.LayerNorm(HP.decoder_dim)

        # 7.准备dropout正则化
        self.dropout = nn.Dropout(HP.decoder_drop_prob)

    def forward(self,trg, enc_src, trg_mask, src_mask):
        '''
        decoder layer的forward计算过程
        :param trg: 就是outputs
        :param enc_src: encoder的输出,即encoder输出的上下文向量
        :param trg_mask: 对output的mask
        :param src_mask: 这就是encoder输出的上下文向量的mask，因为前面第一个attention输入qkv的时候本身就是带着trg_mask的，所以encoder的也需要带着mask
        :return:
        '''

        # 1.按照decoderlayer的图示，先进入第一个masked多头自注意力
        _trg,_ = self.mask_self_att(trg, trg, trg, trg_mask)

        # 2.实现残差连接以及之后的ln层归一化
        trg = self.mask_self_norm(trg + self.dropout(_trg))

        # 3.继续进入第二个多头注意力层；注意：这层不是自注意力，刚才输出的trg要作为该attention层的q
        _trg,attention_score = self.mha(trg, enc_src, enc_src, src_mask)

        # 4.实现残差连接以及之后的ln层归一化
        trg = self.mha_norm(trg + self.dropout(_trg))

        # 5.继续进入feedforward层
        _trg = self.pff(trg)

        # 6.实现残差连接以及之后的ln层归一化
        trg = self.pff_norm(trg + self.dropout(_trg))

        # 7.最终，decoderlayer返回结果
        return trg,attention_score

# 最终封装完整的decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # output完成word embedding词嵌入
        self.token_embedding = nn.Embedding(HP.phonemes_size,HP.decoder_dim)

        # 位置编码
        self.pe = PositionalEncoding(d_model=HP.decoder_dim, max_len=HP.MAX_DECODE_STEP)

        # 准备6层decoder
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(HP.decoder_layers)])

        # 准备最后的线性分类层
        self.fc_out = nn.Linear(HP.decoder_dim,HP.phonemes_size)

        # 准备正则化
        self.dropout = nn.Dropout(HP.decoder_drop_prob)

        # 注册变量
        self.register_buffer('scale',torch.sqrt(torch.tensor(HP.decoder_dim).float()))

    def forward(self,trg, enc_src, trg_mask, src_mask):

        # token先embedding
        token_emb = self.token_embedding(trg)

        # 位置编码
        pos_emb = self.pe(token_emb * self.scale)

        # 正则化
        trg = self.dropout(pos_emb)

        # 6层decoderlayer继续传递
        for idx, layer in enumerate(self.layers):
            trg,attention_score = layer(trg, enc_src, trg_mask, src_mask)

        # 最后多分类
        out = self.fc_out(trg)

        return out, attention_score

# 最终，实现完整的transformer模型
class Transformer(nn.Module):

    def __init__(self):
        super(Transformer, self).__init__()

        # 创建encoder
        self.encoder = Encoder()

        # 创建decoder
        self.decoder = Decoder()

    # 封装encoder的input的mask操作实现
    # 为了方便调用，这里直接将该方法封装为静态方法
    @staticmethod
    def create_src_mask(src):
        '''
        对对encoder的输入进行mask操作
        :param src: encoder的input
        :return:
        '''

        return (src != HP.ENCODER_PAD_IDX).unsqueeze(1).unsqueeze(2).to(HP.device) # src是否等于pad的token id，等于就返回True，否则返回False
        # (src != HP.ENCODER_PAD_IDX)就表示masked的位置不应该包含pad填充，因为填充就只是在补齐不足的长度，所以pad位置就不能mask
        # 所以这个逻辑就是只能对非pad token进行填充
        # unsqueeze(1)是在第二维度上增加一个维度，即[batch_size,seq_len] -> [batch_size,1,seq_len]
        # unsqueeze(2)是继续在第三维上增加一个维度
        # 总体目标是改变当前输出的维度

    # 封装decoder的output的mask操作实现
    @staticmethod
    def create_trg_mask(trg):
        '''
        对decoder的输入进行mask操作
        :param trg: decoder的output
        :return:
        '''

        # 1.先获取长度
        trg_len = trg.size(1)

        # 2.output进行mask
        pad_mask = (trg != HP.DECODER_PAD_IDX).unsqueeze(1).unsqueeze(2).to(HP.device) # 这里的masked的位置不应该包含pad填充，因为填充就只是在补齐不足的长度，所以pad位置就不能mask

        # 3. 这里就是将当前step及其之后step的信息进行mask
        # tril方法生成一个tensor张量的下三角形矩阵，配合形状的行列相同，这里其实就是对一个方阵生成其下三角矩阵，值的类型转换为int，然后在转换为bool类型
        sub_mask = torch.tril(torch.ones(size=(trg_len, trg_len),dtype=torch.uint8)).bool().to(HP.device)

        return pad_mask & sub_mask # 这里最终就是对两个bool矩阵做逻辑与运算，即只有两个矩阵中对应位置都为True时，最终结果才为True，否则为False

    def forward(self,src,trg):
        '''

        :param src: input
        :param trg: output
        :return:
        '''

        # 先准备输入的mask
        src_mask = self.create_src_mask(src)

        # 再准备输出的mask
        trg_mask = self.create_trg_mask(trg)

        # 先走encoder
        enc_src = self.encoder(src,src_mask) # 得到encoder的输出

        # 再走decoder
        output,attention_score = self.decoder(trg,enc_src,trg_mask,src_mask) # 得到decoder的输出

        return output,attention_score

    def infer(self,x):
        '''
        实现模型的推理过程
        :param x:
        :return:
        '''

        # 给定输入x,获取其batch大小
        batch_size = x.size(0)

        # 准备输入的mask
        src_mask = self.create_src_mask(x)

        # encoder走一遍
        enc_src = self.encoder(x,src_mask)

        # 初始化输出output的矩阵的形状 - 这里为何要初始化decoder的output embedding？因为这里是推理了，不再是训练了，decoder在推理的时候就没有第一次那个masked attention计算过程了，第二次的多头注意力就没有q了，没q，光一个kv，咋算？所以，来个价格嘛噶的q，占个位置就行了
        trg = torch.zeros(size=(batch_size,1)).fill_(HP.DECODER_SOS_IDX).long().to(HP.device)

        # 准备decoder的解码步数的计数
        decoder_step = 0

        while True:

            # 每一次解码的迭代步开始执行之前，先做判断：是否达到了规定的最大解码迭代步数，如果达到，结束，否则没达到，继续解码迭代
            if decoder_step == HP.MAX_DECODE_STEP:
                print('Warning:reached max decoder step!')
                break

            # 先创建output的mask
            trg_mask = self.create_trg_mask(trg)

            # decoder进行一次forward
            output,attention_score = self.decoder(trg,enc_src,trg_mask,src_mask)

            # output里面就是预测结果
            pred_token = output.argmax(-1)[:,-1]

            # 拼接
            trg = torch.cat((trg,pred_token.unsqueeze(0)),dim=-1)

            # 如果解码到了end标识符
            if pred_token.item() == HP.DECODER_EOS_IDX:
                print('decoder finished!')
                break

            decoder_step += 1

        return trg[:,1:],attention_score




































