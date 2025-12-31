# -*- coding: utf-8 -*-
'''
File:dataloaders.py                                         
Time:2025/7/7 20:33                                    
IDE:PyCharm                                     
Author:Barranzi An                                        
email:awc19930818@outlook.com                           
github:https://github.com/La0bALanG                     
Barranzi's Blog:私聊巴郎子要文档链接                  
requirement:(Please describle your requirement here) -->
'''

from torch.utils.data import Dataset, DataLoader
import json

from utils import *
from config import *
import torch

# 创建自定义dataloader对象
class G2PDataset(Dataset):
    '''
    自定义数据集类
    '''

    def __init__(self,dataset_path):
        '''
        初始化函数
        :param dataset_path: 数据集的路径
        '''

        # 使用json对象的load方法从json文件中加载数据，转换为python中的字典
        data_dict = json.load(open(dataset_path,'r'))

        # 调用字典的items方法同时获取key和value，然后在使用tuple元组类型封装key-value对，整体使用list来返回
        self.data_pairs = list(data_dict.items()) # [('fission', 'F IH1 SH AH0 N'), ('hagenlocker', 'HH EY1 G AH0 N L AA2 K ER0'),...
        # print(f"加载完毕的数据：{self.data_pairs}")


    def __len__(self):

        return len(self.data_pairs)

    def __getitem__(self, idx):

        # 获取指定某索引位的样本对的单词及其音素序列
        word,phone_seq = self.data_pairs[idx][0],self.data_pairs[idx][1]

        # 然后使用封装好的工具函数将单词序列和音素序列映射为token id序列
        return word2id(word),phoneme2id(phone_seq)


def collate_fn(batch):
    '''
    自定义的collate_fn函数，用于对一个batch下的数据进行预处理
    :param batch: 一个batch下的数据
    :return:
    '''
    # print('+++')

    # 1.先获取一个batch下的样本总数，即batch_size的大小
    N = len(batch)

    # 2.*号拆解batch序列，然后使用zip函数将其打包为一个元组，然后再使用list函数将其转换为列表后得到两个结果，一个是：一个batch下所有word的token id序列
    # 一个是：一个batch下所有phoneme的token id序列
    word_indexes,phonemes_indexes = [list(it) for it in zip(*batch)]
    # print(f"word_indexes: {word_indexes}")
    # print(f"phonemes_indexes: {phonemes_indexes}")
    # print()
    # print()
    # word_indexes: [[8, 11, 21, 21, 11, 17, 16]]
    # phonemes_indexes: [[34, 38, 58, 9, 47]]


    # 3.将所有的word token id序列和phonmemes token id 序列添加起始的token和结束的token
    [it.insert(0,graphemes_char2id['<s>']) for it in word_indexes]
    [it.append(graphemes_char2id['</s>']) for it in word_indexes]

    [it.insert(0,phonemes_char2id['<s>']) for it in phonemes_indexes]
    [it.append(phonemes_char2id["</s>"]) for it in phonemes_indexes]

    # print(f"word_indexes完成开始和结束填充后的token id 序列: {id2word(word_indexes[0])}")
    # print(f"phonemes_indexes完成开始和结束填充后的token id 序列: {id2phoneme(phonemes_indexes[0])}")
    # print()
    # print()
    # word_indexes完成开始和结束填充后的token id 序列: <s>fission</s>
    # phonemes_indexes完成开始和结束填充后的token id 序列: <s> F IH1 SH AH0 N </s>



    # 4.对word_indexes序列进行降序排序 - 这里的排序其实并没有直接对word_indexes进行排序，而是对word_indexes的长度进行排序
    #                                                        for循环遍历一个batch下的所有word单词的token id 序列
    #                                                it拿到的，就是其中每一个word token id 序列，len获取其长度
    #                                               获取后的结果，重构为一个list
    #                                  tensor方法将其转换为long类型整数张量
    # 调用torch.sort获取其中长度最大的word的token id 序列的长度 - 降序排序后，第一个值不就是长度最大的word token id序列的长度么？
    word_lengths,sort_idx = torch.sort(torch.tensor([len(it) for it in word_indexes]).long(),descending=True)
    # print(f'word_lengths: {word_lengths}')
    # print(f'sort_idx: {sort_idx}')
    # print(f"word_indexes完成开始和结束填充后的token id 序列: {id2word(word_indexes[0])}")
    # print()
    # print()

    # 这不就是在获取最长的word的长度？
    # 5.获取word的最大长度
    max_word_len = word_lengths[0]

    # 6.开始准备对word进行填充
    # 6.1.首先创建一个全0的tensor，其shape为：(batch_size,max_word_len)
    word_padded = torch.zeros((N,max_word_len)).long()

    # 7. 再获取音素序列的最大长度
    max_phoneme_len = max([len(it) for it in phonemes_indexes])

    # 8. 开始准备对phoneme进行填充
    # 8.1.首先创建一个全0的tensor，其shape为：(batch_size,max_phoneme_len)
    phoneme_padded = torch.zeros((N,max_phoneme_len)).long()
    phoneme_length = torch.zeros((N,)).long()

    # 9.统一将数据转换为tensor张量对象 - 即这里就开始对长度不足的word和phoneme进行填充
    for idx,idx_s in enumerate(sort_idx.tolist()):

        # 这里是对word进行填充
        word_padded[idx][:word_lengths[idx]] = torch.tensor(word_indexes[idx_s]).long()

        # 这里就是对phoneme进行填充
        phoneme_padded[idx][:len(phonemes_indexes[idx_s])] = torch.tensor(phonemes_indexes[idx_s]).long()
        phoneme_length[idx] = len(phonemes_indexes[idx_s])

    # 10.返回最终一个batch下的所有处理完毕的数据
    return word_padded,word_lengths,phoneme_padded,phoneme_length

if __name__ == '__main__':
    # 实例化自定义的dataloader对象
    datasets = G2PDataset('./data/data_val.json')
    dataloaders = DataLoader(datasets,batch_size=4,collate_fn=collate_fn) # 这里注意：实际在使用Dataloader按一个batch封装数据时，需要对一个batch下的数据进行预处理
    # 要将其处理为合适模型输入的格式，所以这里的collate_fn就是一个单独需要实现的回调函数，在该函数内实现数据的处理过程即可

    for batch in dataloaders:
        word_padded,word_lengths,phoneme_padded,phoneme_length = batch
        print(word_padded)
        print()
        print()
        print()
        print(word_lengths)
        print()
        print()
        print()
        print(phoneme_padded)
        print()
        print()
        print()
        print(phoneme_length)
        break


'''
测试的日志输出结果：
/root/miniconda3/bin/python3 /gemini/code/transformers_project/dataloaders.py 
+++
word_indexes: [[8, 11, 21, 21, 11, 17, 16], [10, 3, 9, 7, 16, 14, 17, 5, 13, 7, 20], [15, 17, 16, 22, 7, 21, 11, 16, 17], [15, 23, 14, 10, 17, 14, 14, 3, 16, 6]]
phonemes_indexes: [[34, 38, 58, 9, 47], [36, 32, 35, 9, 47, 45, 5, 44, 28], [46, 49, 47, 59, 25, 57, 41, 47, 49], [46, 11, 45, 36, 4, 45, 9, 47, 23]]


word_indexes完成开始和结束填充后的token id 序列: <s>fission</s>
phonemes_indexes完成开始和结束填充后的token id 序列: <s> F IH1 SH AH0 N </s>


word_lengths: tensor([13, 12, 11,  9])
sort_idx: tensor([1, 3, 2, 0])
word_indexes完成开始和结束填充后的token id 序列: <s>fission</s>


tensor([[ 1, 10,  3,  9,  7, 16, 14, 17,  5, 13,  7, 20,  2],
        [ 1, 15, 23, 14, 10, 17, 14, 14,  3, 16,  6,  2,  0],
        [ 1, 15, 17, 16, 22,  7, 21, 11, 16, 17,  2,  0,  0],
        [ 1,  8, 11, 21, 21, 11, 17, 16,  2,  0,  0,  0,  0]])



tensor([13, 12, 11,  9])



tensor([[ 1, 36, 32, 35,  9, 47, 45,  5, 44, 28,  2],
        [ 1, 46, 11, 45, 36,  4, 45,  9, 47, 23,  2],
        [ 1, 46, 49, 47, 59, 25, 57, 41, 47, 49,  2],
        [ 1, 34, 38, 58,  9, 47,  2,  0,  0,  0,  0]])



tensor([11, 11, 11,  7])

进程已结束，退出代码为 0

'''
















