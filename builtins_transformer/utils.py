# -*- coding: utf-8 -*-
'''
File:utils.py                                         
Time:2025/7/7 20:32                                    
IDE:PyCharm                                     
Author:Barranzi An                                        
email:awc19930818@outlook.com                           
github:https://github.com/La0bALanG                     
Barranzi's Blog:私聊巴郎子要文档链接                  
requirement:(Please describle your requirement here) -->

插件工具函数的准备

'''

# 准备完整的字素序列，其包含pad（补齐、填充） token、start token、end token以及26个英文小写字母；一般来说，pad token位于序列的第一元素位，即其位于序列中索引为0的位置；
graphemes = ["<pad>", "<s>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz")
# e.g. barranzi -> <s> b a r r a n z i </s>

# 同样，准备完整的音素序列，其包含pad（补齐、填充） token、start token、end token以及26个英文小写字母的音素；音素总计69个
phonemes = ["<pad>", "<s>", "</s>"] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
                'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH',
                'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1',
                'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2',
                                       'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']

# 对字素序列做标签编码，即id2char
graphemes_id2char = dict(enumerate(graphemes)) # 字素序列的标签编码结果
# {0: '<pad>', 1: '<s>', 2: '</s>'...

# 对音素序列同样需要做标签编码，即id2char
phonemes_id2char = dict(enumerate(phonemes))
# {0: '<pad>', 1: '<s>', 2: '</s>',...

# 标签编码后的字素序列及音素序列同样也需要char2id
graphemes_char2id = dict((v,k) for k,v in enumerate(graphemes))
phonemes_char2id = dict((v,k) for k,v in enumerate(phonemes))


# 再封装一系列的工具函数
def word2id(word):
    '''
    对给定的word进行标签编码，转换成id序列
    :param word:
    :return:
    '''
    return [graphemes_char2id[c] for c in list(word)]

def id2word(id_list):
    '''
    对给定的id序列进行解码，转换成word
    :param id_list:字素序列的id序列
    :return:
    '''
    return "".join([graphemes_id2char[i] for i in id_list])

def phoneme2id(phoneme_seq):
    '''
    对给定的phoneme进行标签编码，转换成音素id序列
    :param phoneme_seq:给定的音素序列
    :return:
    '''
    return [phonemes_char2id[c] for c in phoneme_seq.split(' ')]

def id2phoneme(id_list):
    '''
    对给定的音素id序列进行解码，转换成phoneme
    :param id_list:给定的音素id序列
    :return:
    '''
    return " ".join([phonemes_id2char[i] for i in id_list])


if __name__ == '__main__':
    print(f"字素序列：{graphemes}")
    print(f"音素序列：{phonemes}")
    print(f"字素序列的标签编码结果：{graphemes_id2char}")
    print(f"音素序列的标签编码结果：{phonemes_id2char}")
    print(f"字素序列的char to id的结果：{graphemes_char2id}")
    print(f"音素序列的char to id的结果：{phonemes_char2id}")
    print()
    print('======')
    print()

    word = 'jack'
    phone_seq = 'JH AE1 K'

    print(f'将单词:{word}转换成字素id序列：{word2id(word)}')
    print(f'将音素序列:{phone_seq}转换成音素id序列：{phoneme2id(phone_seq)})')

    print(f'将字素id序列：{word2id(word)}转换回字素系列：{id2word(word2id(word))}')
    print(f'将音素id序列：{phoneme2id(phone_seq)}转换回音素序列：{id2phoneme(phoneme2id(phone_seq))}')





















