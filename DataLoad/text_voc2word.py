#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on 03/10/2019 14:42 
@Author: XinZhi Yao 
"""

from DataLoader_Yao import TorchText_DataLoader

train_iter = TorchText_DataLoader(
    train_path='data/sdp_tsv.tsv',
    batch_size=15,
    default_pre_train=False,
    pretrain_vectors='data/glove_deprel',
    split_by='|'
)

def vec2word(vec, index2word):
    word_list = []
    for i in vec:
        word_list.append(index2word[i])
    return '|'.join(word_list)

if __name__ == '__main__':
    for index, data in enumerate(train_iter):
        sent = vec2word(data.text[0][0], index2word)
        print(sent)
        if index == 3:
            break
