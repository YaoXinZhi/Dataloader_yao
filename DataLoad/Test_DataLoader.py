#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on 25/09/2019 15:34
@Author: XinZhi Yao
"""

from DataLoader_Yao import TorchText_DataLoader
# import TorchText_DataLoader

train_iter = TorchText_DataLoader(
    train_path='data/test_tsv_500.tsv',
    pre_train=False,
    pretrain_vectors='data/glove_deprel',
    split_by=None
)

