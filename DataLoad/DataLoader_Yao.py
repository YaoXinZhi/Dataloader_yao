#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on 25/09/2019 9:59 
@Author: XinZhi Yao 
"""

import spacy
from torchtext.vocab import Vectors
from torchtext.data import Field, BucketIterator, TabularDataset

'''
25/9 
inputs: csv file
output: Iterator of train data
'''


def TorchText_DataLoader(train_path, pretrain_vectors='glove.42B.300d', split_by=None,
                         language='en', include_lengths=True, batch_size=25,
                         skip_header=True, default_pre_train=True, init_token=None,
                         eos_token=None, fix_length=None):
    """
    :param root_path: Training file root directory
    :param train_csv: Training file
    :param data_fields: Csv file listing name [('id', None'), ('comment_text', TEXT), ('threat', LABEL)]
    :param pretrain_vectors:  Pre-training word embedding vector
    :param language: Language, the default is English
    :param split_by: Text segmentation symbol
    :param include_lengths: Returns whether the length is included
    :param batch_size: Mini batch size
    :param skip_header: Whether the csv file contains a header
    :return: train_data_iter

    9/25 只能读取 csv文件 每次换文件需要改 data_fields

    29/9 添加 bool pre_train
    :param pre_train: bool True--> pre-train embedding | False--> Specified word embedded file
    :param split_by: split text by specified symbol

    3/10 添加 init_token, eos_token 只有翻译任务时候加入eos 和 pos
    """

    # Tokenizer
    spacy_tok = spacy.load(language)

    def tokenize_spacy(text):
        return [tok.text for tok in spacy_tok.tokenizer(text)]

    def tokenize_by(text):
        return [tok for tok in text.split('|')]

    # Field
    if split_by is None:
        Text = Field(sequential=True, tokenize=tokenize_spacy, include_lengths=include_lengths,
                     batch_first=True, lower=True, init_token=init_token, eos_token=eos_token, fix_length=fix_length)
    else:
        Text = Field(sequential=True, tokenize=tokenize_by, include_lengths=include_lengths,
                     batch_first=True, lower=True, init_token=init_token, eos_token=eos_token, fix_length=fix_length)

    # u have to modify it if file head was changed
    data_fields = [('id', None), ('text', Text)]

    # loading data
    print('loading data: {0}'.format(train_path))
    train_data = TabularDataset(
        path=train_path,
        format='tsv',
        skip_header=skip_header,
        fields=data_fields
    )

    # loading pre_train embedding
    if default_pre_train:
        print ('loading pre-train vectors: {0}'.format(pretrain_vectors))
        Text.build_vocab(train_data, vectors=pretrain_vectors)
    else:
        print ('load your vectors: {0}'.format(pretrain_vectors))
        vectors_new = Vectors(name=pretrain_vectors)
        Text.build_vocab(train_data, vectors=vectors_new)

    # Iterator
    train_iter = BucketIterator(
        train_data,
        batch_size=batch_size,
        sort_key=lambda x: len(x.text),
        device=None,
        sort_within_batch=False,
        repeat=False
    )

    return train_iter
