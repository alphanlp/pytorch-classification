# -*- coding：utf-8 -*-
# Copyright 2018 alphaTech.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
pytorch Convolutional Neural Networks for Sentence Classification
author: alpha hu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import os
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score

torch.manual_seed(1)


class TextCNN(nn.Module):
    """
    user pytorch conv1d to implement text cnn classification
    """

    def __init__(self, vocab_size, emb_size, num_filters, kernel_sizes, dropout, num_classes):
        super(TextCNN, self).__init__()
        # 1. embedding layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim=emb_size)
        # 2. cnn layer
        self.convs = nn.ModuleList([nn.Conv1d(emb_size, num_filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, input):
        # self.embeddings(input)'s output size is [batch_size, seq_length, emb_size]
        # conv1d's input size is [batch_size, in_channels, seq_length]
        embedded = self.embeddings(input).permute(0, 2, 1)
        # conv1d output size: [batch_size, out_channeld, out_len]
        x = [F.relu(conv(embedded).permute(0, 2, 1).max(dim=1)[0]) for conv in self.convs]
        x = self.dropout(torch.cat(x, dim=1))
        x = self.fc(x)
        return x


def sentence_to_id(text, word_to_id, max_length):
    """
    convert text to id. unk token will be padded.
    """
    text = text.split()
    text = [word_to_id[x] for x in text if x in word_to_id]
    if len(text) < max_length:
        text += [0] * (max_length - len(text))
    return text[:max_length]


class Corpus():
    def __init__(self, pos_file, neg_file, vocab_file, dev_split=0.01, max_length=40, vocab_size=10000):
        pos_examples = [s.strip() for s in open(pos_file, mode='r', encoding='utf-8') if s.strip()]
        neg_examples = [s.strip() for s in open(neg_file, mode='r', encoding='utf-8') if s.strip()]
        pos_len = len(pos_examples)
        neg_len = len(neg_examples)
        # sample, not necessary
        if pos_len < neg_len:
            neg_examples = neg_examples[0:pos_len]
        else:
            pos_examples = pos_examples[0:neg_len]
        x_data = pos_examples + neg_examples
        y_data = [0.] * len(pos_examples) + [1.] * len(neg_examples)  # label

        counter = Counter()
        if not os.path.exists(vocab_file):
            for s in x_data:
                counter.update(s.split())
            count_pairs = counter.most_common(vocab_size - 1)
            words, _ = list(zip(*count_pairs))
            self.words = ['<pad>'] + list(words)
            self.word_to_id = dict(zip(self.words, range(len(self.words))))
            with open(vocab_file, mode='w', encoding='utf-8') as fout:
                fout.write('\n'.join(self.words) + '\n')
        else:
            with open(vocab_file, mode='r', encoding='utf-8') as fin:
                self.words = [word.strip() for word in fin if word.strip()]
            self.word_to_id = dict(zip(self.words, range(len(self.words))))

        for i in range(len(x_data)):  # tokenizing and padding
            x_data[i] = sentence_to_id(x_data[i], self.word_to_id, max_length)

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        # shuffle
        ids = np.random.permutation(np.arange(len(x_data)))
        x_data = x_data[ids]
        y_data = y_data[ids]

        num_train = int((1 - dev_split) * len(x_data))
        self.x_train = x_data[:num_train]
        self.y_train = y_data[:num_train]
        self.x_test = x_data[num_train:]
        self.y_test = y_data[num_train:]

    def __str__(self):
        return 'Training: {}, Testing: {}, Vocabulary: {}'.format(len(self.x_train), len(self.x_test), len(self.words))


def eval(data, model, loss):
    model.eval()

    data_loader = DataLoader(data, batch_size=16)
    data_len = len(data)
    total_loss = 0.0
    y_true, y_pred = [], []
    y_output = []

    with torch.no_grad():
        for data, label in data_loader:
            output = model(data)
            losses = loss(output, label)

            total_loss += losses.item()
            pred = torch.max(output.data, dim=1)[1].numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(label.tolist())

            output = F.softmax(output, dim=1)
            y_output.extend(output.data.tolist())

    acc = (np.array(y_true) == np.array(y_pred)).sum()
    auc = roc_auc_score(np.array(y_true), np.array(y_output)[:, 1])
    return acc / data_len, total_loss / data_len, auc


def train():
    corpus = Corpus(pos_file='data/rt-polarity.pos.txt', neg_file='data/rt-polarity.neg.txt',
                    vocab_file='data/vocab')
    vocab_size = len(corpus.words)

    train_data = TensorDataset(torch.tensor(corpus.x_train, dtype=torch.long), torch.tensor(corpus.y_train, dtype=torch.long))
    test_data = TensorDataset(torch.tensor(corpus.x_test, dtype=torch.long), torch.tensor(corpus.y_test, dtype=torch.long))

    model = TextCNN(vocab_size, emb_size=128, num_filters=100, kernel_sizes=[3, 4, 5], dropout=0.1, num_classes=2)
    # print model
    # for idx, m in enumerate(model.modules()):
    #     print(idx, '->', m)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    step = 0
    model.train()
    while step < 10000:
        train_loader = DataLoader(train_data, batch_size=64)
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            inputs, targets = x_batch, y_batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            step += 1

            acc = (torch.max(outputs, dim=1)[1] == targets).sum().item()
            print("training step={}, acc={}, loss={}".format(step, acc / len(targets), loss.item()))

            if step % 100 == 0:
                test_acc, test_loss, test_auc = eval(test_data, model, criterion)
                print("valid step={}, acc={}, auc={}, loss={}".format(step, test_acc, test_auc, test_loss))
                model.train()

            if step % 1000 == 0:
                if not os.path.exists('saved_models'):
                    os.makedirs('saved_models')
                torch.save(model.state_dict(), 'saved_models/text_cnn_{}.pt'.format(int(step / 1000)))

            if step > 10000:
                break


def predict(model_name, string):
    with open('data/vocab', mode='r', encoding='utf-8') as fin:
        words = [word.strip() for word in fin if word.strip()]
    word_to_id = dict(zip(words, range(len(words))))
    vocab_size = len(words)
    model = TextCNN(vocab_size, emb_size=128, num_filters=100, kernel_sizes=[3, 4, 5], dropout=0.1, num_classes=2)
    model.load_state_dict(torch.load('saved_models/' + model_name))
    model.eval()

    predict_data = []
    words = list(string.strip().split('\t'))
    text = ' '.join([word for word in words if word.strip()])
    predict_data.append(text)
    for i in range(len(predict_data)):  # tokenizing and padding
        predict_data[i] = sentence_to_id(predict_data[i], word_to_id, 40)
    # predict_data = np.array(predict_data)
    data = torch.tensor(predict_data, dtype=torch.long)
    output = model(data)
    output = F.softmax(output, dim=1)
    pred = torch.max(output.data, dim=1)[1].numpy().tolist()
    print(pred)
    return pred


if __name__ == '__main__':
    #train()
    predict('text_cnn_4.0.pt', "it's so laddish and juvenile , only teenage boys could possibly find it funny .")
