import re, sys, os
import fnmatch
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import matplotlib.pyplot as plt
import pickle
from helper_classes import *

#train and validate the model
def model_train(net, train_dataloader, optimizer, criterion, LAMBDA_E, LAMBDA_R, toprint = False ):
    net.train()
    losses = []
    i = 0
    for batch_idx, sentence in enumerate(train_dataloader):
        if toprint:
          print(batch_idx)
#         optimizer.zero_grad()
        ner1_results, entity_labels, relation_results, relation_labels,\
        _, _ = net(sentence)
        entity_labels = torch.Tensor(entity_labels).long()
        relation_labels = torch.Tensor(relation_labels).long()
        loss = LAMBDA_E*criterion(ner1_results, entity_labels) + LAMBDA_R*criterion(relation_results, relation_labels)
        loss.backward()
        if i == 4:
            optimizer.step()
            optimizer.zero_grad()
            i = 0
#         optimizer.step()
        i += 1
        losses.append(loss.item())
    return np.mean(np.array(losses))
#validate the model, return the loss
def model_validate(net, val_dataloader, criterion, entity_types, relation_types, LAMBDA_E, LAMBDA_R):
    net.eval()
    losses = []
    for batch_num, sentence in enumerate(val_dataloader):
        ner1_results, entity_labels, relation_results, relation_labels,\
        _, _ = net(sentence)
        entity_labels = torch.Tensor(entity_labels).long()
        relation_labels = torch.Tensor(relation_labels).long()
        loss = LAMBDA_E*criterion(ner1_results, entity_labels) + LAMBDA_R*criterion(relation_results, relation_labels)
        losses.append(loss.item())
    losses = np.array(losses)
    scorer = F1_scorer(entity_types, relation_types)
    for sentence in val_dataloader:
        predicted_entities, predicted_relations = net.predict(sentence)
        scorer.update_sentence(sentence, predicted_entities, predicted_relations)
    # print(f'the validation entity f1 is {scorer.get_entity_f1()}')
    # print(f'the validation relation f1 is {scorer.get_relation_f1()}')
    return np.mean(losses), scorer
