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
from Joint_net import *
from train_validate_functions import *


#load data
with open('train_data', 'rb') as fb:
  train_data = pickle.load(fb)
with open('test_data', 'rb') as fb:
  test_data = pickle.load(fb)
with open('dev_data', 'rb') as fb:
  dev_data = pickle.load(fb)

#MAX_LEN is designed to be the max length of a recognized entity
MAX_LEN = 8
#MAX_ENTITIES is the maximum of entities in a sentence
MAX_ENTITIES = 10
#lambda e and lambda r are the loss weights assigned to each task
LAMBDA_E = 1
LAMBDA_R = 1

#construct a relation list and an entity list
entity_types = []
relation_types = []
for paragraph in train_data:
    for sentence in paragraph:
        for entity in sentence.entities:
            if not entity.type in entity_types:
                entity_types.append(entity.type)
        for relation in sentence.relations:
            if not relation.type in relation_types:
                relation_types.append(relation.type)
entity_types.sort()
relation_types.sort()
entity_types.append('Nontype')
relation_types.append('Nontype')

#argv1 contains the filepath to the model
model = torch.load(sys.argv[1])
criterion = nn.CrossEntropyLoss() 
test_dataloader = []
for paragraph in test_data:
    for sentence in paragraph:
        test_dataloader.append(sentence)

loss, scorer = model_validate(model, test_dataloader, criterion, entity_types,\
 relation_types, LAMBDA_E, LAMBDA_R)
#print out the results
entity_precision = scorer.get_entity_precision()
entity_recall = scorer.get_entity_recall()
entity_f1 = scorer.get_entity_f1()
relation_precision = scorer.get_relation_precision()
relation_recall = scorer.get_relation_recall()
relation_f1 = scorer.get_relation_f1()
n = len(entity_f1)
for i in range(n):
    print(f'Entity Type: {entity_types[i]} Precision: {entity_precision[i]}'+
        f' Recall: {entity_recall[i]} F1: {entity_f1[i]}')
n = len(relation_f1)
for i in range(n):
    print(f'Relation Type: {relation_types[i]} Precision: {relation_precision[i]}'+
        f' Recall: {relation_recall[i]} F1: {relation_f1[i]}')
