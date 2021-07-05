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

train_dataloader = []
for paragraph in test_data:
    for sentence in paragraph:
        train_dataloader.append(sentence)

model = Joint_net(entity_types, relation_types, MAX_ENTITIES, MAX_LEN)
train_losses = []
criterion = nn.CrossEntropyLoss() 
LAMBDA_E = 1
LAMBDA_R = 0
optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay= 1e-4)
for i in range(5):
    print(f'for the {i}th epoch')
    train_losses.append(model_train(model, train_dataloader, optimizer, criterion, LAMBDA_E, LAMBDA_R,toprint = False))
    print(f'current training loss: {train_losses[-1]}')
optimizer = optim.Adam(model.parameters(), lr = 1e-4, weight_decay= 1e-4)
for i in range(5):
    print(f'for the {i+5}th epoch')
    train_losses.append(model_train(model, train_dataloader, optimizer, criterion, LAMBDA_E, LAMBDA_R,toprint = False))
    print(f'current training loss: {train_losses[-1]}')
LAMBDA_E = 1
LAMBDA_R = 1
optimizer = optim.Adam(model.parameters(), lr = 1e-4, weight_decay= 1e-4)
for i in range(20):
    print(f'for the {i+10}th epoch')
    train_losses.append(model_train(model, train_dataloader, optimizer, criterion, LAMBDA_E, LAMBDA_R,toprint = False))
    print(f'current training loss: {train_losses[-1]}')

torch.save(model, sys.argv[1])
