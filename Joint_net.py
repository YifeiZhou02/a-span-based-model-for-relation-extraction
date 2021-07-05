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

#a joint model to predict entities and relations at the same time
class Joint_net(nn.Module):
    def __init__(self, entity_types, relation_types, MAX_ENTITIES, MAX_LEN):
        super(Joint_net, self).__init__()
        self.ner1 = nn.Linear(512*2+2, 100)
        self.drop = nn.Dropout(p=0.4)
        self.bilstm = nn.LSTM(768, 256, bidirectional=True)
        self.ner2 = nn.Linear(100, len(entity_types))
        self.relation1 = nn.Linear((512*2+2+len(entity_types))*2, 100)
        self.relation2 = nn.Linear(100, len(relation_types))
        self.entity_types = entity_types
        self.relation_types = relation_types
        self.MAX_ENTITIES = MAX_ENTITIES
        self.MAX_LEN = MAX_LEN

    #forward prediction takes a sentence as input
    def forward(self, sentence):
        #obtain entities and relations
        MAX_ENTITIES = self.MAX_ENTITIES
        entity_types = self.entity_types
        relation_types = self.relation_types
        all_candidates, all_labels = sentence.enumerate_all_entities(self.MAX_LEN)
        entity_labels = [entity_types.index(label) for label in all_labels]
        sentence_embeddings, _ = self.bilstm(torch.Tensor(sentence.embeddings).view(-1, 1, 768))
        sentence_embeddings = sentence_embeddings.view(-1,512)
        relation2type = {}
        for relation in sentence.relations:
            relation2type[(relation.arg_1.start, relation.arg_1.end,\
                          relation.arg_2.start, relation.arg_2.end)] = relation.type
        
        entity_embeddings = []
        for entity in all_candidates:
            entity_embeddings.append(torch.Tensor(torch.cat([sentence_embeddings[entity[0]],\
                                    sentence_embeddings[entity[1]],\
                                    torch.Tensor([entity[0], entity[1]])], dim = -1)).reshape(1,-1))
        entity_embeddings = torch.cat(entity_embeddings, dim = 0)
        #pass entity_embeddings into a feed_forward neural network to predict if they are
        #entities and entity types
        
        ner1_results = F.relu(self.ner1(entity_embeddings))
        ner1_results = self.drop(ner1_results)
        ner1_results = self.ner2(ner1_results)
        after_ner1_embeddings = torch.cat([entity_embeddings, ner1_results], dim = 1)
        ner1_types = torch.argmax(ner1_results, dim = -1)
        
        #select the beam entity spans
        entity_none = len(self.entity_types) - 1
        chosen_indexes = []
        for i, type_id in enumerate(ner1_types):
            if type_id != entity_none:
                chosen_indexes.append(i)
        #put 0 in the list if there's nothing left
        if len(chosen_indexes) == 0:
            chosen_indexes.append(0)
        beam_candidates = []
        beam_embeddings = []
        for index in chosen_indexes:
            if ner1_types[index] < 0 and len(beam_candidates) > 0:
                break
            beam_candidates.append(all_candidates[index])
            beam_embeddings.append(after_ner1_embeddings[index])
        n = len(beam_embeddings)
        relation_labels = torch.zeros(n,n)
        all_relations = []
        for i in range(n):
            for j in range(n):
                temp_relation = (beam_candidates[i][0], beam_candidates[i][1],
                                beam_candidates[j][0], beam_candidates[j][1])
                all_relations.append(temp_relation)
                if temp_relation in relation2type:
                    relation_labels[i,j] = relation_types.index(relation2type[temp_relation])
                else:
                    relation_labels[i,j] = len(relation_types) - 1
        relation_labels = relation_labels.flatten()
        relation_embeddings = []
        for i in range(n):
            for j in range(n):
                relation_embeddings.append(torch.cat([beam_embeddings[i],\
                                                    beam_embeddings[j]], dim = -1).reshape(1,-1))
        relation_embeddings = torch.cat(relation_embeddings, dim = 0)
        relation_results = F.relu(self.relation1(relation_embeddings))
        relation_results = self.drop(relation_results)
        relation_results = self.relation2(relation_results)
        #return: ner1_results is the prediction score for each span for each label
        #entity_labels is the true label for each span
        #relation_results is the score for each relation
        #relation labels is the true label for each relation
        #all_candidates is the span for all entities
        #all_relations is the span for all relations
        return ner1_results, entity_labels, relation_results, relation_labels,  all_candidates, all_relations
    #predict is used to predict entity and relations
    def predict(self, sentence):
        ner1_results, entity_labels, relation_results, relation_labels, \
        all_entities, all_relations = self.forward(sentence)
        predicted_entities = {}
        predicted_relations = {}
        ner_predictions = torch.argmax(ner1_results, dim = 1).numpy()
        relation_predictions = torch.argmax(relation_results, dim = 1).numpy()
        entity_none = len(self.entity_types) - 1
        relation_none = len(self.relation_types) - 1
        for i, entity_type in enumerate(ner_predictions):
            if entity_type != entity_none:
                predicted_entities[all_entities[i]] = entity_type
        for i, relation_type in enumerate(relation_predictions):
            if relation_type != relation_none:
                predicted_relations[all_relations[i]] = relation_type
        return predicted_entities, predicted_relations
