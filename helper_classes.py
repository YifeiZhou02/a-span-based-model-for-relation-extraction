import numpy as np
import re, sys, os
#a class to contain information of a entity mention
class Entity_mention:
    def __init__(self, ne_id, ne_type, ne_start, ne_end, text, sentenceNo):
        self.id = ne_id
        self.type = ne_type
        self.start = ne_start
        self.end = ne_end
        self.text = text
        self.sentenceNo = sentenceNo
#a class to contain information of a relation mention
class Relation_mention:
    def __init__(self, rel_id, rel_type, arg_1, arg_2, sentenceNo):
        self.id = rel_id
        self.type = rel_type
        self.arg_1 = arg_1
        self.arg_2 = arg_2
        self.sentenceNo = sentenceNo
#a basic class of sentence
class Sentence:
    def __init__(self, sentenceNo, text):
        self.sentenceNo = sentenceNo
        self.text = text
        self.entities = []
        self.relations = []
        self.embeddings = []
    def update_embeddings(self, tokenizer, bert):
    #compute the weighted embeddings for each token
        all_words = re.split(' +',self.text)
        all_tokens = []
        for word in all_words:
            tokens = tokenizer.encode_plus(word,\
                              add_special_tokens = False, return_token_type_ids = False,\
                              return_attention_mask = False, return_tensors = 'pt')
            all_tokens.append(len(tokens['input_ids'].flatten()))
        tokens = tokenizer.encode_plus(self.text,\
                              add_special_tokens = True, return_token_type_ids = False,\
                              return_attention_mask = False, return_tensors = 'pt')
        bertids = tokens['input_ids']
        #discard the first and last token embeddings
        all_embeddings = bert(bertids)[0].detach().numpy().reshape(-1, 768)[1:-1]
        check_point = 0
        n = len(all_tokens)
        for i in range(n):
            new_check = check_point + all_tokens[i]
            if len(all_embeddings[check_point: new_check]) == 0:
                raise RuntimeError('got an empty embeddings')
            self.embeddings.append(np.max(all_embeddings[check_point: new_check],axis = 0))
            check_point = new_check
        assert check_point == len(all_embeddings)
        self.embeddings = np.array(self.embeddings)
        
    #this function enumerates all substrings with token length shorter 
    #or equal to MAX_LEN
    def enumerate_all_entities(self, MAX_LEN):
        n = len(self.embeddings)
        gold2entity = {}
        for entity in self.entities:
            gold2entity[(entity.start, entity.end)] = entity
        all_entities = []
        all_labels = []
        for consider_len in range(MAX_LEN):
            for start in range(n - consider_len):
                all_entities.append((start, start+ consider_len))
                if (start, start+consider_len) in gold2entity:
                    all_labels.append(gold2entity[(start, start+consider_len)].type)
                else:
                    all_labels.append('Nontype')
        return all_entities, all_labels

#the class to model F1 score
class F1_scorer:
    def __init__(self, entity_types, relation_types):
        #use a 2-d array to record truth number, predicted truth number, predicted number
        #for each type
        self.entity_types = entity_types
        self.relation_types = relation_types
        self.entity_f1 = np.zeros((len(entity_types)-1, 3))
        self.relation_f1 = np.zeros((len(relation_types)-1, 3))
    
    #update the counter for a sentence and its predictions
    def update_sentence(self, sentence, predicted_entities, predicted_relations):
        #update entity counts
        for entity in sentence.entities:
            entity_type = self.entity_types.index(entity.type)
            self.entity_f1[entity_type][0] += 1
            if (entity.start, entity.end) in predicted_entities and\
            predicted_entities[(entity.start, entity.end)] == entity_type:
                self.entity_f1[entity_type][1] += 1
        for entity in predicted_entities:
            entity_type = predicted_entities[entity]
            self.entity_f1[entity_type][2] += 1
        #update relations
        for relation in sentence.relations:
            relation_type = self.relation_types.index(relation.type)
            self.relation_f1[relation_type][0] += 1
            if (relation.arg_1.start, relation.arg_1.end, relation.arg_2.start,\
                relation.arg_2.end) in predicted_relations and\
            predicted_relations[(relation.arg_1.start, relation.arg_1.end, relation.arg_2.start,\
                relation.arg_2.end)] == relation_type:
                self.relation_f1[relation_type][1] += 1
        for relation in predicted_relations:
            relation_type = predicted_relations[relation]
            self.relation_f1[relation_type][2] += 1
    
    #get entity precision
    def get_entity_precision(self):
        return self.entity_f1[:,1]/self.entity_f1[:,2]
    
    #get entity recall
    def get_entity_recall(self):
        return self.entity_f1[:,1]/self.entity_f1[:,0]
    
    #get type f1 for entities
    def get_entity_f1(self):
        return 2/((1/self.get_entity_precision())+(1/self.get_entity_recall()))
    #get relation precision
    def get_relation_precision(self):
        return self.relation_f1[:,1]/self.relation_f1[:,2]
    
    #get relation recall
    def get_relation_recall(self):
        return self.relation_f1[:,1]/self.relation_f1[:,0]
    #get type f1 for relations
    def get_relation_f1(self):
        return 2/((1/self.get_relation_precision())+(1/self.get_relation_recall()))