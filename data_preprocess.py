import re, sys, os
import fnmatch
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import pickle
from helper_classes import *
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
from transformers import AutoModel
bert = AutoModel.from_pretrained("bert-base-cased")

#convert the original sgm text to text in the desired format
def process_raw_text(paths):
    docs = []
    for file_path in paths:
        doc = open(file_path+'.sgm', 'r').read()
        doc = re.sub(r"<[^>]+>", "", doc)
        doc = re.sub(r"(\S+)\n(\S[^:])", r"\1 \2", doc)
        docs.append(doc)
    return docs

#process the apfxml text to get named entites in the form of a list of dictionaries
def get_named_entites_and_relations(paths):
    all_nes = []
    all_rels = []
    for file_path in paths:
        apf_tree = ET.parse(file_path+'.apf.xml')
        apf_root = apf_tree.getroot()
        named_entities = {}
        #check nes is used to remove duplicate entities
        check_nes = {}
        ne_map = {}
        #get named entites
        for entity in apf_root.iter('entity'):
            ne_type = entity.attrib["TYPE"]
            for mention in entity.iter('entity_mention'):
                ne_id = mention.attrib["ID"]
                for child in mention:
                    if child.tag == 'extent':
                        for charseq in child:
                            start = int(charseq.attrib["START"])
                            end = int(charseq.attrib["END"])
                            text = re.sub(r"\n", r" ", charseq.text)
                            ne_tuple = (ne_type, start, end, text)
                            if ne_tuple in check_nes:
#                                 sys.stderr.write("duplicated entity %s\n" % (ne_id))
                                ne_map[ne_id] = check_nes[ne_tuple]
                                continue
                            check_nes[ne_tuple] = ne_id
                            named_entities[ne_id] = [ne_type, start, end, text]
        #get relations
        rels = {}
        check_rels = []
        for relation in apf_root.iter('relation'):
            rel_type = relation.attrib["TYPE"]
            for mention in relation.iter('relation_mention'):
                rel_id = mention.attrib["ID"]
                rel = [ rel_type, "", ""]
                for arg in mention.iter('relation_mention_argument'):
                    arg_id = arg.attrib["REFID"]
                    #deal with some anomaly situations
                    if arg.attrib["ROLE"] != "Arg-1" and arg.attrib["ROLE"] != "Arg-2":
                        continue
                    if arg_id in ne_map:
                        arg_id = ne_map[arg_id]
                    rel[int(arg.attrib["ROLE"][-1])] = arg_id
                    if not arg_id in named_entities:
                        continue
                if rel[1:] in check_rels:
                    continue
                check_rels.append(rel[1:])
                rels[rel_id] = rel
        all_nes.append(named_entities)
        all_rels.append(rels)
    return all_nes, all_rels

#construct the data structure: sentence for later use
def construct_sentence_data(docs, all_ners, all_relations):
    num_doc = len(docs)
    all_data = []
    for j in range(num_doc):
        doc = docs[j]
        ner_dict = all_ners[j]
        relation_dict = all_relations[j]
        #ners and relations store the ners and relations in the specific class
        ners = []
        relations = []
        #char label contains the sentence No. and token No. for each character
        #starting from 0
        char_labels = []
        n = len(doc)
        sentence = 0
        token = -1
        for i in range(n):
            char_labels.append((sentence, token))
            if i >= n-1:
                continue
            if doc[i] in ['.', '?', '!'] and doc[i+1] in [' ', '\n'] and \
            (not i == 0 and (doc[i-1] < 'A' or doc[i-1] > 'Z')):
                sentence += 1
                token = -1
            if doc[i] in [' ', '\n'] and (not doc[i+1] in [' ', '\n']):
                token += 1        
        sentences = re.split('(?<![A-Z])[.!?][ |\\n]+',doc)
        id2ne = {}
        #prepare nes
        for ne_id in ner_dict.keys():
            ne_type, start, end, text = ner_dict[ne_id]
            ners.append(Entity_mention(ne_id, ne_type, char_labels[start][1], \
                                       char_labels[end][1], text, char_labels[start][0]))
            id2ne[ne_id] = ners[-1]
        #prepare relations
        for relation_id in relation_dict.keys():
            rel_type, arg_1, arg_2 = relation_dict[relation_id]
            relations.append(Relation_mention(relation_id, rel_type, id2ne[arg_1],\
                                              id2ne[arg_2], id2ne[arg_2].sentenceNo))
        #now get sentences down
        paragraph = []
        for i,sentence in enumerate(sentences):
            if sentence == '':
                paragraph.append(None)
                continue
            new_sentence = re.sub('\n',' ', sentence)
            new_sentence = re.sub('^ +', '', new_sentence)
            new_sentence = re.sub(' +$', '', new_sentence)
            paragraph.append(Sentence(i, new_sentence))
        for entity in ners:
            paragraph[entity.sentenceNo].entities.append(entity)
        for relation in relations:
            paragraph[relation.sentenceNo].relations.append(relation)
        while (None in paragraph):
            paragraph.remove(None)
        all_data.append(paragraph)
    return all_data

#iterate over the directory to get all file paths
#train paths and test paths contains all paths to the examples
#train_path + '.sgm' can be used to access the file
#simililar for test_path
train_paths = []
prefix = 'corpus/train/'
prefix2 = '/timex2norm/'
for first_file in os.listdir('corpus/train'):
    if first_file == '.DS_Store':
        continue
    for second_file in os.listdir(f'corpus/train/{first_file}/timex2norm'):
        if second_file == '.DS_Store':
            continue
        if fnmatch.fnmatch(second_file, '*.sgm'):
            train_paths.append(prefix+first_file+prefix2+second_file.replace('.sgm',''))
test_paths = []
prefix = 'corpus/test/'
prefix2 = '/timex2norm/'
for first_file in os.listdir('corpus/test'):
    if first_file == '.DS_Store':
        continue
    for second_file in os.listdir(f'corpus/test/{first_file}/timex2norm'):
        if second_file == '.DS_Store':
            continue
        if fnmatch.fnmatch(second_file, '*.sgm'):
            test_paths.append(prefix+first_file+prefix2+second_file.replace('.sgm',''))

#prepare train_data and test_data
train_docs = process_raw_text(train_paths)
test_docs = process_raw_text(test_paths)
train_ners, train_relations = get_named_entites_and_relations(train_paths)
test_ners, test_relations = get_named_entites_and_relations(test_paths)
train_data = construct_sentence_data(train_docs, train_ners, train_relations)
train_data, dev_data = train_test_split(train_data, train_size = .9, random_state = 0)
test_data = construct_sentence_data(test_docs, test_ners, test_relations)

#update data with bert embeddings
for i,paragraph in enumerate(train_data):
    if i % 20 == 0:
        print(f'using bert to get the contexualize embeddings for {i}th document', end="\r")
    for sentence in paragraph:
        sentence.update_embeddings(tokenizer, bert)
for paragraph in test_data:
    for sentence in paragraph:
        sentence.update_embeddings(tokenizer, bert)
for paragraph in dev_data:
    for sentence in paragraph:
        sentence.update_embeddings(tokenizer, bert)


with open('train_data', 'wb') as fb:
  pickle.dump(train_data, fb)
with open('test_data', 'wb') as fb:
  pickle.dump(test_data, fb)
with open('dev_data', 'wb') as fb:
  pickle.dump(dev_data, fb)