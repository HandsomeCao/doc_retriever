#coding: utf-8

import jieba

def load_corpus(filepath):
    corpus = []
    questions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            sample = line.split('\t')
            questions.append(sample[1].strip())
            corpus.append(sample[2].strip())
    return questions, corpus
            

def seg_line(line):
    return list(jieba.cut(line))


def get_stopwords(filepath):
    pass

def filter_stopwords():
    pass