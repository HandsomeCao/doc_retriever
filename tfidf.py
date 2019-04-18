#coding: utf-8
# This file implement the TFIDF

import math
import numpy as np
from tqdm import tqdm
from collections import Counter
from utils import load_corpus, seg_line

questions, corpus = load_corpus('./dureader.txt')

class TFIDF(object):
    def __init__(self, corpus, vec_length=500):
        self.word_list = [seg_line(line) for line in corpus]
        self.count_list = [Counter(line) for line in self.word_list]
        self.vec_length = vec_length
        self.matrix = self.get_tfidf_matrix()

    def __len__(self):
        return len(self.count_list)


    def tf(self, word, count):
        return count[word] / sum(count.values())


    def n_containing(self, word, count_list):
        return sum(1 for count in count_list if word in count)


    def idf(self, word, count_list):
        return math.log(len(count_list) / (1 + self.n_containing(word, count_list)))


    def tfidf(self, word, count, count_list):
        return self.tf(word, count) * self.idf(word, count_list)

    def doc2vec(self, text):
        word_list = seg_line(text)
        count = Counter(word_list)
        return self.count2vec(count)


    def count2vec(self, count):
        vec = [self.tfidf(word, count, self.count_list) for word in count]
        vec = vec + (self.vec_length - len(vec)) * [0] \
                if len(vec) < self.vec_length else vec[: self.vec_length]
        return np.array(vec)



    def get_tfidf_matrix(self):
        matrix = []
        for i, count in enumerate(tqdm(self.count_list)):
            # scores = {word: tfidf(word, count, count_list) for word in count}
            vec = [self.tfidf(word, count, self.count_list) for word in count]
            vec = vec + (self.vec_length - len(vec)) * [0] \
                if len(vec) < self.vec_length else vec[: self.vec_length]
            matrix.append(vec)
        return np.array(matrix)


tfidf = TFIDF(corpus[:30])
print(tfidf.matrix.shape)
print(tfidf.matrix[:4])
