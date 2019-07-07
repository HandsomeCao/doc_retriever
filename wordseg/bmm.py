# coding: utf-8
# 基于反向最大匹配的中文分词

import os

_get_abs_path = lambda path: os.path.normpath(os.path.join(os.getcwd(), path))

class BMM(object):
    def __init__(self,
                 dict_path='./dict.txt',
                 max_word_length=5,
                 split_mark='/'):
        self.dict = self.load_dict(_get_abs_path(dict_path))
        self.max_word_length = max_word_length
        self.split_mark = split_mark

    def load_dict(self, path):
        word_dict = []
        with open(path, 'r', encoding='utf-8') as fr:
            for line in fr:
                word = line.strip().split(' ')[0]
                word_dict.append(word)
        return word_dict

    def seg(self, line):
        """
        seg the line into word list
        """
        word_list = []
        s1 = line[-self.max_word_length:]
        while len(s1):
            if s1 in self.dict or len(s1) == 1:
                word_list.append(s1)
                s1 = line[:len(line)-len(''.join(word_list))]
            else:
                s1 = s1[1:]
        return self.split_mark.join(word_list[::-1])


if __name__ == "__main__":
    bmm = BMM()
    print(bmm.seg('今天是个好日子啊'))
