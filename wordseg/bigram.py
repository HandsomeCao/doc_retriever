# -*- coding:utf-8 -*-


import pickle
import os
import copy
import re


def _get_abs_path(path): return os.path.normpath(
    os.path.join(os.getcwd(), path))


class Bigram(object):
    def __init__(self, pkl_path='./data/data.pickle'):
        self.pkl_path = _get_abs_path(pkl_path)
        with open(self.pkl_path, 'rb') as fr:
            data = pickle.load(fr)
        self.word_count = data[0]
        self.word2_dict = data[1]
        self.length = sum(list(self.word_count.values()))  # 所有词语数量
        self.length += len(self.word_count)  # 平滑操作
        # 每个词后面的后继词的总数,用来计算概率
        self.word_next_count = {}
        for word in self.word2_dict:
            self.word_next_count[word] = 0
            for word_in in self.word2_dict[word]:
                self.word_next_count[word] += self.word2_dict[word][word_in]
        self.all_situation = []

    def _all_situation(self, context, start=0, result=[]):
        """
        得到所有分词可能性的情况，采用了递归方式
        """
        length = len(context)
        if start == length:
            self.all_situation.append(copy.deepcopy(result))
        else:
            flag = False
            if start + 6 <= length and context[start: start + 6] in self.word_count:
                flag = True
                result.append(context[start: start + 6])
                self._all_situation(context, start + 6, result)
                result.pop()
                return

            if start + 5 <= length and context[start: start + 5] in self.word_count:
                flag = True
                result.append(context[start: start + 5])
                self._all_situation(context, start + 5, result)
                result.pop()
                return

            if start + 4 <= length and context[start: start + 4] in self.word_count:
                flag = True
                result.append(context[start: start + 4])
                self._all_situation(context, start + 4, result)
                result.pop()

            if start + 3 <= length and context[start: start + 3] in self.word_count:
                flag = True
                result.append(context[start: start + 3])
                self._all_situation(context, start + 3, result)
                result.pop()

            if start + 2 <= length and context[start: start + 2] in self.word_count:
                flag = True
                result.append(context[start: start + 2])
                self._all_situation(context, start + 2, result)
                result.pop()

            if context[start] in self.word_count:
                flag = True
                result.append(context[start: start + 1])
                self._all_situation(context, start + 1, result)
                result.pop()

            if flag == False:
                result.append(context[start: start + 1])
                self._all_situation(context, start + 1, result)
                result.pop()

    def _pretreatment(self, context):
        tmp = []
        for line in context:
            if len(line):
                tmp.append(line)
        return tmp

    def _get_prob(self, result):
        """
        计算每种情况的概率分数
        """
        p = 1.0
        for index in range(len(result)):
            if index == 0:
                if result[index] in self.word_count:
                    # 第一项a在词频表中，p = (count(a) + 1) / count(all)
                    p *= ((self.word_count[result[index]] + 1) / self.length)
                else:
                    # 若不在则 p = 1 / count(all)
                    p *= (1 / self.length)
            else:
                if result[index - 1] in self.word2_dict and \
                        result[index] in self.word2_dict[result[index - 1]]:
                    # 若前一项a存在于gram表，且下一项b存在于它对应的value中
                    # p = (count(b|a) + 1) / count(|a)
                    p *= ((self.word2_dict[result[index - 1]][result[index]] + 1) /
                          self.word_next_count[result[index - 1]])
                elif result[index - 1] in self.word2_dict:
                    # 若前一项a存在于gram表，但下一项b不存在于它对应的value中
                    # p = 1 / count(|a)
                    p *= (1 / self.word_next_count[result[index - 1]])
                else:
                    p = p * pow(0.1, 10)
        return p

    def _cut(self, context):
        self._all_situation(context)
        max_prob = 0
        for result in self.all_situation:
            p = self._get_prob(result)
            if p > max_prob:
                max_prob = p
                max_result = result
        # 清除之后给下次使用
        self.all_situation.clear()
        return max_result

    def cut(self, context):
        result = []
        # 以标点符号分割开
        context = re.split(
            '：|-|/|【|？|】|\?|。|，|\.|、|《|》| |（|）|”|“|；|\n', context)
        context = self._pretreatment(context)
        for line in context:
            result += self._cut(line)
        return result

    @classmethod
    def preprocess_pkl(cls, train_file, output_path):
        word_count, word2_dict = {}, {}
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                split_line = line.strip().split()
                for index in range(len(split_line)):
                    if split_line[index] in word_count:
                        word_count[split_line[index]] += 1
                    else:
                        word_count[split_line[index]] = 1

                    if index == len(split_line) - 1:
                        break
                    # bigram
                    if split_line[index] in word2_dict:
                        if split_line[index + 1] in word2_dict[split_line[index]]:
                            word2_dict[split_line[index]
                                       ][split_line[index + 1]] += 1
                        else:
                            word2_dict[split_line[index]
                                       ][split_line[index + 1]] = 1
                    else:
                        word2_dict[split_line[index]] = {
                            split_line[index + 1]: 1}
        print("LEN COUNT:", len(word_count))
        print('LEN DICT:', len(word2_dict))
        with open(output_path, 'wb') as fw:
            pickle.dump((word_count, word2_dict), fw)
        print("Done")


if __name__ == "__main__":
    # print('Create the pkl file from trainset...')
    # train_file = _get_abs_path('./data/train_utf_8.txt')
    # output_file = _get_abs_path('./data/data.pickle')
    # Bigram.preprocess_pkl(train_file, output_file)
    string = "本报北京１２月３１日讯新华社记者陈雁、\
                  本报记者何加正报道：在度过了非凡而辉煌的１９９７年，\
                  迈向充满希望的１９９８年之际，’９８北京新年音乐会今晚在人民大会堂举行。\
                  党和国家领导人江泽民、李鹏、乔石、朱镕基、李瑞环、刘华清、尉健行、李岚清与万名首都各界群众和劳动模范代表一起，\
                  在激昂奋进的音乐声中辞旧迎新。"
    bigram = Bigram()
    print(bigram.cut(string))
