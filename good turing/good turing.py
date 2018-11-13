# good turing smoothing  unigram & bi-gram by wk
'''1. 标点符号是否算作一个词。
    是
2. 如何进行分句
   接受按行分句或者按句号\问号\感叹号分句，请在文档中注明分句方式
3. 文档库大小
   按training set n-gram的总数量（unigram计算1-gram, bigram计算2-gram数量）
4. 数据输出格式
   每一行输出两个结果，先输出unigram，后输出bigram
5. 关于单词词性
   可将同一单词不同词性看做同一个词或者不同词，请在文档中注明具体选择'''

import os
import numpy as np

train_data_dir = r"F:\NLP-lesson\dataset\train"
valid_data_dir = r"F:\NLP-lesson\dataset\valid"
test_data_dir = r"F:\NLP-lesson\dataset\testB"

BOS = "<BOS>"  # 开始符
EOS = "<EOS>"  # 结束符

MAXIMUM_N = 100000


unigram_f = {}
bigram_f = {}
add_unigram = {}
add_bigram = {}
V = 0  # unique word number
N_u = 0  # 总的词数量
num_sentence = 0
num_unseen = 0
num_unseen_V = 0
p_r = np.zeros([MAXIMUM_N], np.float32)
bigram_p_r = {}
num_unseen_bi = {}
count_happy = 0



class dataset:  # 用于每次发送一个完整的句子

    def __init__(self, mode, word, type, data_dir):
        # mode表示是一行还是句子，word表示是按照字还是划分好的词, type是否考虑词性（只对按词划分有效）
        self.mode = mode
        self.word = word
        self.data_dir = data_dir
        self.type = type

    def fetch_data(self):
        f = os.listdir(self.data_dir)
        if self.mode == "line":
            for file in f:
                ff = open(os.path.join(self.data_dir, file), "r", encoding="gbk")
                for cont in ff:
                    if self.word == True:  # 按照字分词，不受type影响
                        word_list = cont.split(" ")
                        while "" in word_list:
                            word_list.remove("")
                        charac_list = []
                        for word in word_list:
                            for charac in word:
                                if charac == '\n' or charac == '\\' or charac == '/':  # 非特殊符号
                                    continue
                                if charac >= u'\u4e00' and charac <= u'\u9fa5':  # 处理中文
                                    charac_list.append(charac)
                                elif (not ((charac >= u'\u0041' and charac <= u'\u005a') or (
                                        charac >= u'\u0061' and charac <= u'\u007a'))):  # 非字母
                                    charac_list.append(charac)
                        yield charac_list
                    else:  # 按照词分词
                        word_list = cont.split(" ")
                        while "" in word_list:
                            word_list.remove("")
                        while "\n" in word_list:
                            word_list.remove("\n")
                        if self.type == True:  # 考虑词性
                            yield word_list
                        else:  # 不考虑词性
                            charac_list = []
                            for word in word_list:
                                str = ""
                                for charac in word:
                                    if charac == '\n' or charac == '\\' or charac == '/':  # 非特殊符号
                                        continue
                                    if charac >= u'\u4e00' and charac <= u'\u9fa5':  # 处理中文
                                        str += charac
                                    elif (not ((charac >= u'\u0041' and charac <= u'\u005a') or (
                                            charac >= u'\u0061' and charac <= u'\u007a'))):  # 非字母
                                        str += charac
                                if str != "":
                                    charac_list.append(str)
                            yield (charac_list)
                ff.close()

        elif self.mode == "sentence":
            for file in f:
                ff = open(os.path.join(self.data_dir, file), "r", encoding="gbk")
                whole_word = []  # 整个文本内的全部数据
                for cont in ff:  # 提取一行数据
                    word_list = cont.split(" ")
                    for word in word_list:
                        whole_word.append(word)
                while "" in whole_word:
                    whole_word.remove("")
                while "\n" in whole_word:
                    whole_word.remove("\n")
                # 得到全部单词后，开始分句
                last_start = 0
                word_list = []
                for idx, word in enumerate(whole_word):
                    if '。' in word:
                        for k in range(last_start, idx + 1):
                            word_list.append(whole_word[k])
                        last_start = idx + 1
                        if self.word == True:  # 按照字分词
                            charac_list = []
                            for word in word_list:
                                for charac in word:
                                    if charac == '\n' or charac == '\\' or charac == '/':  # 非特殊符号
                                        continue
                                    if charac >= u'\u4e00' and charac <= u'\u9fa5':  # 处理中文
                                        charac_list.append(charac)
                                    elif (not ((charac >= u'\u0041' and charac <= u'\u005a') or (
                                                    charac >= u'\u0061' and charac <= u'\u007a'))):  # 非字母
                                        charac_list.append(charac)
                            yield charac_list
                        else:  # 按照词分词
                            if self.type == True:  # 考虑词性
                                yield word_list
                            else:  # 不考虑词性
                                charac_list = []
                                for word in word_list:
                                    str = ""
                                    for charac in word:
                                        if charac == '\n' or charac == '\\' or charac == '/':  # 非特殊符号
                                            continue
                                        if charac >= u'\u4e00' and charac <= u'\u9fa5':  # 处理中文
                                            str += charac
                                        elif (not ((charac >= u'\u0041' and charac <= u'\u005a') or (
                                                        charac >= u'\u0061' and charac <= u'\u007a'))):  # 非字母
                                            str += charac
                                    if str != "":
                                        charac_list.append(str)
                                yield (charac_list)
                        word_list = []
                    else:
                        continue
                ff.close()


def train():
    global V, N_u, unigram_f, bigram_f, num_sentence
    # 构建unigram_f和bigram_f
    train_dataset = dataset("sentence", False, True, train_data_dir)
    for idx, word_list in enumerate(train_dataset.fetch_data()):
        num_sentence += 1
        p_list = [BOS]
        # 统计unigram的词频
        for item in word_list:
            p_list.append(item)
            N_u += 1
            if item not in unigram_f.keys():
                V += 1
                unigram_f[item] = 1
            else:
                unigram_f[item] += 1

        # 统计bi-gram的词频，按照一行分句
        p_list.append(EOS)
        for idx, item in enumerate(p_list[:-1]):
            if item not in bigram_f.keys():
                bigram_f[item] = {}
                bigram_f[item][p_list[idx + 1]] = 1
            else:
                if p_list[idx + 1] not in bigram_f[item].keys():
                    bigram_f[item][p_list[idx + 1]] = 1
                else:
                    bigram_f[item][p_list[idx + 1]] += 1

def pre_good_turing():
    global num_unseen, num_unseen_V, num_unseen_bi
    more_uni = {}
    unigram_f[BOS] = num_sentence
    test_dataset = dataset("sentence", False, True, test_data_dir)
    # 先得到unseen的个数
    for idx, word_list in enumerate(test_dataset.fetch_data()):
        p_list = [BOS]
        #uni-gram
        for item in word_list:
            p_list.append(item)
            if item not in unigram_f.keys():
                num_unseen += 1
                if item not in more_uni.keys():
                    num_unseen_V += 1
                    more_uni[item] = 1
        #bi-gram
        p_list.append(EOS)
        for idx, item in enumerate(p_list[:-1]):
            if item not in unigram_f.keys():
                continue
            if item not in num_unseen_bi.keys():
                num_unseen_bi[item] = 0
            if p_list[idx + 1] not in bigram_f[item].keys():
                num_unseen_bi[item] += 1

def good_turing_unigram():
    global aver_uni_p, p_r, unigram_f
    #用于进行good-turing的概率处理
    #首先对uni-gram进行处理
    n_r= np.zeros([MAXIMUM_N], np.float32)  #记录n_r
    max_appear = 0
    cut_appear = 0
    for k in unigram_f.keys():
        n_r[unigram_f[k]] += 1
        if unigram_f[k] > max_appear:
            max_appear = unigram_f[k]
    for k in range(1, max_appear):   #除0修正
        if n_r[k] < 1:
            cut_appear = k-1
            break
    #计算r*
    r_star = np.zeros([MAXIMUM_N], np.float32)
    for k in range(1, cut_appear):
        r_star[k] = (k + 1) * n_r[k+1] / n_r[k]
    for k in range(cut_appear, max_appear + 1):
        if n_r[k] != 0:
            r_star[k] = k
        else:
            r_star[k] = 0
    #概率计算及归一化概率

    p_r[0] = num_unseen / N_u
    p_sum = p_r[0]
    for k in range(1, max_appear + 1):
        p_r[k] = r_star[k] / N_u
        p_sum += p_r[k]
    for k in range(max_appear + 1):
        p_r[k] = p_r[k] / p_sum


def good_turing_bigram():
    #与uni-gram的区别在于，每个词都要计算P_r，所以需要一个dict存下来
    global bigram_p_r, num_unseen_bi
    for key in unigram_f.keys():
        p_rr = np.zeros([30000], np.float32)
        n_r = np.zeros([30000], np.float32)
        max_appear = 0
        cut_appear = 0
        for k in bigram_f[key].keys():
            n_r[bigram_f[key][k]] += 1
            if bigram_f[key][k] > max_appear:
                max_appear = bigram_f[key][k]
        for k in range(1, max_appear):
            if n_r[k] < 1:
                cut_appear = k-1
                break
        # 计算r*
        r_star = np.zeros([30000], np.float32)
        for k in range(1, cut_appear):
            r_star[k] = (k + 1) * n_r[k + 1] / n_r[k]
        for k in range(cut_appear, max_appear + 1):
            if n_r[k] != 0:
                r_star[k] = k
            # 概率计算及归一化概率
        if key not in num_unseen_bi.keys():
            p_rr[0] = 0
        else:
            p_rr[0] = num_unseen_bi[key] / unigram_f[key]
        p_sum = p_rr[0]
        for k in range(1, max_appear + 1):
            p_rr[k] = r_star[k] / unigram_f[key]
            p_sum += p_rr[k]
        for k in range(max_appear + 1):
            p_rr[k] = p_rr[k] / p_sum
        bigram_p_r[key] = p_rr


def get_unigram(word):
    global num_unseen
    if word in unigram_f.keys():
        return p_r[unigram_f[word]]
    else:
        return p_r[0]

def get_bigram(word):
    global count_happy
    global bigram_p_r
    if word[0] in bigram_f.keys():
        if word[1] in bigram_f[word[0]].keys():
            return bigram_p_r[word[0]][bigram_f[word[0]][word[1]]]
        else:
            if bigram_p_r[word[0]][0] == 0:
                return 1
                #gg
            return bigram_p_r[word[0]][0]
    else:
        return 1 / N_u

def test():
    uni_p = []
    bi_p = []
    test_dataset = dataset("sentence", False, True, test_data_dir)
    for idx, word_list in enumerate(test_dataset.fetch_data()):
        prob = 1.0
        for item in word_list:
            print(item)
            print(get_unigram(item))
            prob = prob * (get_unigram(item) ** (1 / len(word_list)))
        uni_p.append(prob)
    print(1 / (sum(uni_p) / len(uni_p)))
    print("--------------------")
    for idx, word_list in enumerate(test_dataset.fetch_data()):
        p_list = [BOS]
        for word in word_list:
            p_list.append(word)
        p_list.append(EOS)
        prob = 1.0
        for idx, item in enumerate(p_list[:-1]):
            print((item, p_list[idx + 1]))
            print(get_bigram((item, p_list[idx+1])))
            prob = prob * (get_bigram((item, p_list[idx+1])) ** (1 / len(word_list)))
        bi_p.append(prob)
    print("ans=")
    print(1 / (sum(bi_p) / len(bi_p)))


if __name__ == "__main__":
    train()
    print("train finished")
    pre_good_turing()
    print("pre finised")
    good_turing_unigram()
    print("unigram finished")
    good_turing_bigram()
    print("bigram finished")
    test()
