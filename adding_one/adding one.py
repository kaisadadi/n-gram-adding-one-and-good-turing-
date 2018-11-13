# adding-one smoothing  unigram & bi-gram by wk
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

train_data_dir = r"F:\NLP-lesson\dataset\train"
valid_data_dir = r"F:\NLP-lesson\dataset\valid"
test_data_dir = r"F:\NLP-lesson\dataset\testB"

BOS = "<BOS>"  # 开始符
EOS = "<EOS>"  # 结束符

unigram_f = {}
bigram_f = {}
add_unigram = {}
add_bigram = {}
V = 0  # unique word number
N_u = 0  # 总的词数量
num_sentence = 0


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
            if (item, p_list[idx + 1]) not in bigram_f.keys():
                bigram_f[(item, p_list[idx + 1])] = 1
            else:
                bigram_f[(item, p_list[idx + 1])] += 1



def unigram_prob(word):
    cnt = 0
    if word in unigram_f.keys():
        cnt += unigram_f[word] + 1.0
    else:
        cnt += 1.0
    return cnt / (N_u + V)


def bigram_prob(word):  # 此处word是词对
    global num_sentence
    cnt = 0
    if word in bigram_f.keys():
        cnt += bigram_f[word] + 1.0
    else:
        cnt += 1.0
    if word[0] == BOS:
        return cnt / (V + num_sentence)
    if word[0] in unigram_f.keys():
        return cnt / (V + unigram_f[word[0]])
    else:
        return cnt / (V + 1)


def test():
    # 需要adding-one的统计完毕之后，再进行prob的计算
    uni_p = []
    bi_p = []
    test_dataset = dataset("sentence", False, True, test_data_dir)
    for idx, word_list in enumerate(test_dataset.fetch_data()):
        p_list = [BOS]
        # 先计算unigram的概率
        prob = 1.0
        for item in word_list:
            print(item)
            print(unigram_prob(item))
            p_list.append(item)
            prob = prob * (unigram_prob(item) ** (1 / len(word_list)))
        uni_p.append(prob)
        # 再计算bigram的概率
        print("---------------")
        p_list.append(EOS)
        prob = 1.0
        for idx, item in enumerate(p_list[:-1]):
            print((item, p_list[idx + 1]))
            print(bigram_prob((item, p_list[idx + 1])))
            prob = prob * (bigram_prob((item, p_list[idx + 1])) ** (1 / len(word_list)))
        # prob = prob ** (1 / len(word_list))
        bi_p.append(prob)

    print(1 / (sum(uni_p) / len(uni_p)))
    print("--------------------")
    print(1 / (sum(bi_p) / len(bi_p)))


if __name__ == "__main__":
    train()
    test()
