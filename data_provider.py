# encoding=utf-8

from collections import Counter

import nltk
import os
import re


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

DOC_SEPARATOR = '___END___OF___DOCUMENT___'


def process(pos_docs, neg_docs, retain_freq_num):
    pos_docs_processed, pos_counter = split_to_words(pos_docs)
    neg_docs_processed, neg_counter = split_to_words(neg_docs)
    counter = pos_counter + neg_counter

    def freq(n):
        num = 0
        for k, v in counter.items():
            if v >= n:
                num += 1
        return num
    print('总共词汇数：%s' % len(counter))
    print('词频超过4的个数：%s' % freq(retain_freq_num))

    def process_doc(docs_processed):
        for doc_id in range(len(docs_processed)):
            for sen_id in range(len(docs_processed[doc_id])):
                for word_id in range(len(docs_processed[doc_id][sen_id])):
                    word = docs_processed[doc_id][sen_id][word_id]
                    if counter[word] < retain_freq_num:
                        docs_processed[doc_id][sen_id][word_id] = '__UNK_WORD__'

    process_doc(pos_docs_processed)
    process_doc(neg_docs_processed)
    return pos_docs_processed, neg_docs_processed


def split_to_words(documents):
    new_documents = []
    counter = Counter()
    for doc in documents:
        document = []
        for sentence in doc:
            n_sentence = []
            words = clean_str(sentence).split(" ")
            for word in words:
                word = word.strip()
                if word:
                    n_sentence.append(word)
                    counter[word] += 1
            if n_sentence:
                document.append(n_sentence)
        if document:
            new_documents.append(document)
    return new_documents, counter


def read(dir_path):
    sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
    documents = []
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        with open(file_path, 'r', encoding='utf-8') as fp:
            line = fp.readline()
            sentences = sent_tokenizer.tokenize(line)
            documents.append(sentences)
    return documents


def write_doc(docs, vocab, filename):
    docs = sorted(docs, key=lambda x: len(x))
    with open(filename, 'w') as f:
        for doc in docs:
            for sentence in doc:
                sentence = [str(vocab[word]) for word in sentence]
                f.write(','.join(sentence)+'\n')
            f.write('\n')
        f.flush()


def write_vocab(vocab, vocab_file):
    with open(vocab_file, 'w') as f:
        for word, index in vocab.items():
            f.write(word+' '+str(index)+'\n')


def pre_process(pos_dir, neg_dir, save_dir):
    pos = read(pos_dir)
    neg = read(neg_dir)
    pos_processed, neg_processed = process(pos, neg, 4)
    word_index = 1
    vocab = {}
    for doc in pos_processed:
        for sen in doc:
            for word in sen:
                if word not in vocab:
                    vocab[word] = word_index
                    word_index += 1
    for doc in neg_processed:
        for sen in doc:
            for word in sen:
                if word not in vocab:
                    vocab[word] = word_index
                    word_index += 1

    all_docs = pos_processed + neg_processed
    doc_len = []
    sentence_len = []
    for doc in all_docs:
        doc_len.append(len(doc))
        for sen in doc:
            sentence_len.append(len(sen))
    print('最长文档句子数：%s' % max(doc_len))
    print('最短文档句子数：%s' % min(doc_len))
    print('平均文档句子数：%s' % (float(sum(doc_len))/len(doc_len)))

    print('最长句子词数：%s' % max(sentence_len))
    print('最短句子词数：%s' % min(sentence_len))
    print('平均句子词数：%s' % (float(sum(sentence_len))/len(sentence_len)))

    write_doc(pos_processed, vocab, save_dir+'train_pos.dat')
    write_doc(neg_processed, vocab, save_dir+'train_neg.dat')
    write_vocab(vocab, save_dir+'vocab.txt')


def load_data(pos_file, neg_file):
    pass


if __name__ == '__main__':
    # pre_process('D:/data/train/neg/', 'D:/data/train/pos/', 'D:/data/train/')
    pass
