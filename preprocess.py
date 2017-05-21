# encoding=utf-8

from collections import Counter

import nltk
import os
import re
import random


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

MAX_WORD_PER_SENTENCE = 20
MAX_SENTENCE_PER_DOC = 20
MIN_FREQ_WORD_NUM = 5


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
    print('number of vocabulary：%s' % len(counter))
    print('number of frequency more than %d：%s' % (retain_freq_num, freq(retain_freq_num)))

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
        discard = False
        for sentence in doc:
            n_sentence = []
            words = clean_str(sentence).split(" ")
            # if any sentence's length is over  MAX_WORD_PER_SENTENCE,
            # discard the whole document for simplicity
            if len(words) > MAX_WORD_PER_SENTENCE:
                discard = True
                break
            for word in words:
                word = word.strip()
                if word:
                    n_sentence.append(word)
                    counter[word] += 1
            if n_sentence:
                document.append(n_sentence)
        # only accept document that has more than one sentence and less than MAX_SENTENCE_PER_DOC,
        # again, for simplicity's sake
        if 1 < len(document) <= MAX_SENTENCE_PER_DOC and not discard:
            new_documents.append(document)
    return new_documents, counter


def read(dir_path):
    sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
    documents = []
    for filename in os.listdir(dir_path):
        if filename.startswith('.'):
            continue
        file_path = os.path.join(dir_path, filename)
        with open(file_path, 'r', encoding='utf-8') as fp:
            line = fp.readline()
            sentences = sent_tokenizer.tokenize(line)
            documents.append(sentences)
    return documents


def write_doc(pos_docs, neg_docs, vocab, filename):
    docs = [(1, doc) for doc in pos_docs] + [(0, doc) for doc in neg_docs]
    len_to_data = {}
    for doc in docs:
        doc_len = len(doc[1])
        if doc_len in len_to_data:
            len_to_data[doc_len].append(doc)
        else:
            len_to_data[doc_len] = [doc]
    for value in len_to_data.values():
        random.shuffle(value)
    keys = list(len_to_data.keys())
    sorted_docs = []
    for key in sorted(keys):
        sorted_docs.extend(len_to_data[key])
    with open(filename, 'w') as f:
        for content in sorted_docs:
            line = '%d:' % content[0]
            for sentence in content[1]:
                sentence = [str(vocab[word]) for word in sentence]
                line += ','.join(sentence) + '#'
            f.write(line[:-1]+'\n')
        f.flush()


def write_vocab(vocab, vocab_file):
    with open(vocab_file, 'w') as f:
        for word, index in vocab.items():
            f.write(word+' '+str(index)+'\n')


def pre_process(pos_dir, neg_dir, save_dir):
    pos = read(pos_dir)
    neg = read(neg_dir)
    pos_processed, neg_processed = process(pos, neg, MIN_FREQ_WORD_NUM)
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
    print('total number of documents: %s, pos: %s, neg: %s' %
          (len(all_docs), len(pos_processed), len(neg_processed)))
    print('max num of document sentences：%s' % max(doc_len))
    print('min num of document sentences：%s' % min(doc_len))
    print('avg num of document sentences：%s' % (float(sum(doc_len))/len(doc_len)))

    print('max num of sentence words：%s' % max(sentence_len))
    print('min num of sentence words：%s' % min(sentence_len))
    print('avg num of sentence words：%s' % (float(sum(sentence_len))/len(sentence_len)))

    write_doc(pos_processed, neg_processed, vocab, save_dir+'data.dat')
    write_vocab(vocab, save_dir+'vocab.txt')


if __name__ == '__main__':
    path = '/Users/carlxie/Downloads/data/train/'
    pre_process(path+'neg/', path+'pos/', path)
