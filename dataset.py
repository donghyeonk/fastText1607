import csv
import html
import numpy as np
import re
import spacy
import torch
from torch.utils.data import Dataset


class AGData(object):
    def __init__(self, config):
        self.config = config
        self.num_classes = config.num_classes
        self.max_len = config.max_len

        np.random.seed(config.seed)

        # TODO hashing trick

        self.ngram2idx = dict()
        self.idx2ngram = dict()
        self.ngram2idx['PAD'] = 0
        self.idx2ngram[0] = 'PAD'
        self.ngram2idx['UNK'] = 1
        self.idx2ngram[1] = 'UNK'

        self.html_tag_re = re.compile(r'<[^>]+>')
        self.train_data, self.test_data = self.load_csv()
        self.train_data, self.valid_data = \
            self.divide_tr_va(n_class_examples=config.valid_size_per_class)
        self.count_labels()

    def load_csv(self):
        train_data = list()
        test_data = list()

        # https://spacy.io/usage/facts-figures#benchmarks-models-english
        # Run the following command on terminal
        # python3 -m spacy download en_core_web_lg
        nlp = spacy.load('en_core_web_lg', disable=['parser', 'tagger', 'ner'])
        nlp.add_pipe(nlp.create_pipe('sentencizer'))

        # train
        with open(self.config.train_data_path, 'r', newline='',
                  encoding='utf-8') as f:
            reader = csv.reader(f, quotechar='"')
            for idx, features in enumerate(reader):
                y = int(features[0]) - 1
                assert 0 <= y < self.num_classes, y
                x, x_len = self.process_example(features[1], features[2], nlp,
                                                is_train=True)
                train_data.append([x, x_len, y])

                if (idx + 1) % 10000 == 0:
                    print(idx + 1)

        # test
        with open(self.config.test_data_path, 'r', newline='',
                  encoding='utf-8') as f:
            reader = csv.reader(f, quotechar='"')
            for idx, features in enumerate(reader):
                y = int(features[0]) - 1
                assert 0 <= y < self.num_classes, y
                x, x_len = self.process_example(features[1], features[2], nlp,
                                                is_train=False)
                test_data.append([x, x_len, y])

        print('dictionary size', len(self.ngram2idx))

        return train_data, test_data

    def process_example(self, title, description, nlp, is_train=True,
                        padding=True):
        # concat
        title_desc = title + '. ' + description

        if '\\' in title_desc:
            title_desc = title_desc.replace('\\', ' ')

        # unescape html
        title_desc = html.unescape(title_desc)

        # remove html tags
        if '<' in title_desc and '>' in title_desc:
            title_desc = self.html_tag_re.sub('', title_desc)

        # create bow and bag-of-ngrams
        doc = nlp(title_desc)
        b_o_w = [token.text for token in doc]

        # add tags for ngrams
        tagged_title_desc = \
            '<p> ' + ' </s> '.join([s.text for s in doc.sents]) + \
            ' </p>'
        doc = nlp(tagged_title_desc)
        n_gram = get_ngram([token.text for token in doc],
                           n=self.config.n_gram)
        b_o_ngrams = b_o_w + n_gram

        # limit max len
        if padding:
            if self.max_len < len(b_o_ngrams):
                b_o_ngrams = b_o_ngrams[:self.max_len]

        # update dict.
        if is_train:
            for ng in b_o_ngrams:
                idx = self.ngram2idx.get(ng)
                if idx is None:
                    idx = len(self.ngram2idx)
                    self.ngram2idx[ng] = idx
                    self.idx2ngram[idx] = ng

        # assign ngram idxs
        x = [self.ngram2idx[ng] if ng in self.ngram2idx
             else self.ngram2idx['UNK']
             for ng in b_o_ngrams]

        x_len = len(x)

        # padding
        if padding:
            while len(x) < self.max_len:
                x.append(self.ngram2idx['PAD'])
            assert len(x) == self.max_len

        return x, x_len

    def count_labels(self):
        def count(data):
            count_dict = dict()
            for d in data:
                if d[-1] not in count_dict:
                    count_dict[d[-1]] = 1
                else:
                    count_dict[d[-1]] += 1

            return count_dict

        print('train', count(self.train_data))
        print('valid', count(self.valid_data))
        print('test ', count(self.test_data))

    def get_dataloaders(self, batch_size=32, shuffle=True, num_workers=4):
        train_loader = torch.utils.data.DataLoader(
            AGDataset(self.train_data),
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=batchify,
            pin_memory=True
        )

        valid_loader = torch.utils.data.DataLoader(
            AGDataset(self.valid_data),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=batchify,
            pin_memory=True
        )

        test_loader = torch.utils.data.DataLoader(
            AGDataset(self.test_data),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=batchify,
            pin_memory=True
        )
        return train_loader, valid_loader, test_loader

    def divide_tr_va(self, n_class_examples=1900):
        count = 0
        class_item_set_dict = dict()
        item_all = list()

        while count < n_class_examples * self.config.num_classes:
            rand_pick = np.random.randint(len(self.train_data))
            # print(rand_pick)
            label = self.train_data[rand_pick][-1]
            if label in class_item_set_dict:
                item_set = class_item_set_dict[label]
                if len(item_set) < n_class_examples \
                        and rand_pick not in item_set:
                    item_set.add(rand_pick)
                    item_all.append(rand_pick)
                    count += 1
            else:
                class_item_set_dict[label] = set()
                class_item_set_dict[label].add(rand_pick)
                item_all.append(rand_pick)
                count += 1

        train_data2 = list()
        valid_data = list()
        for idx, td in enumerate(self.train_data):
            if idx in item_all:
                valid_data.append(td)
            else:
                train_data2.append(td)

        print(len(train_data2), len(valid_data))
        return train_data2, valid_data


def get_ngram(words, n=2):
    # TODO add ngrams up to n
    return [' '.join(words[i: i+n]) for i in range(len(words)-(n-1))]


def batchify(b):
    x = [e[0] for e in b]
    x_len = [e[1] for e in b]
    y = [e[2] for e in b]

    x = torch.tensor(x, dtype=torch.int64)
    x_len = torch.tensor(x_len, dtype=torch.int64)
    y = torch.tensor(y, dtype=torch.int64)

    return x, x_len, y


class AGDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


if __name__ == '__main__':
    import argparse
    import pickle
    import pprint
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str,
                        default='./data/train.csv')
    parser.add_argument('--test_data_path', type=str,
                        default='./data/test.csv')
    parser.add_argument('--pickle_path', type=str, default='./data/ag.pkl')
    parser.add_argument('--seed', type=int, default=2018)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--valid_size_per_class', type=int, default=1900)
    parser.add_argument('--n_gram', type=int, default=2)
    parser.add_argument('--max_len', type=int, default=200)  #
    args = parser.parse_args()

    pprint.PrettyPrinter().pprint(args.__dict__)

    # import os
    # if os.path.exists(args.pickle_path):
    #     with open(args.pickle_path, 'rb') as f_pkl:
    #         agdata = pickle.load(f_pkl)
    # else:
    agdata = AGData(args)
    with open(args.pickle_path, 'wb') as f_pkl:
        pickle.dump(agdata, f_pkl)

    tr_loader, _, _ = agdata.get_dataloaders(batch_size=256, num_workers=4)
    # print(len(tr_loader.dataset))
    for batch_idx, batch in enumerate(tr_loader):
        if (batch_idx + 1) % 100 == 0:
            print('batch', batch_idx + 1)
