import csv
from datetime import datetime
import html
import numpy as np
import re
import spacy
import sys
import torch
from sklearn.feature_extraction import FeatureHasher
from torch.utils.data import Dataset


class FTData(object):
    def __init__(self, config):
        self.config = config
        self.num_classes = config.num_classes
        self.train_data_path = os.path.join(config.data_dir, 'train.csv')
        self.test_data_path = os.path.join(config.data_dir, 'test.csv')
        self.max_len = config.max_len
        self.n_over_max_len = 0
        self.real_max_len = 0
        self.valid_size_per_class = config.valid_size_per_class

        np.random.seed(config.seed)

        csv.field_size_limit(sys.maxsize)

        self.ngram2idx = dict()
        self.idx2ngram = dict()
        self.ngram2idx['PAD'] = 0
        self.idx2ngram[0] = 'PAD'
        self.ngram2idx['UNK'] = 1
        self.idx2ngram[1] = 'UNK'

        self.html_tag_re = re.compile(r'<[^>]+>')
        self.train_data, self.test_data = self.load_csv()

        self.hashed = False

        # except for PAD
        if len(self.ngram2idx) > 10 * 1000000 + 1:
            print(datetime.now(), 'Hashing Trick ... It may take long time.')
            self.hashing_trick()
            print(datetime.now(), 'Done')
            self.hashed = True

        if self.valid_size_per_class > 0:
            self.train_data, self.valid_data = \
                self.split_tr_va(n_class_examples=config.valid_size_per_class)
        self.count_labels()

        print('real_max_len', self.real_max_len)
        if self.n_over_max_len > 0:
            print('n_over_max_len {}/{} ({:.1f}%)'.
                  format(self.n_over_max_len, len(self.train_data),
                         100 * self.n_over_max_len / len(self.train_data)))

    def load_csv(self):
        train_data = list()
        test_data = list()

        # spacy.prefer_gpu()

        # https://spacy.io/usage/facts-figures#benchmarks-models-english
        # python3 -m spacy download en_core_web_lg --user
        nlp = spacy.load('en_core_web_lg', disable=['parser', 'tagger', 'ner'])
        nlp.add_pipe(nlp.create_pipe('sentencizer'))

        # train
        with open(self.train_data_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=",", quotechar='"')
            for idx, features in enumerate(reader):
                y = int(features[0]) - 1
                assert 0 <= y < self.num_classes, y

                if len(features) == 3:  # AG, Sogou, DBpedia, Amz P., Amz F.
                    x, x_len = self.process_example(features[1], features[2],
                                                    nlp,
                                                    is_train=True,
                                                    padding=args.padding)
                elif len(features) == 2:  # Yelp P., Yelp F.
                    x, x_len = self.process_example(features[1], None,
                                                    nlp,
                                                    is_train=True,
                                                    padding=args.padding)
                elif len(features) == 4:  # Yahoo A.
                    if features[2]:
                        f12 = features[1] + ' ' + features[2]
                    else:
                        f12 = features[1]
                    x, x_len = self.process_example(f12, features[3],
                                                    nlp,
                                                    is_train=True,
                                                    padding=args.padding)
                else:
                    raise ValueError

                train_data.append([x, x_len, y])

                if (idx + 1) % self.config.log_interval == 0:
                    print(datetime.now(), 'train', idx + 1)

        # test
        with open(self.test_data_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=",", quotechar='"')
            for idx, features in enumerate(reader):
                y = int(features[0]) - 1
                assert 0 <= y < self.num_classes, y

                if len(features) == 3:  # AG, Sogou, DBpedia, Amz P., Amz F.
                    x, x_len = self.process_example(features[1], features[2],
                                                    nlp,
                                                    is_train=False,
                                                    padding=args.padding)
                elif len(features) == 2:  # Yelp P., Yelp F.
                    x, x_len = self.process_example(features[1], None,
                                                    nlp,
                                                    is_train=False,
                                                    padding=args.padding)
                elif len(features) == 4:  # Yahoo A.
                    if features[2]:
                        f12 = features[1] + ' ' + features[2]
                    else:
                        f12 = features[1]
                    x, x_len = self.process_example(f12, features[3],
                                                    nlp,
                                                    is_train=False,
                                                    padding=args.padding)
                else:
                    raise ValueError

                test_data.append([x, x_len, y])

                if (idx + 1) % self.config.log_interval == 0:
                    print(datetime.now(), 'test', idx + 1)

        print('dictionary size (before hashing) {}'.format(len(self.ngram2idx)))

        return train_data, test_data

    def process_example(self, text0, text1, nlp, is_train=True, padding=0):

        text0 = text0.strip()  # Sogou

        if text0 and text0[-1] not in ['.', '?', '!']:
            text0 = text0 + '.'

        # TODO to handle /n (yelp p., yelp f.)

        # concat
        if text1:
            text1 = text1.strip()  # DBpedia
            title_desc = text0 + ' ' + text1
        else:
            title_desc = text0

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

        try:
            tagged_title_desc = \
                '<p> ' + ' </s> '.join([s.text for s in doc.sents]) + \
                ' </p>'
        except ValueError:
            # print(title_desc, e)
            tagged_title_desc = '<p> ' + title_desc + ' </p>'

        doc = nlp(tagged_title_desc)
        n_gram = get_ngram([token.text for token in doc],
                           n=self.config.n_gram)
        b_o_ngrams = b_o_w + n_gram

        ngs_len = len(b_o_ngrams)

        if self.max_len < ngs_len:

            # limit max len
            if padding > 0:
                b_o_ngrams = b_o_ngrams[:self.max_len]

            self.n_over_max_len += 1

        if self.real_max_len < ngs_len:
            self.real_max_len = ngs_len

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
        if padding > 0:
            while len(x) < self.max_len:
                x.append(self.ngram2idx['PAD'])
            assert len(x) == self.max_len

        return x, x_len

    def hashing_trick(self):

        def lst2dict(lst):
            count_dict = dict()
            for i in lst:
                if str(i) in count_dict:
                    count_dict[str(i)] += 1
                else:
                    count_dict[str(i)] = 1
            return count_dict

        def set_hashed(hasher, examples):
            f = hasher.transform((lst2dict(e[0]) for e in examples))
            for htr, e in zip(f, examples):
                sprs2hbow = list()
                for idc, d in zip(htr.indices, htr.data):
                    for _ in range(int(d)):
                        sprs2hbow.append(idc + 1)
                e[0] = sprs2hbow

        # bigram 10M, otherwise 100M
        n_features = 10 * 1000000 if self.config.n_gram == 2 else 100 * 1000000
        h = FeatureHasher(n_features=n_features, alternate_sign=False)
        print('FeatureHasher #features', n_features)

        set_hashed(h, self.train_data)
        set_hashed(h, self.test_data)

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
        if self.valid_size_per_class > 0:
            print('valid', count(self.valid_data))
        print('test ', count(self.test_data))

    def get_dataloaders(self, batch_size=32, shuffle=True, num_workers=4,
                        pin_memory=True):
        train_loader = torch.utils.data.DataLoader(
            FTDataset(self.train_data),
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self.batchify,
            pin_memory=pin_memory
        )

        valid_loader = None
        if self.valid_size_per_class > 0:
            valid_loader = torch.utils.data.DataLoader(
                FTDataset(self.valid_data),
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=self.batchify,
                pin_memory=pin_memory
            )

        test_loader = torch.utils.data.DataLoader(
            FTDataset(self.test_data),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self.batchify,
            pin_memory=pin_memory
        )

        return train_loader, valid_loader, test_loader

    def split_tr_va(self, n_class_examples=1900):
        count = 0
        class_item_set_dict = dict()
        item_all = list()

        print('Splitting..')
        while count < n_class_examples * self.num_classes:
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

    def batchify(self, b):
        x_len = [e[1] for e in b]
        batch_max_len = max(x_len)

        x = list()
        y = list()
        for e in b:
            while len(e[0]) < batch_max_len:
                e[0].append(self.ngram2idx['PAD'])
            assert len(e[0]) == batch_max_len

            x.append(e[0])
            y.append(e[2])

        x = torch.tensor(x, dtype=torch.int64)
        x_len = torch.tensor(x_len, dtype=torch.int64)
        y = torch.tensor(y, dtype=torch.int64)

        return x, x_len, y

    def batchify_multihot(self, b):
        i = list()
        for eidx, e in enumerate(b):
            for ev in e[0][:e[1]]:
                i.append([eidx, ev])
        v = torch.ones(len(i))
        i = torch.LongTensor(i)
        x = torch.sparse.FloatTensor(i.t(), v,
                                     torch.Size([len(b),
                                                 len(self.ngram2idx)]))\
            .to_dense()

        y = torch.tensor([e[2] for e in b], dtype=torch.int64)

        return x, y


def get_ngram(words, n=2):
    # TODO add ngrams up to n
    return [' '.join(words[i: i+n]) for i in range(len(words)-(n-1))]


class FTDataset(Dataset):
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
    parser.add_argument('--data_dir', type=str,
                        default='./data/ag_news_csv/')
    parser.add_argument('--pickle_name', type=str,
                        default='ag.pkl')
    parser.add_argument('--num_classes', type=int,
                        default=4)
    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--valid_size_per_class', type=int, default=0)
    parser.add_argument('--n_gram', type=int, default=2)
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--max_len', type=int, default=467)  #
    parser.add_argument('--log_interval', type=int, default=10000)
    args = parser.parse_args()

    pprint.PrettyPrinter().pprint(args.__dict__)

    import os
    pickle_path = os.path.join(args.data_dir, args.pickle_name)
    if os.path.exists(pickle_path):
        print(datetime.now(), 'Found an existing pickle', pickle_path)
        with open(pickle_path, 'rb') as f_pkl:
            ftdata = pickle.load(f_pkl)

        print('max len', ftdata.max_len)
        print('real max len', ftdata.real_max_len)
        print('vocab size', len(ftdata.ngram2idx))
    else:
        ftdata = FTData(args)
        with open(pickle_path, 'wb') as f_pkl:
            pickle.dump(ftdata, f_pkl, protocol=4)

    tr_loader, _, _ = ftdata.get_dataloaders(batch_size=256, num_workers=4)
    # print(len(tr_loader.dataset))
    for batch_idx, batch in enumerate(tr_loader):
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(tr_loader):
            print(datetime.now(), 'batch', batch_idx + 1)
