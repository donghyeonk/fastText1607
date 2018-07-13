from datetime import datetime
import spacy
import torch
from torch.utils.data import Dataset


class AGData(object):
    def __init__(self, config):
        self.config = config
        self.data_path = config.data_path
        self.n_train_examples = config.n_train_examples
        self.n_valid_examples = config.n_valid_examples
        self.n_test_examples = config.n_test_examples
        self.num_classes = config.num_classes
        self.top_categories = self.get_top_categories(topn=self.num_classes)
        self.max_len = config.max_len

        # TODO hashing trick
        self.ngram2idx = dict()
        self.idx2ngram = dict()
        self.ngram2idx['PAD'] = 0
        self.idx2ngram[0] = 'PAD'
        self.ngram2idx['UNK'] = 1
        self.idx2ngram[1] = 'UNK'

        self.train_data, self.valid_data, self.test_data = self.load()
        self.count_labels()

    def load(self):
        train_data = list()
        valid_data = list()
        test_data = list()

        cat_counts = [0] * self.num_classes

        # https://spacy.io/usage/facts-figures#benchmarks-models-english
        # Run the following command on terminal
        # python3 -m spacy download en_core_web_lg
        nlp = spacy.load('en_core_web_lg', disable=['parser', 'tagger', 'ner'])

        ngram_set = set()

        max_len_real = 0
        bon_lens = list()

        line_cnt = 0
        errs = 0
        with open(self.data_path, 'r', encoding='latin-1') as f:
            lines = f.read().split('\\N')
            for line_idx, line in enumerate(lines):
                line_cnt += 1

                cols = line.split('\t')
                if 6 > len(cols):
                    errs += 1
                    continue

                category = cols[4]
                if category not in self.top_categories:
                    continue

                category_idx = self.top_categories.index(category)
                cat_counts[category_idx] += 1
                if cat_counts[category_idx] > \
                        (self.n_train_examples + self.n_valid_examples +
                         self.n_test_examples):
                    continue

                is_valid = \
                    self.n_train_examples < cat_counts[category_idx] <= \
                    self.n_train_examples + self.n_valid_examples
                is_test = \
                    cat_counts[category_idx] > \
                    self.n_train_examples + self.n_valid_examples

                title = cols[2]
                description = cols[5]
                title_desc = title + ' ' + description

                # create bag-of-ngrams
                b_o_w = [token.text for token in nlp(title_desc)]
                ngram = get_ngram(b_o_w, n=self.config.n_gram)
                b_o_ngrams = b_o_w + ngram

                bon_len = len(b_o_ngrams)

                # stats
                if max_len_real < bon_len:
                    max_len_real = bon_len
                bon_lens.append(bon_len)
                for ng in b_o_ngrams:
                    ngram_set.add(ng)

                # limit max len
                if self.max_len < bon_len:
                    # self.max_len = len(b_o_ngrams)
                    b_o_ngrams = b_o_ngrams[:self.max_len]

                # update dict.
                for ng in b_o_ngrams:
                    idx = self.ngram2idx.get(ng)
                    if idx is None and not is_test:
                        idx = len(self.ngram2idx)
                        self.ngram2idx[ng] = idx
                        self.idx2ngram[idx] = ng

                # assign ngram idxs
                x = [self.ngram2idx[ng] if ng in self.ngram2idx
                     else self.ngram2idx['UNK']
                     for ng in b_o_ngrams]
                x_len = len(x)

                # padding
                while len(x) < self.max_len:
                    x.append(self.ngram2idx['PAD'])
                assert len(x) == self.max_len

                y = category_idx

                if is_test:
                    test_data.append(x + [x_len] + [y])
                elif is_valid:
                    valid_data.append(x + [x_len] + [y])
                else:
                    train_data.append(x + [x_len] + [y])

                if (len(train_data) + len(valid_data) + len(test_data)) \
                        % 10000 == 0:
                    print(datetime.now(), line_cnt,
                          len(train_data), len(valid_data), len(test_data))

        print('\nlines', line_cnt)
        print('errs', errs)
        print('# of unique ngrams', len(ngram_set))
        print('dictionary size', len(self.ngram2idx))
        print('max_len (setting)', self.max_len)
        print('max_len (real)', max_len_real)
        print('avg_len {:.1f}'.format(sum(bon_lens) / len(bon_lens)))
        print('max_len coverage {:.3f}'.format(
              sum([1 for bl in bon_lens if bl <= self.max_len]) / len(bon_lens)))

        return train_data, valid_data, test_data

    def get_top_categories(self, topn=4):
        category_dict = dict()

        with open(self.data_path, 'r', encoding='latin-1') as f:
            lines = f.read().split('\\N')
            for line in lines:
                # line = line.replace('\t\t', '\t')
                # if '\t\t' in line:
                #     pass
                cols = line.split('\t')
                if 6 > len(cols):
                    continue

                category = cols[4]
                if category in category_dict:
                    category_dict[category] += 1
                else:
                    category_dict[category] = 1

        print('Top {} categories'.format(topn))
        top_categories = list()
        for cat in \
                sorted(category_dict,
                       key=category_dict.get, reverse=True)[:topn]:
            print(cat, category_dict[cat], sep='\t')
            top_categories.append(cat)
        return top_categories

    def count_labels(self):
        def count(data):
            count_dict = dict()
            for d in data:
                cnt = count_dict.get(d[-1])
                if cnt is None:
                    count_dict[d[-1]] = 1
                else:
                    count_dict[d[-1]] = cnt + 1
            print(count_dict)

        count(self.train_data)
        count(self.valid_data)
        count(self.test_data)

    def get_dataloaders(self, batch_size=32, shuffle=True, num_workers=8):
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


def get_ngram(words, n=2):
    # TODO add ngrams up to n
    return [' '.join(words[i: i+n]) for i in range(len(words)-(n-1))]


def batchify(b):
    x = [e[:-2] for e in b]
    x_len = [e[-2] for e in b]
    y = [e[-1] for e in b]

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
    import os
    import pickle
    import pprint
    parser = argparse.ArgumentParser()
    # http://www.di.unipi.it/~gulli/newsspace200.xml.bz
    parser.add_argument('--data_path', type=str, default='./data/newsSpace')
    parser.add_argument('--pickle_path', type=str, default='./data/ag.pkl')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--n_gram', type=int, default=2)
    parser.add_argument('--n_train_examples', type=int, default=27000)
    parser.add_argument('--n_valid_examples', type=int, default=3000)
    parser.add_argument('--n_test_examples', type=int, default=1900)
    parser.add_argument('--max_len', type=int, default=200)
    args = parser.parse_args()

    pprint.PrettyPrinter().pprint(args.__dict__)

    if os.path.exists(args.pickle_path):
        with open(args.pickle_path, 'rb') as f_pkl:
            agdata = pickle.load(f_pkl)
    else:
        agdata = AGData(args)
        with open(args.pickle_path, 'wb') as f_pkl:
            pickle.dump(agdata, f_pkl)

    tr_loader, _, _ = agdata.get_dataloaders(batch_size=24, num_workers=8)
    print(len(tr_loader.dataset))
    for batch_idx, batch in enumerate(tr_loader):
        if (batch_idx + 1) % 1000 == 0:
            print(batch_idx + 1)
