import spacy
import torch
from torch.utils.data import Dataset


class AGData(object):
    def __init__(self, config):
        self.config = config
        self.data_path = config.data_path
        self.n_train_examples = config.n_train_examples
        self.n_test_examples = config.n_test_examples
        self.num_classes = config.num_classes
        self.top_categories = self.get_top_categories(topn=self.num_classes)
        # TODO feature hashing
        self.ngram2idx = dict()
        self.idx2ngram = dict()
        self.ngram2idx['PAD'] = 0
        self.idx2ngram[0] = 'PAD'

        self.max_len = 200

        self.nlp = spacy.load('en_core_web_sm')
        self.train_data, self.test_data = self.load()

    def load(self):
        train_data = list()
        test_data = list()

        cat_counts = [0] * self.num_classes
        ngram_set = set()

        max_len_real = 0

        line_cnt = 0
        errs = 0
        with open(self.data_path, 'r', encoding='latin-1') as f:
            lines = f.read().split('\\N')
            for line_idx, line in enumerate(lines):
                line_cnt += 1

                if line_cnt % 100000 == 0:
                    print(line_cnt)
                
                cols = line.split('\t')
                if 6 > len(cols):
                    errs += 1
                    continue

                category = cols[4]
                if category not in self.top_categories:
                    continue

                cat_counts[self.top_categories.index(category)] += 1
                if cat_counts[self.top_categories.index(category)] > \
                        (self.n_train_examples + self.n_test_examples):
                    continue

                is_test = \
                    cat_counts[self.top_categories.index(category)] > \
                    self.n_train_examples

                title = cols[2]
                description = cols[5]
                title_desc = title + ' ' + description

                # b_o_w = ['<s>'] + title_desc.split(' ') + ['</s>']
                b_o_w = \
                    ['<s>'] \
                    + [token.text for token in self.nlp(title_desc)] \
                    + ['</s>']
                ngram = get_ngrams(b_o_w, n=self.config.n_grams)
                b_o_ngrams = b_o_w + ngram
                for ng in b_o_ngrams:
                    ngram_set.add(ng)
                    idx = self.ngram2idx.get(ng)
                    if idx is None:
                        idx = len(self.ngram2idx)
                        self.ngram2idx[ng] = idx
                        self.idx2ngram[idx] = ng

                bon_len = len(b_o_ngrams)

                if self.max_len < bon_len:
                    # self.max_len = len(b_o_ngrams)
                    b_o_ngrams = b_o_ngrams[:self.max_len]

                if max_len_real < bon_len:
                    max_len_real = bon_len

                x = [self.ngram2idx[ng] for ng in b_o_ngrams]
                while len(x) < self.max_len:
                    x.append(self.ngram2idx['PAD'])
                assert len(x) == self.max_len

                y = self.top_categories.index(category)

                if not is_test:
                    train_data.append(x + [y])
                else:
                    test_data.append(x + [y])

        print('\nlines', line_cnt)
        print('errs', errs)
        print('# of unique ngrams', len(ngram_set))
        print('max_len (setting)', self.max_len)
        print('max_len (real)', max_len_real)

        return train_data, test_data

    def get_top_categories(self, topn=4):
        category_dict = dict()

        with open(self.data_path, 'r', encoding='latin-1') as f:
            lines = f.read().split('\\N')
            for line in lines:
                # line = line.replace('\t\t', '\t')
                if '\t\t' in line:
                    pass
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

    def get_dataloaders(self, batch_size=32, shuffle=True, num_workers=4):
        train_loader = torch.utils.data.DataLoader(
            AGDataset(self.train_data),
            shuffle=shuffle,
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
        return train_loader, test_loader


def get_ngrams(words, n=2):
    return [' '.join(words[i: i+n]) for i in range(len(words)-(n-1))]


def batchify(b):
    x = [e[:-1] for e in b]
    y = [e[-1] for e in b]
    return x, y


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
    parser = argparse.ArgumentParser()
    # http://www.di.unipi.it/~gulli/newsspace200.xml.bz
    parser.add_argument('--data_path', type=str, default='./data/newsSpace')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--n_grams', type=int, default=2)
    parser.add_argument('--n_features', type=int, default=2 ** 12)  # TODO
    parser.add_argument('--n_train_examples', type=int, default=30000)
    parser.add_argument('--n_test_examples', type=int, default=1900)
    args = parser.parse_args()
    agdata = AGData(args)

    with open('./data/ag.pkl', 'wb') as f_pkl:
        pickle.dump(agdata, f_pkl)

    tr_loader, _ = agdata.get_dataloaders(batch_size=24, num_workers=4)
    print(len(tr_loader.dataset))
    for batch_idx, batch in enumerate(tr_loader):
        if batch_idx % 1000 == 0:
            print(batch_idx)
