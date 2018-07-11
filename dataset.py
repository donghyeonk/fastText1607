from sklearn.feature_extraction import FeatureHasher
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
        self.hasher = FeatureHasher(n_features=config.n_features,
                                    input_type='string')
        self.train_data, self.test_data = self.load()

    def load(self):
        train_data = list()
        test_data = list()

        cat_counts = [0] * self.num_classes
        ngram_set = set()

        line_cnt = 0
        errs = 0
        with open(self.data_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.read().split('\\N')
            for line in lines:
                line_cnt += 1
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

                bow = title_desc.split(' ')
                bow = ['<s>'] + bow + ['</s>']
                ngrams = get_ngrams(bow, n=self.config.n_grams)

                for w in bow:
                    ngram_set.add(w)

                for ng in ngrams:
                    ngram_set.add(ng)

                y = self.top_categories.index(category)
                if not is_test:
                    train_data.append([bow + ngrams, y])
                else:
                    test_data.append([bow + ngrams, y])

        print('lines', line_cnt)
        print('errs', errs)
        print('# of unique ngrams', len(ngram_set))

        f = self.hasher.transform(td[0] for td in (train_data+test_data))
        fh = f.toarray()
        print(fh.shape)

        for idx, tr_d in enumerate(train_data):
            tr_d[0] = fh[idx]

        for idx, te_d in enumerate(test_data):
            te_d[0] = fh[idx + self.n_train_examples * self.num_classes]

        return train_data, test_data

    def get_top_categories(self, topn=4):
        category_dict = dict()

        with open(self.data_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.read().split('\\N')
            for line in lines:
                cols = line.split('\t')
                if 6 > len(cols):
                    continue

                category = cols[4]

                if category in category_dict:
                    category_dict[category] += 1
                else:
                    category_dict[category] = 1

        top_categories = list()
        for cat in \
                sorted(category_dict,
                       key=category_dict.get, reverse=True)[:topn]:
            print(cat, category_dict[cat], sep='\t')
            top_categories.append(cat)
        return top_categories

    def get_dataloaders(self, batch_size=8, shuffle=True, num_workers=4):
        train_loader = torch.utils.data.DataLoader(
            AGDataset(self.train_data),
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )

        test_loader = torch.utils.data.DataLoader(
            AGDataset(self.test_data),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )
        return train_loader, test_loader


def get_ngrams(words, n=2):
    return ['_'.join(words[i: i+n]) for i in range(len(words)-(n-1))]


class AGDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


if __name__ == '__main__':
    import argparse
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
    # tr_loader, _ = agdata.get_dataloaders(batch_size=24, num_workers=4)
    # print(len(tr_loader.dataset))
    # for batch_idx, batch in enumerate(tr_loader):
    #     if batch_idx % 100 == 0:
    #         print(batch_idx)
