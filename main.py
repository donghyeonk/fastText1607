from datetime import datetime
import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from dataset import AGData

# Reference
# https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/3%20-%20Faster%20Sentiment%20Analysis.ipynb


class FastText(nn.Module):
    def __init__(self, config):
        super(FastText, self).__init__()
        self.config = config

        self.bon_embed = nn.Embedding(config.vocab_size, config.embedding_dim,
                                      padding_idx=0)
        self.hidden = nn.Linear(config.embedding_dim, config.hidden_size)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

        self.init_linears()

        self.optimizer = optim.SGD(self.parameters(), lr=config.lr)
        self.criterion = nn.NLLLoss()

    def forward(self, x):
        # (batch size, max len) -> (max len, batch size)
        x = torch.transpose(x, 0, 1)

        # (max len, batch size) -> (max len, batch size, embedding_dim)
        embed = self.bon_embed(x)

        # (max len, batch size, embedding_dim)
        # -> (batch size, max len, embedding_dim)
        embed = embed.permute(1, 0, 2)

        # (batch size, max len, embedding_dim) -> (batch size, embedding_dim)
        pooled = F.avg_pool2d(embed, (embed.shape[1], 1)).squeeze(1)

        hdn = F.relu(self.hidden(pooled))

        # TODO hierarchical softmax

        return F.log_softmax(self.fc(hdn), dim=1)

    def init_linears(self):
        nn.init.xavier_uniform_(self.hidden.weight, gain=1)
        nn.init.uniform_(self.hidden.bias)
        nn.init.xavier_uniform_(self.fc.weight, gain=1)
        nn.init.uniform_(self.fc.bias)

    def lr_decay(self, epoch):
        next_lr = self.config.lr * (1. - epoch / self.config.epochs)
        print('Next learning rate: {:.3f}'.format(next_lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = next_lr


def train(device, loader, model, epoch, config):
    model.train()
    train_loss = 0.
    example_count = 0
    for batch_idx, ex in enumerate(loader):
        targets = torch.tensor(ex[1], dtype=torch.int64, device=device)
        model.optimizer.zero_grad()
        outputs = model(torch.tensor(ex[0], dtype=torch.int64, device=device))
        loss = model.criterion(outputs, targets)
        loss.backward()
        model.optimizer.step()

        batch_loss = len(outputs) * loss.item()
        train_loss += batch_loss
        example_count += len(targets)

        if (batch_idx + 1) % config.log_interval == 0 \
                or batch_idx == len(loader) - 1:
            _progress = \
                '\r{} Train Epoch {}, [{}/{} ({:.1f}%)],\tBatch Loss: {:.6f}' \
                .format(datetime.now(), epoch,
                        example_count, len(loader.dataset),
                        100. * example_count / len(loader.dataset),
                        batch_loss / len(outputs))
            print(_progress)
    train_loss /= len(loader.dataset)
    print('{} Train Epoch {}, Avg. Loss: {:.6f}'.format(datetime.now(), epoch,
                                                        train_loss))
    return train_loss


def evaluate(device, loader, model, epoch, mode):
    model.eval()
    eval_loss = 0.
    correct = 0
    with torch.no_grad():
        for batch_idx, ex in enumerate(loader):
            target = torch.tensor(ex[1], dtype=torch.int64, device=device)
            output = \
                model(torch.tensor(ex[0], dtype=torch.int64, device=device))
            loss = model.criterion(output, target)
            eval_loss += len(output) * loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    eval_loss /= len(loader.dataset)
    acc = correct / len(loader.dataset)
    print('{} {} Epoch {}, Avg. Loss: {:.6f}, '
          'Accuracy: {}/{} ({:.2f}%)'.format(datetime.now(), mode, epoch,
                                             eval_loss,
                                             correct, len(loader.dataset),
                                             100. * acc))
    return eval_loss, acc


def save_model(model, args, model_save_path):
    # save a model and args
    model_dict = dict()
    model_dict['state_dict'] = model.state_dict()
    model_dict['m_config'] = args
    model_dict['m_optimizer'] = model.optimizer.state_dict()
    # now_t = datetime.now().strftime("%Y%m%d_%H%M")[2:]
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    torch.save(model_dict, model_save_path)
    print('Saved', model_save_path)


def load_model(model, load_path):
    print('\t-> load checkpoint %s' % load_path)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['state_dict'])
    # model.optimizer.load_state_dict(checkpoint['m_optimizer'])


def main():
    import argparse
    import pickle
    import pprint
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='fastText1607')
    parser.add_argument('--checkpoint_dir', type=str, default='./ckpt/')
    parser.add_argument('--seed', type=int, default=2018)
    parser.add_argument('--data_path', type=str, default='./data/ag.pkl')
    parser.add_argument('--hidden_size', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=4)

    parser.add_argument('--n_grams', type=int, default=2)
    parser.add_argument('--embedding_dim', type=int, default=50)  #
    parser.add_argument('--n_train_examples', type=int, default=30000)
    parser.add_argument('--n_test_examples', type=int, default=1900)

    parser.add_argument('--lr', type=float, default=5e-1)  #
    parser.add_argument('--wd', type=float, default=0)  #
    parser.add_argument('--batch_size', type=int, default=256)  #
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--log_interval', type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    with open(args.data_path, 'rb') as f:
        ag_dataset = pickle.load(f)
    args_dict = vars(args)
    args_dict['vocab_size'] = len(ag_dataset.ngram2idx)
    pprint.PrettyPrinter().pprint(args.__dict__)
    train_loader, valid_loader, test_loader = \
        ag_dataset.get_dataloaders(batch_size=args.batch_size)

    ft = FastText(args).to(device)

    best_acc = 0.
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train(device, train_loader, ft, epoch, args)
        valid_loss, valid_acc = \
            evaluate(device, valid_loader, ft, epoch, 'Valid')
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_acc = valid_acc
            save_model(ft, args, os.path.join(args.checkpoint_dir,
                                              '{}.pth'.format(args.name)))
        else:
            # TODO early stopping
            pass

        print('\tLowest Loss {:.6f}, Acc. {:.2f}%'.format(best_loss,
                                                          100 * best_acc))

        # optional
        evaluate(device, test_loader, ft, epoch, 'Test')

        if epoch < args.epochs:
            ft.lr_decay(epoch)

    # load the best
    load_model(ft, os.path.join(args.checkpoint_dir,
                                '{}.pth'.format(args.name)))
    evaluate(device, test_loader, ft, args.epochs, 'Test')


if __name__ == '__main__':
    main()
