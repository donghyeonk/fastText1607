from datetime import datetime
import torch
from torch import nn, optim
import torch.nn.functional as F
from dataset import AGData

# Reference
# https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/3%20-%20Faster%20Sentiment%20Analysis.ipynb


class fastText(nn.Module):
    def __init__(self, config):
        super(fastText, self).__init__()
        self.config = config

        self.bon_embed = nn.Embedding(config.vocab_size, config.embedding_dim,
                                      padding_idx=0)
        self.hidden = nn.Linear(config.embedding_dim, config.hidden_size)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

        self.optimizer = optim.SGD(self.parameters(), lr=config.lr)
        self.criterion = nn.NLLLoss()

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        embed = self.bon_embed(x)
        embed = embed.permute(1, 0, 2)
        pooled = F.avg_pool2d(embed, (embed.shape[1], 1)).squeeze(1)
        hdn = F.relu(self.hidden(pooled))
        return F.log_softmax(self.fc(hdn), dim=1)

    def lr_decay(self, epoch):
        next_lr = self.config.lr * (1. - epoch / self.config.epochs)
        print('learning rate is to be {:.3f}'.format(next_lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = next_lr


def train(device, loader, model, epoch, config):
    model.train()
    train_loss = 0.
    example_count = 0
    for batch_idx, ex in enumerate(loader):
        targets = torch.tensor(ex[1], dtype=torch.float64, device=device)
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
                '\r{} Train Epoch {}, [{}/{} ({:.1f}%)]\tLoss: {:.6f}' \
                .format(datetime.now(), epoch,
                        example_count, len(loader.dataset),
                        100. * example_count / len(loader.dataset),
                        batch_loss / len(outputs))
            print(_progress)
    train_loss /= len(loader.dataset)
    return train_loss


def test(device, loader, model, epoch):
    model.eval()
    eval_loss = 0.
    correct = 0
    with torch.no_grad():
        for batch_idx, ex in enumerate(loader):
            target = torch.tensor(ex[1], dtype=torch.float64, device=device)
            output = \
                model(torch.tensor(ex[0], dtype=torch.int64, device=device))
            loss = model.criterion(output, target)
            eval_loss += len(output) * loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    eval_loss /= len(loader.dataset)
    acc = correct / len(loader.dataset)
    print('{} Test Epoch {}, Loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)'.format(
        datetime.now(), epoch, eval_loss, correct, len(loader.dataset),
        100. * acc))
    return eval_loss, acc


def main():
    import argparse
    import pickle
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="fastText_")
    parser.add_argument('--seed', type=int, default=2018)
    parser.add_argument('--data_path', type=str, default='./data/ag.pkl')
    parser.add_argument('--hidden_size', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=4)

    parser.add_argument('--n_grams', type=int, default=2)
    parser.add_argument('--embedding_dim', type=int, default=100)  #
    parser.add_argument('--n_train_examples', type=int, default=30000)
    parser.add_argument('--n_test_examples', type=int, default=1900)

    parser.add_argument('--lr', type=float, default=5e-3)  #
    parser.add_argument('--wd', type=float, default=0)  #
    parser.add_argument('--batch_size', type=int, default=256)  #
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--log_interval', type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    with open(args.data_path, 'rb') as f:
        ag_dataset = pickle.load(f)

    args_dict = vars(args)
    args_dict['vocab_size'] = len(ag_dataset.ngram2idx)

    ft = fastText(args)
    train_loader, test_loader = \
        ag_dataset.get_dataloaders(batch_size=args.batch_size)

    best_acc = 0.
    for epoch in range(1, args.epochs + 1):
        train(device, train_loader, ft, epoch, args)
        _, acc = test(device, test_loader, ft, epoch)
        if acc > best_acc:
            best_acc = acc
        print('\tBest Acc. {:.2f}%'.format(100 * best_acc))
        ft.lr_decay(epoch)


if __name__ == '__main__':
    main()
