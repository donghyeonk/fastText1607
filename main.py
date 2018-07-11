from datetime import datetime
import torch
from torch import nn, optim
import torch.nn.functional as F
from dataset import AGData


class fastText(nn.Module):
    def __init__(self, config):
        super(fastText, self).__init__()
        self.fc0 = nn.Linear(config.n_features, 10)
        self.fc1 = nn.Linear(10, config.num_classes)

        self.optimizer = optim.SGD(self.parameters(), lr=config.lr,
                                   weight_decay=config.wd)
        self.criterion = nn.NLLLoss()

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


def train(device, loader, model, epoch, config):
    model.train()
    train_loss = 0.
    example_count = 0
    for batch_idx, ex in enumerate(loader):
        targets = ex[1].to(device)
        model.optimizer.zero_grad()
        outputs = model(ex[0].float().to(device))
        loss = model.criterion(outputs, targets)
        loss.backward()
        model.optimizer.step()

        train_loss += len(outputs) * loss.item()
        example_count += len(targets)

        if (batch_idx + 1) % config.log_interval == 0 \
                or batch_idx == len(loader) - 1:
            _progress = \
                '\r{} Train Epoch {}, [{}/{} ({:.1f}%)]\tLoss: {:.6f}' \
                .format(datetime.now(), epoch,
                        example_count, len(loader.dataset),
                        100. * example_count / len(loader.dataset),
                        train_loss / example_count)
            print(_progress)

    train_loss /= len(loader.dataset)

    return train_loss


def test(device, loader, model, epoch):
    model.eval()
    eval_loss = 0.
    correct = 0
    with torch.no_grad():
        for batch_idx, ex in enumerate(loader):
            target = ex[1].to(device)
            output = model(ex[0].float().to(device))
            loss = model.criterion(output, target)
            eval_loss += len(output) * loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    eval_loss /= len(loader.dataset)
    print('\t{} Epoch {}\tLoss: {:.6f}, Accuracy: {}/{} ({:.1f}%)'.format(
        datetime.now(), epoch, eval_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))

    return eval_loss


def main():
    import argparse
    # import time
    # start_t = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="fastText_")
    parser.add_argument('--seed', type=int, default=2018)
    parser.add_argument('--data_path', type=str, default='./data/newsSpace')
    parser.add_argument('--num_classes', type=int, default=4)

    parser.add_argument('--n_grams', type=int, default=2)
    parser.add_argument('--n_features', type=int, default=2 ** 12)  # TODO
    parser.add_argument('--n_train_examples', type=int, default=30000)
    parser.add_argument('--n_test_examples', type=int, default=1900)

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=1e-5)  #
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=1000)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    ag_dataset = AGData(args)

    ft = fastText(args)
    train_loader, test_loader = \
        ag_dataset.get_dataloaders(batch_size=args.batch_size)

    for epoch in range(1, args.epochs + 1):
        train(device, train_loader, ft, epoch, args)
        test(device, test_loader, ft, epoch)


if __name__ == '__main__':
    main()
