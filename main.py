import argparse
from datetime import datetime
import os
import pickle
import pprint
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from dataset import FTData
from model import FastText

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='fastText1607')
parser.add_argument('--checkpoint_dir', type=str, default='./ckpt/')
parser.add_argument('--seed', type=int, default=2019)
parser.add_argument('--data_path', type=str,
                    default='./data/ag_news_csv/ag.pkl')

parser.add_argument('--n_grams', type=int, default=2)
parser.add_argument('--embedding_dim', type=int, default=10)

parser.add_argument('--lr', type=float, default=5e-1,
                    help='0.05, 0.1, 0.25, 0.5')
parser.add_argument('--momentum', type=float, default=5e-1)  # SGD
parser.add_argument('--wd', type=float, default=0)

parser.add_argument('--grad_max_norm', type=float, default=0)
parser.add_argument('--use_bn', type=int, default=0)
parser.add_argument('--use_dropout', type=int, default=0)

parser.add_argument('--batch_size', type=int, default=4096)  #
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--yes_cuda', type=int, default=0)
parser.add_argument('--num_processes', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--num_threads', type=int, default=20)


def train_epoch(device, loader, model, epoch, optimizer, config):
    model.train()
    pid = os.getpid()
    train_loss = 0.
    example_count = 0
    correct = 0
    start_t = datetime.now()
    for batch_idx, ex in enumerate(loader):
        target = ex[2].to(device)
        optimizer.zero_grad()
        output = model(ex[0].to(device), ex[1].to(device))
        loss = F.nll_loss(output, target)
        loss.backward()
        if config.grad_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           config.grad_max_norm)
        optimizer.step()

        batch_loss = len(output) * loss.item()
        train_loss += batch_loss
        example_count += len(target)

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

        if (batch_idx + 1) % config.log_interval == 0 \
                or batch_idx == len(loader) - 1:
            _progress = \
                'pid {}, {} Train Epoch {}, [{}/{} ({:.1f}%)],' \
                '\tBatch Loss: {:.6f}' \
                .format(pid, datetime.now(), epoch,
                        example_count, len(loader.dataset),
                        100. * example_count / len(loader.dataset),
                        batch_loss / len(output))
            print(_progress)
    train_loss /= len(loader.dataset)
    acc = correct / len(loader.dataset)
    print('{} Train Epoch {}, Avg. Loss: {:.6f}, Accuracy: {}/{} ({:.1f}%)'.
          format(datetime.now()-start_t, epoch, train_loss,
                 correct, len(loader.dataset), 100. * acc))
    return train_loss


def evaluate_epoch(device, loader, model, epoch, mode):
    model.eval()
    eval_loss = 0.
    correct = 0
    start_t = datetime.now()
    with torch.no_grad():
        for batch_idx, ex in enumerate(loader):
            target = ex[2].to(device)
            output = model(ex[0].to(device), ex[1].to(device))
            loss = F.nll_loss(output, target)
            eval_loss += len(output) * loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    eval_loss /= len(loader.dataset)
    acc = correct / len(loader.dataset)
    print('{} {} Epoch {}, Avg. Loss: {:.6f}, '
          'Accuracy: {}/{} ({:.1f}%)'.format(datetime.now()-start_t, mode,
                                             epoch, eval_loss,
                                             correct, len(loader.dataset),
                                             100. * acc))
    return eval_loss, acc


def save_model(model, optimizer, args, model_save_path):
    # save a model and args
    model_dict = dict()
    model_dict['state_dict'] = model.state_dict()
    model_dict['m_config'] = args
    model_dict['optimizer'] = optimizer.state_dict()
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    torch.save(model_dict, model_save_path)
    print('Saved', model_save_path)


def load_model(model, optimizer, load_path):
    print('\t-> load checkpoint %s' % load_path)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# https://github.com/pytorch/examples/tree/master/mnist_hogwild
def train(rank, device, model, args, use_cuda):
    torch.manual_seed(args.seed + rank)

    with open(args.data_path, 'rb') as f:
        ft_dataset = pickle.load(f)

    args_dict = vars(args)
    args_dict['num_classes'] = ft_dataset.num_classes
    if hasattr(ft_dataset, 'hashed') and ft_dataset.hashed:
        n_features = 10 * 1000000 if ft_dataset.config.n_gram == 2 \
            else 100 * 1000000
        args_dict['vocab_size'] = 1 + n_features  # PAD, #hasher features
    else:
        args_dict['vocab_size'] = len(ft_dataset.ngram2idx)
    print('real_max_len', ft_dataset.real_max_len)

    train_loader, valid_loader, test_loader = \
        ft_dataset.get_dataloaders(batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   pin_memory=use_cuda)
    print(len(train_loader.dataset), len(valid_loader.dataset),
          len(test_loader.dataset))

    # optimizer = optim.SGD(model.parameters(), lr=args.lr,
    #                       momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    for epoch in range(1, args.epochs + 1):
        train_epoch(device, train_loader, model, epoch, optimizer, args)
        evaluate_epoch(device, valid_loader, model, epoch, 'Valid')
        evaluate_epoch(device, test_loader, model, epoch, 'Test')


def hog_wild():
    args = parser.parse_args()

    assert args.yes_cuda == 0

    pprint.PrettyPrinter().pprint(args.__dict__)

    use_cuda = args.yes_cuda > 0 and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    print('CUDA device_count {0}'.format(torch.cuda.device_count())
          if use_cuda else 'CPU')
    print(torch.get_num_threads())

    model = FastText(args).to(device)
    model.share_memory()

    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(rank, device, model, args, use_cuda))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def main():
    args = parser.parse_args()

    print('torch version', torch.__version__)

    use_cuda = args.yes_cuda > 0 and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    print('CUDA device_count {0}'.format(torch.cuda.device_count())
          if use_cuda else 'CPU')

    torch.set_num_threads(args.num_threads)
    print('#threads', torch.get_num_threads())

    with open(args.data_path, 'rb') as f:
        ft_dataset = pickle.load(f)

    args_dict = vars(args)
    args_dict['num_classes'] = ft_dataset.num_classes
    if hasattr(ft_dataset, 'hashed') and ft_dataset.hashed:
        n_features = 10 * 1000000 if ft_dataset.config.n_gram == 2 \
            else 100 * 1000000
        args_dict['vocab_size'] = 1 + n_features  # PAD, #hasher features
    else:
        args_dict['vocab_size'] = len(ft_dataset.ngram2idx)
    print('real_max_len', ft_dataset.real_max_len)

    pprint.PrettyPrinter().pprint(args.__dict__)
    train_loader, valid_loader, test_loader = \
        ft_dataset.get_dataloaders(batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   pin_memory=use_cuda)
    print('train size', len(train_loader.dataset))
    if valid_loader is not None:
        print('valid size', len(valid_loader.dataset))
    print('test size', len(test_loader.dataset))

    model = FastText(args).to(device)

    # optimizer = optim.SGD(model.parameters(), lr=args.lr,
    #                       weight_decay=args.wd, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.wd, amsgrad=True)

    best_acc = 0.
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        print()
        train_epoch(device, train_loader, model, epoch, optimizer, args)

        # valid
        if valid_loader is not None:
            valid_loss, valid_acc = \
                evaluate_epoch(device, valid_loader, model, epoch, 'Valid')
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_epoch = epoch
                save_model(model, optimizer, args,
                           os.path.join(args.checkpoint_dir,
                                        '{}.pth'.format(args.name)))
            # TODO early stopping

            print('\tHighest Valid Accuracy {:.2f}%, Epoch {}'.
                  format(100 * best_acc, best_epoch))

        # optional
        evaluate_epoch(device, test_loader, model, epoch, 'Test')

        # learning rate decay
        if epoch < args.epochs:
            model.lr_decay(epoch, optimizer)

    # load the best
    if valid_loader is not None:
        load_model(model, optimizer, os.path.join(args.checkpoint_dir,
                                                  '{}.pth'.format(args.name)))
        evaluate_epoch(device, test_loader, model, best_epoch, 'Test')


if __name__ == '__main__':
    main()
    # hog_wild()
