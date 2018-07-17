from datetime import datetime
import os
import torch
from dataset import AGData
from model import FastText


def train(device, loader, model, epoch, config):
    model.train()
    train_loss = 0.
    example_count = 0
    correct = 0
    for batch_idx, ex in enumerate(loader):
        target = ex[2].to(device)
        model.optimizer.zero_grad()
        output = model(ex[0].to(device), ex[1].to(device))
        loss = model.criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_max_norm)
        model.optimizer.step()

        batch_loss = len(output) * loss.item()
        train_loss += batch_loss
        example_count += len(target)

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

        if (batch_idx + 1) % config.log_interval == 0 \
                or batch_idx == len(loader) - 1:
            _progress = \
                '\r{} Train Epoch {}, [{}/{} ({:.1f}%)],\tBatch Loss: {:.6f}' \
                .format(datetime.now(), epoch,
                        example_count, len(loader.dataset),
                        100. * example_count / len(loader.dataset),
                        batch_loss / len(output))
            print(_progress)
    train_loss /= len(loader.dataset)
    acc = correct / len(loader.dataset)
    print('{} Train Epoch {}, Avg. Loss: {:.6f}, Accuracy: {}/{} ({:.1f}%)'.
          format(datetime.now(), epoch, train_loss,
                 correct, len(loader.dataset), 100. * acc))
    return train_loss


def evaluate(device, loader, model, epoch, mode):
    model.eval()
    eval_loss = 0.
    correct = 0
    with torch.no_grad():
        for batch_idx, ex in enumerate(loader):
            target = ex[2].to(device)
            output = model(ex[0].to(device), ex[1].to(device))
            loss = model.criterion(output, target)
            eval_loss += len(output) * loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    eval_loss /= len(loader.dataset)
    acc = correct / len(loader.dataset)
    print('{} {} Epoch {}, Avg. Loss: {:.6f}, '
          'Accuracy: {}/{} ({:.1f}%)'.format(datetime.now(), mode, epoch,
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
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    torch.save(model_dict, model_save_path)
    print('Saved', model_save_path)


def load_model(model, load_path):
    print('\t-> load checkpoint %s' % load_path)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.optimizer.load_state_dict(checkpoint['m_optimizer'])


def main():
    import argparse
    import pickle
    import pprint
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='fastText1607')
    parser.add_argument('--checkpoint_dir', type=str, default='./ckpt/')
    parser.add_argument('--seed', type=int, default=2018)
    parser.add_argument('--data_path', type=str, default='./data/ag.pkl')
    parser.add_argument('--num_classes', type=int, default=4)

    parser.add_argument('--n_grams', type=int, default=2)
    parser.add_argument('--embedding_dim', type=int, default=10)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=9e-1)  # SGD
    parser.add_argument('--wd', type=float, default=0)  #
    parser.add_argument('--factor', type=float, default=0.5)  # lr_scheduler
    parser.add_argument('--patience', type=float, default=2)  # lr_scheduler
    parser.add_argument('--grad_max_norm', type=float, default=5.)  #
    parser.add_argument('--use_bn', type=int, default=0)  # bad..
    parser.add_argument('--use_dropout', type=int, default=1)  # good~
    parser.add_argument('--batch_size', type=int, default=256)  #
    parser.add_argument('--epochs', type=int, default=5 * 6)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--yes_cuda', type=int, default=1)
    args = parser.parse_args()

    use_cuda = args.yes_cuda > 0 and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    print('CUDA device_count {0}'.format(torch.cuda.device_count())
          if use_cuda else 'CPU')

    with open(args.data_path, 'rb') as f:
        ag_dataset = pickle.load(f)
    args_dict = vars(args)
    args_dict['vocab_size'] = len(ag_dataset.ngram2idx)
    pprint.PrettyPrinter().pprint(args.__dict__)
    train_loader, valid_loader, test_loader = \
        ag_dataset.get_dataloaders(batch_size=args.batch_size)
    print(len(train_loader.dataset), len(valid_loader.dataset),
          len(test_loader.dataset))

    ft = FastText(args).to(device)

    best_loss = float('inf')
    best_acc = 0.
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        print()
        train(device, train_loader, ft, epoch, args)

        # valid
        valid_loss, valid_acc = \
            evaluate(device, valid_loader, ft, epoch, 'Valid')
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_acc = valid_acc
            best_epoch = epoch
            save_model(ft, args, os.path.join(args.checkpoint_dir,
                                              '{}.pth'.format(args.name)))
        else:
            # TODO early stopping
            pass

        print('\tLowest Valid Loss {:.6f}, Acc. {:.1f}%, Epoch {}'.
              format(best_loss, 100 * best_acc, best_epoch))
        ft.scheduler.step(valid_loss)

        # optional
        evaluate(device, test_loader, ft, epoch, 'Test')

        # if epoch < args.epochs:
        #     ft.lr_decay(epoch)

    # load the best
    load_model(ft, os.path.join(args.checkpoint_dir,
                                '{}.pth'.format(args.name)))
    evaluate(device, test_loader, ft, best_epoch, 'Test')


if __name__ == '__main__':
    main()
