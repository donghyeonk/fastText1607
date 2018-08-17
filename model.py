import torch
import torch.nn.functional as F
from torch import nn


class FastTextMuitiHot(nn.Module):
    def __init__(self, config, hidden_size=10):
        super(FastTextMuitiHot, self).__init__()
        self.config = config

        self.linear = nn.Linear(self.config.vocab_size, hidden_size)
        self.fc = nn.Linear(hidden_size, config.num_classes)

        self.init_linears()

    def forward(self, x):
        # (batch_size, vocab_size) -> (batch_size, 10)
        x = self.linear(x)

        # (batch_size, 10) -> (batch size, num_classes)
        out = self.fc(x)

        # TODO hierarchical softmax

        return F.log_softmax(out, dim=1)

    def init_linears(self):
        nn.init.xavier_uniform_(self.linear.weight, gain=1)
        nn.init.uniform_(self.linear.bias)
        nn.init.xavier_uniform_(self.fc.weight, gain=1)
        nn.init.uniform_(self.fc.bias)

    def lr_decay(self, epoch, optimizer):
        # https://calculus.subwiki.org/wiki/Gradient_descent_with_decaying_learning_rate
        next_lr = self.config.lr / (1. + epoch)
        print('Next learning rate: {:.6f}'.format(next_lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = next_lr


class FastText(nn.Module):
    def __init__(self, config):
        super(FastText, self).__init__()
        self.config = config

        self.bon_embed = nn.Embedding(config.vocab_size, config.embedding_dim,
                                      padding_idx=0)
        self.fc = nn.Linear(config.embedding_dim, config.num_classes)

        self.init_linears()

        if config.use_bn > 0:
            self.bn = nn.BatchNorm1d(config.embedding_dim)

    def forward(self, x, x_len):
        # (batch size, max len) -> (batch size, max len, embedding_dim)
        embed = self.bon_embed(x)

        # (batch size, max len, embedding_dim) -> (batch size, embedding_dim)
        # Avg. embed.
        # Padded parts are zeros
        embed = torch.sum(embed, 1).squeeze(1)
        batch_size = embed.size(0)
        x_len = x_len.float().unsqueeze(1)
        x_len = x_len.expand(batch_size, self.config.embedding_dim)
        embed /= x_len

        if self.config.use_dropout > 0:
            embed = F.dropout(embed, p=0.5, training=self.training)

        if self.config.use_bn > 0:
            embed = self.bn(embed)

        out = self.fc(embed)

        # TODO hierarchical softmax

        return F.log_softmax(out, dim=1)

    def init_linears(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1)
        nn.init.uniform_(self.fc.bias)

    def lr_decay(self, epoch, optimizer):
        # https://calculus.subwiki.org/wiki/Gradient_descent_with_decaying_learning_rate
        next_lr = self.config.lr / (1. + epoch)
        print('Next learning rate: {:.6f}'.format(next_lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = next_lr
