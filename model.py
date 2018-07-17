import torch
import torch.nn.functional as F
from torch import nn, optim


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

        # self.optimizer = \
        #     optim.SGD(self.parameters(), lr=config.lr,
        #               momentum=config.momentum)
        self.optimizer = \
            optim.Adam(self.parameters(), lr=config.lr, amsgrad=True)
        self.criterion = nn.CrossEntropyLoss()

        # self.scheduler = \
        #     optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
        #                                          factor=config.factor,
        #                                          patience=config.patience)

    def forward(self, x, x_len):
        # (batch size, max len) -> (max len, batch size)
        x.transpose_(0, 1)

        # (max len, batch size) -> (max len, batch size, embedding_dim)
        embed = self.bon_embed(x)

        # (max len, batch size, embedding_dim)
        # -> (batch size, max len, embedding_dim)
        embed = embed.permute(1, 0, 2)

        # (batch size, max len, embedding_dim) -> (batch size, embedding_dim)
        # Avg. embed.
        # Padded parts are zeros
        embed = torch.sum(embed, 1).squeeze(1)
        batch_size = x.size(1)
        x_len = x_len.float().unsqueeze(1)
        x_len = x_len.expand(batch_size, self.config.embedding_dim)
        embed /= x_len

        if self.config.use_dropout > 0:
            embed = F.dropout(embed, p=0.5, training=self.training)

        if self.config.use_bn > 0:
            embed = self.bn(embed)

        out = self.fc(embed)

        # TODO hierarchical softmax

        return out

    def init_linears(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1)
        nn.init.uniform_(self.fc.bias)

    def lr_decay(self, epoch):
        next_lr = self.config.lr * (1. - epoch / self.config.epochs)
        print('Next learning rate: {:.3f}'.format(next_lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = next_lr
