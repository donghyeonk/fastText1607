import torch
from torch import nn, optim
import torch.nn.functional as F


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

    def forward(self, x, x_len):
        # (batch size, max len) -> (max len, batch size)
        x = torch.transpose(x, 0, 1)

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

        embed = F.elu(embed)
        embed = F.dropout(embed, training=self.training)
        hdn = F.elu(self.hidden(embed))
        hdn = F.dropout(hdn, training=self.training)
        hdn = self.fc(hdn)

        # TODO hierarchical softmax

        return F.log_softmax(hdn, dim=1)

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
