from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.batchnorm import _BatchNorm


class NodeLevelBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(NodeLevelBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.num_batches_tracked = None

    def _check_input_dim(self, the_input):
        if the_input.dim() != 2:
            raise ValueError('expected 2D the_input (got {}D the_input)'
                             .format(the_input.dim()))

    def forward(self, the_input):
        self._check_input_dim(the_input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        return torch.functional.F.batch_norm(
            the_input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)


class Encoder(nn.Module):
    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, dropout):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size"
        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        # self.device = torch.device('cuda:0')
        self.scale = torch.sqrt(torch.FloatTensor([0.5]))
        self.convs = nn.ModuleList(
            [nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size, padding=(kernel_size - 1) // 2) for _ in range(self.n_layers)]
        )  # convolutional layers
        self.do = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.gn = nn.GroupNorm(8, hid_dim * 2)
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self, protein):
        conv_input = self.fc(protein)
        conv_input = conv_input.permute(0, 2, 1)
        conved = None
        for i, conv in enumerate(self.convs):
            conved = conv(self.do(conv_input))
            conved = F.glu(conved, dim=1)
            conved = conved + conv_input * self.scale.to(protein.device)
            conv_input = conved

        conved = conved.permute(0, 2, 1)
        conved = self.ln(conved)
        return conved


class AttentionBlock(nn.Module):
    """ A class for attention mechanisn with QKV attention """

    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.f_q = nn.Linear(hid_dim, hid_dim)
        self.f_k = nn.Linear(hid_dim, hid_dim)
        self.f_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.f_q(query)
        K = self.f_k(key)
        V = self.f_v(value)

        Q = Q.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)
        K_T = K.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3).transpose(2, 3)
        V = V.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)

        energy = torch.matmul(Q, K_T) / self.scale.to(query.device)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        weighter_matrix = torch.matmul(attention, V)

        weighter_matrix = weighter_matrix.permute(0, 2, 1, 3).contiguous()

        weighter_matrix = weighter_matrix.view(batch_size, self.n_heads * (self.hid_dim // self.n_heads))

        weighter_matrix = self.do(self.fc(weighter_matrix))

        return weighter_matrix


# main model
class CSDTI(torch.nn.Module):
    def __init__(self, n_output, num_features_xd=22, num_features_xt=25,
                 n_filters=32, embed_dim=512, output_dim=128, dropout=0.1, n_layers=4, kernel_size=3, mlp_dropout=0.5, **config):
        super(CSDTI, self).__init__()
        self.n_filters = n_filters
        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        self.conv11 = GCNConv(num_features_xd, dim)
        self.conv12 = GCNConv(num_features_xd, dim)

        self.conv21 = GCNConv(dim * 2, dim)
        self.conv22 = GCNConv(dim * 2, dim)

        self.norm = NodeLevelBatchNorm(dim)

        self.fc1_xd = Linear(dim * 2, output_dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)  # embeding(26,128)

        self.conv_xt_1 = nn.Conv1d(in_channels=1200, out_channels=64, kernel_size=1)

        self.proteinencoder = Encoder(dropout=dropout, protein_dim=embed_dim, hid_dim=64,
                                      n_layers=n_layers, kernel_size=kernel_size)

        # self.fc1_xt = nn.Linear(64 * 512, output_dim)
        self.fc1_xt = nn.Linear(64 * 64, output_dim)

        # cross attention
        self.att = AttentionBlock(128, 1, 0.2)

        # output
        self.fc = nn.Sequential(
            # nn.Linear(256, 1024),
            nn.Linear(128*3, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_output)
        )

    def forward(self, data):
        data = data[0]
        x, edge_index = data.x, data.edge_index
        edge_index2 = data.edge_index2
        target = data.target

        # drug
        x1 = F.relu(self.conv11(x, edge_index))
        x2 = F.relu(self.conv12(x, edge_index2))

        x12 = torch.cat((x1, x2), dim=1)

        x1 = F.relu(self.conv21(x12, edge_index))
        x2 = F.relu(self.conv22(x12, edge_index2))

        x12 = torch.cat((x1, x2), dim=1)

        x = global_mean_pool(x12, data.batch)

        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.1, training=self.training)

        # protein
        embedded_xt = self.embedding_xt(target)
        xt = self.conv_xt_1(embedded_xt)
        xt = self.proteinencoder(xt)
        # xt = xt.view(-1, 64 * 512)
        xt = xt.view(-1, 64 * 64)
        xt = self.fc1_xt(xt)

        # cross attention
        att = self.att(x, xt, xt)

        # mix
        feature = torch.cat((x, att, xt), dim=1)

        # output
        out = self.fc(feature)

        return out
