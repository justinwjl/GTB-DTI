from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm


class TDGraphDTA(nn.Module):
    def __init__(self,
                 block_num,
                 vocab_protein_size=26,
                 embedding_size=128,
                 filter_num=32,
                 n_output=1
                 , **config):
        super().__init__()
        self.protein_encoder = TargetRepresentation(block_num,
                                                    vocab_protein_size,
                                                    embedding_size)
        self.ligand_encoder = GraphDenseNet(num_input_features=78,  # 22,
                                            out_dim=filter_num * 3,
                                            block_config=[8, 8, 8],
                                            bn_sizes=[2, 2, 2])
        self.attn_modulex = InteractiveMultiHeadAttention(input_dim=filter_num * 3, num_heads=32)
        self.attn_moduley = InteractiveMultiHeadAttention(input_dim=filter_num * 3, num_heads=32)

        self.classifier = nn.Sequential(
            nn.Linear(filter_num * 3 * 2, 1024),
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
        target = data.target
        protein_x = self.protein_encoder(target)
        ligand_x = self.ligand_encoder(data)
        x1 = self.attn_modulex(protein_x, ligand_x)
        x2 = self.attn_moduley(ligand_x, protein_x)
        x = torch.cat([x1, x2], dim=-1)
        x = self.classifier(x)

        return x


class convLayerIndReLU(nn.Module):

    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            # Origin convLayerInd, maybe wrong, change to Conv1d
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.inc(x)


class LinearReLU(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Linear(in_features=in_features,
                      out_features=out_features,
                      bias=bias),
            nn.ReLU()
        )

    def forward(self, x):
        return self.inc(x)


class StackCNN(nn.Module):
    def __init__(self, layer_num,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        super().__init__()

        self.inc = nn.Sequential(OrderedDict([('conv_layer0',
                                               convLayerIndReLU(in_channels,
                                                                out_channels,
                                                                kernel_size=kernel_size,
                                                                stride=stride,
                                                                padding=padding))]))
        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1),
                                convLayerIndReLU(out_channels,
                                                 out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding))

        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):
        return self.inc(x).squeeze(-1)


class TargetRepresentation(nn.Module):
    def __init__(self, block_num,
                 vocab_size,
                 embedding_num):
        super().__init__()
        self.embed = nn.Embedding(vocab_size,
                                  embedding_num,
                                  padding_idx=0)
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNN(block_idx + 1,
                         embedding_num,
                         96,
                         3)
            )

        self.linear = nn.Linear(block_num * 96, 96)

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        feats = [block(x) for block in self.block_list]
        x = torch.cat(feats, -1)
        x = self.linear(x)

        return x


class NodeBN(_BatchNorm):

    def __init__(self, num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(NodeBN, self).__init__(
            num_features,
            eps, momentum,
            affine,
            track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
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
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)


class GraphConvBn(nn.Module):
    def __init__(self, in_channels,
                 hidden_channels,
                 out_channels):
        super().__init__()
        self.convLayerIn = gnn.GCNConv(in_channels, hidden_channels)
        self.convLayerOut = gnn.GCNConv(hidden_channels, out_channels)
        self.norm = NodeBN(out_channels)

        self.diffusion_iterations = 5
        self.diffusion_weight = 0.1

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        data.x = F.relu(self.norm(self.convLayerOut(self.convLayerIn(x, edge_index), edge_index)))

        return data


class DenseLayer(nn.Module):
    def __init__(self,
                 num_input_features,
                 growth_rate=32,
                 bn_size=4):
        super().__init__()
        self.convLayerIn = GraphConvBn(num_input_features,
                                       16,
                                       int(growth_rate * bn_size))
        self.convLayerOut = GraphConvBn(int(growth_rate * bn_size),
                                        32,
                                        growth_rate)

    def bn_function(self, data):
        concated_features = torch.cat(data.x, 1)
        data.x = concated_features

        data = self.convLayerIn(data)

        return data

    def forward(self, data):
        if isinstance(data.x, Tensor):
            data.x = [data.x]

        data = self.bn_function(data)
        data = self.convLayerOut(data)

        return data


class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers,
                 num_input_features,
                 growth_rate=32,
                 bn_size=4):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate,
                               growth_rate,
                               bn_size)
            self.add_module('layer%d' % (i + 1), layer)

    def forward(self, data):
        features = [data.x]
        for name, layer in self.items():
            data = layer(data)
            features.append(data.x)
            data.x = features

        data.x = torch.cat(data.x, 1)

        return data


class GraphDenseNet(nn.Module):
    def __init__(self,
                 num_input_features,
                 out_dim,
                 growth_rate=32,
                 block_config=(3, 3, 3, 3),
                 bn_sizes=[2, 3, 4, 4]):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([('convn', GraphConvBn(num_input_features, 16, 32))]))
        num_input_features = 32

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers,
                num_input_features,
                growth_rate=growth_rate,
                bn_size=bn_sizes[i]
            )
            self.features.add_module('block%d' % (i + 1), block)
            num_input_features = num_input_features + int(num_layers * growth_rate)

            trans = GraphConvBn(num_input_features, 32, num_input_features // 2)
            self.features.add_module("transition%d" % (i + 1), trans)
            num_input_features = num_input_features // 2

        self.classifer = nn.Linear(num_input_features, out_dim)

    def forward(self, data):
        data = self.features(data)
        x = gnn.global_mean_pool(data.x, data.batch)
        x = self.classifer(x)

        return x


class InteractiveMultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, l2_reg=0.01):
        super(InteractiveMultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.l2_reg = l2_reg

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim, input_dim)

        self.head_dim = input_dim // num_heads

        self.regularization = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x1, x2):
        q1 = self.query(x1)
        k2 = self.key(x2)
        v2 = self.value(x2)

        q1 = q1.view(q1.size(0),
                     self.num_heads,
                     -1,
                     self.head_dim)
        k2 = k2.view(k2.size(0),
                     self.num_heads,
                     -1,
                     self.head_dim)
        v2 = v2.view(v2.size(0),
                     self.num_heads,
                     -1,
                     self.head_dim)

        attention_scores = torch.matmul(q1, k2.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float().to(x1.device))
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, v2)
        attention_output = attention_output.view(attention_output.size(0), -1)

        output = self.regularization(self.fc(attention_output))

        # output += x1
        output = output + x1
        output = nn.functional.layer_norm(output, output.size()[1:])

        return output
