from collections import OrderedDict

# from transformer import Transformer
import torch.nn as nn
import torch_sparse
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from torch_geometric.utils import to_dense_adj

from utils import *


class IMAEN(torch.nn.Module):
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 k1=1, k2=2, k3=3, embed_dim=128, num_layer=1, embedding_num=128,
                 block_num=3, embedding_size=128,
                 num_feature_xt=25, vocab_protein_size=25 + 1, vocab_size=26, filter_num=32, num_feature_xd=156,
                 n_output=1, output_dim=128, dropout=0.2,
                 conditional=False, conditional_size=0, **config):
        super(IMAEN, self).__init__()
        self.gcnnet = GCNNet(k1, k2, k3, embed_dim, num_layer, embedding_num, block_num, embedding_size,
                             num_feature_xt, vocab_protein_size, vocab_size, filter_num, num_feature_xd,
                             n_output, output_dim, dropout)
        self.vae = VAE(encoder_layer_sizes, latent_size, decoder_layer_sizes,
                       conditional, conditional_size)

        # Prevent VAE from being trained， original code, but need to verify if this leads to better results.
        for param in self.vae.parameters():
            param.requires_grad = False

    def forward(self, data):
        return self.gcnnet(data, self.vae)


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            # nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            # nn.Linear(hidden_size, 1, bias=False)
        )

    # 128,224 -> 224,16 -> 16,1
    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        # b = beta.tolist()
        # title = ['1']
        # t = pd.DataFrame(data=b)
        # t.to_csv('D:\MGASDTI\data/ac.csv')
        return (beta * z), beta
        # return (beta * z).sum(1), beta


class GCNNet(torch.nn.Module):
    def __init__(self, k1, k2, k3, embed_dim, num_layer, embedding_num=128, block_num=3, embedding_size=128,
                 num_feature_xt=25, vocab_protein_size=25 + 1, vocab_size=26, filter_num=32, num_feature_xd=156,
                 n_output=1, output_dim=128, dropout=0.2):
        super(GCNNet, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        # Smile graph branch
        self.Conv1 = GCNConv(num_feature_xd, num_feature_xd)
        self.Conv2 = GCNConv(num_feature_xd, num_feature_xd * 2)
        self.Conv3 = GCNConv(num_feature_xd * 2, num_feature_xd * 4)
        # self.lagcn = LAGCN(2, 128, 128, 40, 3, 0.5).to(device)
        self.embed_dim = embed_dim
        self.num_layer = num_layer
        # self.ligand_encoder = GraphDenseNet(num_input_features=78, out_dim=filter_num * 3, block_config=[8, 8, 8], bn_sizes=[2, 2, 2])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # cfg: {input_dim, hidden_dim, pos_encode, dropout, num_heads, feedforward_dim, num_layers, num_classes}
        self.fc_g1 = nn.Linear(1092, 1024)
        self.fc_g2 = nn.Linear(1024, output_dim)
        self.attention = Attention(128)
        # protien sequence branch (Transformer)
        # self.PE = PositionalEncodings(dim=1000,dropout=0.2)
        self.embedding_xt = nn.Embedding(num_feature_xt + 1, embed_dim, padding_idx=0)
        self.protein_encoder = TargetRepresentation(block_num, vocab_protein_size, embedding_size)
        #                                               3          26                   128
        # self.protein_encoder = Transformer()

        # combined layers
        # self.fc1 = nn.Linear(224, 1024)
        # self.fc1 = nn.Linear(128, 1024)
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, n_output)
        self.tran = nn.Linear(1000, 128)

    def forward(self, data, vae):
        data = data[0]
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target
        X_list = get_augmented_features(x, vae)
        X_list = torch.cat((X_list, x), dim=1)
        adj = to_dense_adj(edge_index)
        # LSTM layer
        # ligand_x = self.ligand_encoder(data)   # 128 96
        if self.k1 == 1:
            h1 = self.Conv1(X_list, edge_index)

            h1 = self.relu(h1)

            h2 = self.Conv2(h1, edge_index)

            h2 = self.relu(h2)

            h3 = self.Conv3(h2, edge_index)

            h3 = self.relu(h3)

        if self.k2 == 2:
            edge_index_square, _ = torch_sparse.spspmm(edge_index, None, edge_index, None, adj.shape[1], adj.shape[1],
                                                       adj.shape[1], coalesced=True)
            h4 = self.Conv1(X_list, edge_index_square)
            h4 = self.relu(h4)
            h5 = self.Conv2(h4, edge_index_square)
            h5 = self.relu(h5)

        if self.k3 == 3:
            edge_index_cube, _ = torch_sparse.spspmm(edge_index_square, None, edge_index, None, adj.shape[1],
                                                     adj.shape[1],
                                                     adj.shape[1], coalesced=True)
            h6 = self.Conv1(X_list, edge_index_cube)
            h6 = self.relu(h6)

        concat = torch.cat([h3, h5, h6], dim=1)  # (4115, 546)

        x = gmp(concat, batch)  # global_max_pooling   (128, 546)

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)  # (128, 128)
        x, att = self.attention(x)
        # LSTM layer
        # embedded_xt = self.embedding_xt(target).permute(0, 2, 1)  # 128,128,1000
        # X = torch.cat((X_list, data.x),dim=1)  # ..,206
        # x = self.lagcn(X_list+[data.x], adj)
        # x = self.lagcn(X, adj)
        embedded_xt = self.embedding_xt(target).permute(0, 2, 1)  # 128,128,1000
        # embedded_xt = self.PE(embedded_xt)

        # step1 计算xt注意力
        # selfatt = torch.tanh(embedded_xt)
        # att = F.softmax(selfatt, dim=1)
        # btt = att.numpy()
        # df = pd.DataFrame(btt)
        # df.to_csv('D:\MGASDTI\data\processed/weight.csv')
        # a = np.savetxt('D:\MGASDTI\data\processed/weight.csv', att.detach().numpy(), fmt='%.2f', delimiter=',')
        # b = att.tolist()
        # title = ['1']
        # t = pd.DataFrame(data=b)
        # t.to_csv('D:\MGASDTI\data/aa.csv')

        # b = pd.DataFrame(data=list)
        # b.to_csv('D:\MGASDTI\data\processed')
        # selfatt = att.transpose(1, 2)
        # self_att = torch.bmm(selfatt, embedded_xt)

        # step2 正则化（做不做都可以，看模型效果）

        # self_att = F.normalize(self_att, p=2, dim=-1)   # all can  128 1000 128

        ###########################################

        # selfatt = torch.tanh(x)
        # att = F.softmax(selfatt, dim=1)
        # print(att.shape,'***')
        # selfatt = att.transpose(1, 2)
        # print(selfatt.shape)
        # self_att = torch.bmm(selfatt, embedded_xt)
        # step2 正则化（做不做都可以，看模型效果）
        # self_x = F.normalize(self_att, p=2, dim=-1)  # all can
        # weight1 = torch.sigmoid(self_att)
        # weight2 = torch.sigmoid(self_x)
        # weight1 = weight1 / (weight1 + weight2)
        # weight2 = 1 - weight1
        # xt = weight1*self_att
        # x = weight2*self_x
        # doc = weight2 * self_x + weight1* self_att
        # self_att = self.tran(self_att)
        # self_att = self_att.transpose(1, 2)

        xt = self.protein_encoder(embedded_xt)  # 128 96
        # xt = self.protein_encoder(self_att)   # 128 96

        # concat
        # xc = torch.cat((x, xt), 1)  # 128 192
        # xc = torch.cat((x, xt), 1)  # 128 192
        # xc = torch.stack([x, xt], dim=1)  # 128 192
        xt, att = self.attention(xt)
        # sava_occ = att.data.cpu()
        # add some dense layers
        xc = torch.cat((x, xt), 1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


class TargetRepresentation(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_num):
        super().__init__()
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):  # 0, 1, 2
            self.block_list.append(
                StackCNN(block_idx + 1, embedding_num, 96, 3)  # 1, 2, 3
            )
        # self.linear = nn.Linear(block_num * 96, 96)
        self.linear = nn.Linear(block_num * 96, 128)

    def forward(self, x):
        feats = [block(x) for block in self.block_list]
        x = torch.cat(feats, -1)
        x = self.linear(x)

        return x


class Conv1dReLU(nn.Module):
    '''
    kernel_size=3, stride=1, padding=1
    kernel_size=5, stride=1, padding=2
    kernel_size=7, stride=1, padding=3
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.inc(x)


class StackCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.inc = nn.Sequential(OrderedDict([('conv_layer0',
                                               Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size,
                                                          stride=stride, padding=padding))]))
        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1),
                                Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=padding))

        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):
        return self.inc(x).squeeze(-1)


def get_augmented_features(x, vae):
    # X_list = []
    # for _ in range(concat):
    z = torch.randn([x.size(0), vae.latent_size]).to(x.device)
    augmented_features = vae.inference(z, x).detach()  # 3933 10 3933 78--->4164 128
    # X_list.append(augmented_features)
    return augmented_features  # 4036 128


class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, conditional_size=0):
        super().__init__()

        if conditional:
            assert conditional_size > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, conditional_size)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, conditional_size)

    def forward(self, x, c=None):
        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def reparameterize(self, means, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return means + eps * std

    def inference(self, z, c=None):
        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, conditional_size):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += conditional_size

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="B{:d}".format(i), module=nn.BatchNorm1d(out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, conditional_size):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + conditional_size
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="B{:d}".format(i), module=nn.BatchNorm1d(out_size))
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="tanh", module=nn.Tanh())

    def forward(self, z, c):

        if self.conditional:
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x
