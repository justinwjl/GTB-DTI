from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
import pickle

# SubMDTA_GINConv model
class SubMDTA(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2, **config):
        super(SubMDTA, self).__init__()

        num_features = config['num_features'] if 'num_features' in config else 78
        hidden_dim = config['hidden_dim'] if 'hidden_dim' in config else 128
        num_gc_layers = config['num_gc_layers'] if 'num_gc_layers' in config else 4
        mode = config['mode'] if 'mode' in config else 'TS'
        times = config['times'] if 'times' in config else 2

        self.n_output = n_output
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1D convolution on protein sequence
        # self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)

        self.fc_xt_2 = nn.Linear(125 * 16, 16)
        # self.LSTM_xt_2 = nn.LSTM(embed_dim, 8, 1, batch_first=True, bidirectional=True)

        self.embedding_xt_1 = nn.Embedding(434, embed_dim)
        self.embedding_xt_2 = nn.Embedding(8188, embed_dim)
        self.embedding_xt_3 = nn.Embedding(94816, embed_dim)

        # self.LSTM_xt = nn.LSTM(embed_dim, 64, 1, batch_first=True, bidirectional=True)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)  # n_output = 1 for regression task

        self.sub = Encoder(num_features, hidden_dim, num_gc_layers, times, mode)
        self.fc_x = nn.Linear(128, 128)
        self.fc_xt = nn.Linear(1000 * 128, 128)

        # 4 94816
        # self.trans = make_model(8188, N=1, d_model=128, d_ff=512, h=8, dropout=0.1, MAX_LEN=1200)
        # self.trans_out = nn.Linear(128*1000, 128)
        word_dict2 = pickle.load(open(config['feat_root'] + '/word_dict2', 'rb'))
        word_dict3 = pickle.load(open(config['feat_root'] + '/word_dict3', 'rb'))
        word_dict4 = pickle.load(open(config['feat_root'] + '/word_dict4', 'rb'))
        self.protein_encoder_1 = TargetRepresentation(1, len(word_dict2), 128)  # 434 421
        self.protein_encoder_2 = TargetRepresentation(1, len(word_dict3), 128)  # 8188 8044
        self.protein_encoder_3 = TargetRepresentation(1, len(word_dict4), 128)  # 94816 74353
        # self.protein_encoder_1 = TargetRepresentation(1, 625, 128)  # 434 421
        # self.protein_encoder_2 = TargetRepresentation(1, 15625, 128)  # 8188 8044
        # self.protein_encoder_3 = TargetRepresentation(1, 390625, 128)  # 94816 74353

        self.linear = nn.Linear(128 * 3, 128)
        # self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, data):
        data = data[0]
        # x, drug_edges, batch = data.drug, data.drug_edges, data.batch
        x, drug_edges, batch = data.x, data.edge_index, data.batch
        # batch = data
        # target, mask = data.target, data.target_mask
        # n_word = data.n_word[0]

        x = self.sub(x, drug_edges, batch, percent=0)
        x = self.fc_x(x)

        target_2, target_3, target_4 = data.target_2, data.target_3, data.target_4

        # embedded_xt_1 = self.embedding_xt_1(target_2)  # 512 1200 128
        xt_1 = self.protein_encoder_1(target_2)

        # embedded_xt_2 = self.embedding_xt_2(target_3)
        xt_2 = self.protein_encoder_2(target_3)

        # embedded_xt_3 = self.embedding_xt_3(target_4)
        xt_3 = self.protein_encoder_3(target_4)

        xt = torch.cat((xt_1, xt_2, xt_3), dim=-1)  # 512 128*3
        xt = self.linear(xt)

        '''
        embedded_xt_1 = self.embedding_xt_1(target_2)  # 512 1200 128
        LSTM_1, _ = self.LSTM_xt(embedded_xt_1)  # 512 1200 128
        x1 = self.fc_xt(LSTM_1.contiguous().view(-1,128*1200))  # 512 128

        embedded_xt_2 = self.embedding_xt_2(target_3)
        LSTM_2, _ = self.LSTM_xt(embedded_xt_2)
        x2 = self.fc_xt(LSTM_2.contiguous().view(-1,128*1200))

        embedded_xt_3 = self.embedding_xt_3(target_4)
        LSTM_3, _ = self.LSTM_xt(embedded_xt_3)
        x3 = self.fc_xt(LSTM_3.contiguous().view(-1,128*1200))


        xt =torch.cat((x1,x2,x3),dim=-1)  # 512 128*3
        xt = self.linear(xt)
        '''

        # conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        # xt = conv_xt.view(-1, 32 * 121)
        # xt = self.fc1_xt(xt)

        # target_mask = torch.unsqueeze(mask,dim=-2)
        # word_vectors_trans = self.trans(target, target_mask)  # [batch, length, feature_len]
        # xt = self.trans_out(F.relu(word_vectors_trans))  # [batch, length, feature_conv_len]
        # embedded_xt = self.embedding_xt(target)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


class Conv1dReLU(nn.Module):
    '''
    kernel_size=3, stride=1, padding=1
    kernel_size=5, stride=1, padding=2
    kernel_size=7, stride=1, padding=3
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            # nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            #           padding=padding),
            nn.LSTM(in_channels, out_channels, 1, bidirectional=True, batch_first=True),
            nn.ReLU()
        )

    def forward(self, x):
        LSTM, _ = self.inc[0](x)

        # return self.inc(x)
        return self.inc[1](LSTM)


class StackCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.inc = nn.Sequential(OrderedDict([('conv_layer0',
                                               Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size,
                                                          stride=stride, padding=padding))]))

        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1),
                                Conv1dReLU(out_channels * 2, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=padding))

        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))  # 自适应池化 输出维度为1

    def forward(self, x):
        return self.inc(x).squeeze(-1)


class TargetRepresentation(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_num):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNN(block_idx + 1, 128, 64, 3)
                # nn.LSTM(128, 64, 1, batch_first=True, bidirectional=True)
                # 改成bilstm
            )

        # self.linear = nn.Linear(block_num * 96, 96)
        self.linear = nn.Linear(block_num * 1000, 128)

    def forward(self, x):
        # x = self.embed(x).permute(0, 2, 1)  # 512 128 1000
        x = self.embed(x)
        feats = [block(x) for block in self.block_list]  # 512 96   *3
        x = torch.cat(feats, -1)
        x = self.linear(x)

        return x


class Encoder(nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, times, mode):
        super(Encoder, self).__init__()

        self.mode = mode
        # self.nce = InfoNCE()
        self.num_gc_layers = num_gc_layers

        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()

        self.ini_embed = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.layer_conv = nn.Conv2d(num_gc_layers, 1, (1, 1))

        self.mlp_before_readout = FF(dim)
        self.num_subgraph = 2 ** times

        self.subgraph_conv = nn.Conv2d(self.num_subgraph, 1, (1, 1))

        self.partition_list = nn.ModuleList()
        self.times = times
        for i in range(self.num_subgraph - 1):
            self.partition_list.append(nn.Sequential(nn.Linear(dim, 2), nn.Softmax(dim=-1)))

        self.mha_trans = nn.Parameter(torch.Tensor(self.num_subgraph, dim, 2), requires_grad=True)
        # for param in self.partition.parameters():
        #     nn.init.xavier_uniform_(param)

        self.dim = dim

        for i in range(num_gc_layers):
            mlp = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(mlp)

            # conv = GCNConv(dim,dim)
            # conv = GATConv(dim,dim)

            bn = torch.nn.BatchNorm1d(dim)

            self.conv_list.append(conv)
            self.bn_list.append(bn)

    def forward(self, x, edge_index, batch, percent):
        node_embed_list = []
        prob_loss = 0
        x = self.ini_embed(x)

        # neg_hn0 = self.create_neg_n0(x, batch)
        # x = torch.cat([x, neg_hn0], dim=0)
        # edge_index_ = torch.cat([edge_index, edge_index + edge_index[0, -1] + 1], dim=-1)
        for i in range(self.num_gc_layers):
            # activation function: per layer or final layer?
            x = F.relu(self.conv_list[i](x, edge_index))
            x = self.bn_list[i](x)
            node_embed_list.append(x)

        layer_nodes_embed = torch.stack(node_embed_list, dim=1).unsqueeze(-1)

        pos_global_node = self.layer_conv(layer_nodes_embed).squeeze()
        # pos_global_node, neg_global_node = global_node_embed[0], global_node_embed[1]
        # neg_global_graph = global_add_pool(neg_global_node, batch)

        if percent:
            mask = self.sampling_subgraph(percent, batch)
            sampling_subgraph_list = []
            for i in range(self.num_sub_graph):
                sampling_subgraph_list.append(global_add_pool(pos_global_node[mask[i, :]], batch=batch[mask[i, :]]))
            sampling_subgraph_ = torch.stack(sampling_subgraph_list, dim=1).unsqueeze(1)
            sampling_subgraph = self.subgraph_conv(sampling_subgraph_).squeeze()
        elif self.mode == 'MH':
            sampling_subgraph, assignment_prob = self.multi_head_subgraph_generation(pos_global_node, batch)
            prob_loss = -torch.sum(torch.abs(assignment_prob[:, :, 0] - assignment_prob[:, :, 1]))
        elif self.mode == 'TS':
            sampling_subgraph_list, assignment_prob = self.generate_subgraphs(pos_global_node, batch, self.times)

            # prob_loss = -torch.sum(torch.abs(assignment_prob[:, :, 0] - assignment_prob[:, :, 1]))
            # sampling_subgraph_ = torch.stack(sampling_subgraph_list, dim=1).unsqueeze(-1)
            # sampling_subgraph = self.subgraph_conv(sampling_subgraph_).squeeze()

            sampling_subgraph = torch.stack(sampling_subgraph_list, dim=1)  # num_sub * batch   dim
            sub = sampling_subgraph.reshape((int(batch[-1]) + 1) * self.num_subgraph, 128)  # batch * 4 128

            # 相似度矩阵
            cos_sim = torch.cosine_similarity(sub.unsqueeze(1), sub.unsqueeze(0), dim=-1)

            # 将对角线设置为0
            cos = cos_sim - torch.diag_embed(torch.diag(cos_sim))
            # print(cos_sim.shape)

            # 替换n
            sub_new = []
            num_sub = self.num_subgraph
            for i in range(0, int(batch[-1] + 1) * num_sub, num_sub):  # 以子图数目为一个步长
                # 在子图中选出相似度最高的一半子图替换

                sub_tmp = sub[i:i + num_sub, :]  # 子图特征

                cos_value = np.array(cos[i:i + num_sub, :].cpu().detach().numpy())  #
                # 不应该删除，应该设为0
                cos_value[:, 0:num_sub] = 0  # 将同一子图内的设置为0

                key = [int(np.argmax(cos_value[j])) for j in range(0, num_sub)]
                value = [np.max(cos_value[j]) for j in range(0, num_sub)]

                # 排序
                # 遇见排序后重复的怎么办 按照排序的默认方式
                sorted_nums = sorted(enumerate(value), key=lambda x: x[1], reverse=True)
                idx = [i[0] for i in sorted_nums]
                nums = [i[1] for i in sorted_nums]

                # 排序取子图的一半
                for m in range(len(idx) // 2):
                    sub_tmp[idx[m]] = sub[key[idx[m]]]

                sub_new.append(sub_tmp)

            sub_graph = torch.cat(sub_new, dim=0).reshape(int(batch[-1] + 1), num_sub, self.dim).unsqueeze(-1)
            sub_graph = self.subgraph_conv(sub_graph).squeeze()
            # sampling_subgraph_ = torch.stack(sub_new, dim=1).unsqueeze(-1)
            # sampling_subgraph = self.subgraph_conv(sampling_subgraph_).squeeze()


        else:
            assert False, 'wrong parameter for subgraphs'

        pos_global_graph = global_add_pool(pos_global_node, batch)

        # info_nce = self.nce_out(pos_global_node, pos_global_graph, neg_global_graph, batch)

        # 子图替换

        return pos_global_graph

    # def nce_out(self, pos_node, pos_graph, neg_graph, batch):
    #     out = 0
    #     for i in range(batch[-1] + 1):
    #         all_ = 0
    #         for j in pos_node[batch == i]:
    #             all_ += self.nce(j.unsqueeze(0), pos_graph[0].unsqueeze(0), pos_graph)
    #
    #         out += all_ / (batch == i).sum()
    #     return out / (batch[-1] + 1)

    # 该函数用于从给定的batch中随机抽样生成subgraph。函数首先创建一个与batch相同形状的全零mask矩阵，然后使用给定的percent参数确定
    # 每个subgraph中每个节点的抽样比例。接下来，函数使用np.random.choice函数从每个batch中选择抽样节点，并将抽样节点的索引保存在mask矩阵中。
    # 最后，函数返回mask矩阵中值为1的元素，表示抽样节点
    def sampling_subgraph(self, percent, batch):
        mask = torch.stack([torch.zeros_like(batch) for i in range(self.num_subgraph)],
                           dim=0)  # num_sub_graph x num_total_nodes

        for _ in range(int(self.num_subgraph)):
            for k, p in enumerate(percent):
                last_node_num = 0
                for i in range(batch[-1] + 1):
                    node_num = batch[batch == i].size(0)
                    sample_node_num = int(np.ceil(node_num * p))
                    idx = np.random.choice(node_num, sample_node_num)
                    mask[_ + k, idx + last_node_num] = 1
                    last_node_num += node_num

        return mask == 1

    # 该函数用于生成subgraph。函数首先将节点嵌入向量列表和分区神经网络的输出概率列表赋值给相应的变量。
    # 然后，函数根据概率值生成一个掩码矩阵，将概率大于等于0.5的节点标记为1，否则标记为0。
    # 接下来，函数将掩码矩阵扩展为与节点嵌入向量相同形状，并生成另一个掩码矩阵，
    # 将概率小于0.5的节点标记为1，否则标记为0。最后，函数将节点嵌入向量与两个掩码矩阵相乘，
    # 得到两个子图的节点嵌入向量列表，并返回该列表和概率列表。
    def generate_subgraph(self, node_embeddings, partition_nn):
        node_emb_list = []
        assignment_prob = partition_nn(node_embeddings)
        mask = (assignment_prob >= 0.5)[:, 0:1].long()
        mask = mask.expand_as(node_embeddings)
        mask_ = 1 - mask
        node_emb_list = node_emb_list + [node_embeddings * mask, node_embeddings * mask_]
        return node_emb_list, assignment_prob

    def generate_subgraphs(self, node_embeddings, batch, times):
        subgraph_list = []
        assign_prob_list = []
        node_embed_list = []
        for i in range(times):
            if i == 0:
                node_emb_l, assign_prob = self.generate_subgraph(node_embeddings, self.partition_list[0])
                node_embed_list = node_emb_l
                assign_prob_list.append(assign_prob)
            else:
                temp_list = []
                for j in range(len(node_embed_list)):
                    subgraph_l, assign_prob = self.generate_subgraph(node_embed_list[j],
                                                                     self.partition_list[2 ** i + j - 1])
                    temp_list = temp_list + subgraph_l
                    assign_prob_list.append(assign_prob)
                node_embed_list = temp_list
        for i in range(len(node_embed_list)):
            subgraph_list.append(global_add_pool(node_embed_list[i], batch))
        assign_prob = torch.stack(assign_prob_list, dim=0)
        return subgraph_list, assign_prob

    def create_neg_n0(self, h_n0, batch):
        neg_hn0 = []
        for i in range(int(batch[-1] + 1)):
            mask = batch == i
            h = h_n0[mask, :]
            idx = np.random.choice(h.size(0), h.size(0))
            neg_hn0.append(h[idx])
        return torch.cat(neg_hn0, dim=0)


class FF(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1)
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1, stride=1, padding=0)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1, stride=1, padding=0)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)
