# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/5/7 13:40
@author: LiFan Chen
@Filename: model.py
@Software: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
import pickle

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)


        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self, query, key, value, mask=None):
        if len(query.shape) > len(key.shape):
            bsz = query.shape[0]
        else:
            bsz = key.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(query.device)
        Q, K = Q.cpu(), K.cpu()
        del Q, K

        if mask is not None:
            mask_expanded = mask[:, np.newaxis, np.newaxis, :]
            mask_expanded = mask_expanded.repeat(1, energy.shape[1], energy.shape[2], 1)  # 第二维
            energy = energy.masked_fill(mask_expanded == 0, -1e10)
            
        attention = self.do(F.softmax(energy, dim=-1))

        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)

        return x
    
    
class Encoder(nn.Module):
    """protein feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers,kernel_size , dropout):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        #self.pos_embedding = nn.Embedding(1000, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([0.5]))
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim,self.hid_dim)

    def forward(self, protein):
        #pos = torch.arange(0, protein.shape[1]).unsqueeze(0).repeat(protein.shape[0], 1).to(self.device)
        #protein = protein + self.pos_embedding(pos)
        #protein = [batch size, protein len,protein_dim]

        conv_input = self.fc(protein)

        # conv_input=[batch size,protein len,hid dim]
        #permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        #conv_input = [batch size, hid dim, protein len]
        for i, conv in enumerate(self.convs):
            #pass through convolutional layer
            conved = conv(self.dropout(conv_input))
            #conved = [batch size, 2*hid dim, protein len]

            #pass through GLU activation function
            conved = F.glu(conved, dim=1)
            #conved = [batch size, hid dim, protein len]

            #apply residual connection / high way
            conved = (conved + conv_input) * self.scale.to(protein.device)
            #conved = [batch size, hid dim, protein len]

            #set conv_input to conved for next loop iteration
            conv_input = conved

        conved = conved.permute(0,2,1)
        # conved = [batch size,protein len,hid dim]
        return conved


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = x.permute(0, 2, 1)

        # x = [batch size, hid dim, sent len]

        x = self.do(F.relu(self.fc_1(x)))

        # x = [batch size, pf dim, sent len]

        x = self.fc_2(x)

        # x = [batch size, hid dim, sent len]

        x = x.permute(0, 2, 1)

        # x = [batch size, sent len, hid dim]

        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout)
        self.ea = self_attention(hid_dim, n_heads, dropout)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):

        trg1 = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
        trg1 = self.ln(trg1 + self.do(self.ea(trg1, src, src, src_mask)))
        trg1 = self.ln(trg1 + self.do(self.pf(trg1)))
        src1 = self.ln(src + self.do(self.sa(src, src, src, src_mask)))
        src1 = self.ln(src1 + self.do(self.ea(src1, trg, trg, trg_mask)))
        src1 = self.ln(src1 + self.do(self.pf(src1)))
        trg,src= trg.cpu(),src.cpu()
        del trg,src, trg_mask, src_mask

        # m1 = torch.mean(trg1, 1)
        # trg1 = torch.unsqueeze(m1, 1)
        # m2 = torch.mean(src1, 1)
        # src1 = torch.unsqueeze(m2, 1)
        # m1 = torch.mean(trg1, 1)
        # trg1 = torch.mean(trg1, 1)
        # m2 = torch.mean(src1, 1)
        # src1 = torch.mean(src1, 1)

        return trg1, src1



# IMT

class Decoder(nn.Module):
    """ compound feature extraction."""
    def __init__(self, atom_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        # self.sa = self_attention(hid_dim, n_heads, dropout)
        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout)
             for _ in range(n_layers)])
        # self.ft = nn.Linear(atom_dim, hid_dim)
        # self.do = nn.Dropout(dropout)
        # self.fc_1 = nn.Linear(hid_dim, 256)
        # self.fc_2 = nn.Linear(256, 2)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        for layer in self.layers:
            trg, src = layer(trg, src, trg_mask, src_mask)

        expanded_mask = trg_mask.unsqueeze(-1).expand_as(trg)
        masked_trg = expanded_mask * trg
        valid_counts = expanded_mask.sum(dim=1)
        trg = masked_trg.sum(dim=1) / (valid_counts.type(torch.float)+1e-6)
        
        expanded_mask = src_mask.unsqueeze(-1).expand_as(src)
        masked_src = expanded_mask * src
        valid_counts = expanded_mask.sum(dim=1)
        src = masked_src.sum(dim=1) / (valid_counts.type(torch.float)+1e-6)

        return trg, src


# GAT
# 图注意神经网络（Graph Attention Network，简称GAT）是一种用于图数据的神经网络模型，
# 它可以对每个节点的邻居节点进行不同的权重分配，从而更好地表达图结构中的信息。以下是使用PyTorch实现GAT的示例代码

# 这里定义了两个类，GraphAttentionLayer和GAT，其中GraphAttentionLayer是用来实现GAT中的注意力机制的基本模块，
# 而GAT则是整个GAT网络的

class GraphAttentionLayer(nn.Module):
    """
    实现GAT中的注意力机制的基本模块。
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features          # 输入特征维度:atom_dim
        self.out_features = out_features        # 输出特征维度:hid_dim
        self.dropout = dropout                  # Dropout概率
        self.alpha = alpha                      # LeakyReLU的负斜率
        self.concat = concat                    # 指示是否将多头注意力拼接

        # 定义可训练参数：
        # 参数矩阵W，用于变换输入特征
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)      # 参数初始化，Xavier初始化方法

        # 参数矩阵a，用于计算注意力系数
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # LeakyReLU激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """执行一次图注意力机制操作"""
        """
        前向传播函数。
        :param h: 输入特征，形状为(N, in_features)，其中N为节点数
        :param adj: 邻接矩阵，形状为(N, N)
        :return: 输出特征，形状为(N, out_features)
        """

        # 计算线性变换后的节点特征
        # Wh = torch.mm(h, self.W)
        Wh = torch.matmul(h, self.W)

        # 计算注意力系数，得到形状为(N, N)的矩阵
        a_input = self._prepare_attentional_mechanism_input(Wh)   # 将节点的特征向量拼接成注意力机制输入
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))    # 计算注意力系数

        # 利用注意力系数对邻接矩阵进行加权，因为对节点的特征进行注意机制重构相当于对邻接矩阵进行重构，见唐宇迪课程视频
        zero_vec = -9e15 * torch.ones_like(e)             # 对邻接矩阵中不存在的边的注意力系数设置一个极小值（负无穷）
        attention = torch.where(adj > 0, e, zero_vec)     # 将不存在的边的注意力系数赋为负无穷，将adj>0的地方设置为e(注意力系数)，否则设置为zero_vec
        attention = F.softmax(attention, dim=1)           # 计算注意力系数的softmax
        attention = F.dropout(attention, self.dropout, training=self.training)   # 对注意力系数进行Dropout
        h_prime = torch.matmul(attention, Wh)             # 根据注意力系数进行加权平均得到新的节点特征表示

        if self.concat:
            return F.elu(h_prime)       # 将多头注意力的输出拼接，并使用ELU激活函数
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        """
        将节点的特征向量拼接成注意力机制输入
        """
        N = Wh.size()[0]       # 节点个数
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)    # 在行方向重复N次
        Wh_repeated_alternating = Wh.repeat(N, 1)                 # 在列方向重复N次
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)    # 拼接两个重复矩阵
        return all_combinations_matrix.view(N, N, 2 * self.out_features)


class GAT(nn.Module):
    """定义一个包含多层图注意力机制的GAT模型"""
    def __init__(self, atom_dim, hid_dim, gat_heads, dropout, alpha, n_layers):
        super(GAT, self).__init__()

        self.W_gnn = nn.ModuleList([nn.Linear(atom_dim, atom_dim) for _ in range(n_layers)])
        self.compound_attn = nn.ParameterList(
            [nn.Parameter(torch.randn(size=(2 * atom_dim, 1))) for _ in range(n_layers)])


        self.n_layers = n_layers
        self.dropout = dropout
        self.atom_dim = atom_dim
        self.attentions = [GraphAttentionLayer(atom_dim, hid_dim, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(gat_heads)]     # 多头注意力机制
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(hid_dim * gat_heads, atom_dim, dropout=dropout, alpha=alpha, concat=False)  # 输出层的注意力头


    def forward(self, x, adj):

        for i in range(self.n_layers):
            h = torch.relu(self.W_gnn[i](x))
            size = h.size()[0]    # batch:1
            N = h.size()[1]       # 原子的个数,eg:第二个分子为78个原子，则N为78

            h1 = h.repeat(1, 1, N)          # h:(1,78,34)   h1:(1,78,34*78),表示在第三维度上重复N次，N为78，则34*78=2652
            h2 = h1.view(size, N * N, -1)   # h2:(1,78*78,34)  78*78=6084    (78*34*78)/(78*78)=34
            h3 = h.repeat(1, N, 1)          # h3:(1,78*78,34) ,表示在第二维度上重复N次，N为78，则78*78=6084
            h4 = torch.cat([h2, h3], dim=2)   # h4:(1,78*78,2*34)
            a_input = h4.view(size, N, -1, 2 * self.atom_dim)    # a_input:(1,78,78,68)

            # a_input = torch.cat([h.repeat(1, 1, N).view(size, N * N, -1), h.repeat(1, N, 1)], dim=2).view(size, N, -1, 2 * self.atom_dim)
            e = F.leaky_relu(torch.matmul(a_input, self.compound_attn[i]).squeeze(3))
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)  # 保证softmax 不为 0
            attention = F.softmax(attention, dim=2)
            attention = F.dropout(attention, self.dropout)
            h_prime = torch.matmul(attention, h)
            x = x+h_prime   # (1,78,34)

        return x

# BERT

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding (look-up table)    # vocab_size=8393  d_model=32
        self.pos_embed = nn.Embedding(max_len, d_model)  # position embedding    # max_len = 2048  (8112)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)  # x: [batch_size, seq_len]
        pos = torch.arange(seq_len, device=x.device, dtype=torch.long)  # [seq_len]
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]

        # print(x.max())  # （已转变为张量后）  # 8112(human)  7727(C.elegans) 8400
        # print(x.min())                     # 10(humna)    0(C.elegans)
        # print(self.pos_embed.num_embeddings)  # 2048  (8112)    8112human/C.elegans

        embedding = self.pos_embed(pos)    # 输入到Embedding模块的参数x(即为pos)是一个形状为[batch_size, seq_len]的整数张量，表示一批输入序列的token id。
        embedding = embedding + self.tok_embed(x)
        embedding = self.norm(embedding)
        return embedding

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.attention_map = None

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attention_map = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                                        enc_self_attn_mask)  # enc_inputs to same Q,K,V
        self.attention_map = attention_map
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs


def get_attn_pad_mask(seq):
    batch_size, seq_len = seq.size()
    pad_attn_mask = seq.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    pad_attn_mask_expand = pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]
    return pad_attn_mask_expand

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_head, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(d_k))
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_head, seq_len, seq_len]
        context = torch.matmul(attn, V)  # [batch_size, n_head, seq_len, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_head)
        self.W_K = nn.Linear(d_model, d_k * n_head)
        self.W_V = nn.Linear(d_model, d_v * n_head)

        self.linear = nn.Linear(n_head * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # q_s: [batch_size, n_head, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # k_s: [batch_size, n_head, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_head, d_v).transpose(1, 2)  # v_s: [batch_size, n_head, seq_len, d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_head, 1, 1)
        context, attention_map = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_head * d_v)  # context: [batch_size, seq_len, n_head * d_v]
        output = self.linear(context)
        output = self.norm(output + residual)
        return output, attention_map


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(self.relu(self.fc1(x)))

class BERT(nn.Module):
    def __init__(self, n_word):
        super(BERT, self).__init__()

        global max_len, n_layers, n_head, d_model, d_ff, d_k, d_v, hid_dim,vocab_size
        max_len = 8112     # human 8112  C.elegan 8377
        n_layers = 3
        n_head = 8
        d_model = 32  # 然后再使用FC转成64=hid_dim
        d_ff = 64
        d_k = 32
        d_v = 32
        hid_dim = 64
        vocab_size = n_word

        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, hid_dim)

    def forward(self, input_ids):
        # input_ids[batch_size, seq_len] like[8,1975]
        output = self.embedding(input_ids)  # [batch_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids)  # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)
        output = self.fc(output)

        return output






class InteractionModel(nn.Module):
    def __init__(self, hid_dim, n_heads):
        super(InteractionModel, self).__init__()
        # self.compound_embedding = nn.Linear(compound_feature_size, hidden_size)
        # self.protein_embedding = nn.Linear(protein_feature_size, hidden_size)
        self.compound_attention = nn.MultiheadAttention(hid_dim, n_heads)
        self.protein_attention = nn.MultiheadAttention(hid_dim, n_heads)
        self.compound_fc = nn.Linear(hid_dim, hid_dim)
        self.protein_fc = nn.Linear(hid_dim, hid_dim)
        self.activation = nn.ReLU()

        self.hid_dim = hid_dim

    def forward(self, compound_features, protein_features):
        compound_embedded = self.activation(compound_features)
        protein_embedded = self.activation(protein_features)

        compound_embedded = compound_embedded.permute(1, 0, 2)
        protein_embedded = protein_embedded.permute(1, 0, 2)

        '''
        这两行代码的作用是交换张量的维度。
        在这个模型中，输入的化合物特征和蛋白质特征分别表示为(batch_size, sequence_length, feature_size)的张量。
        然而，在进行自注意力计算时，这些张量需要在序列长度和批次之间进行交换，即变成(sequence_length, batch_size, feature_size)。
        这样，才能使得每个元素在序列长度的维度上进行自注意力计算。
        因此，这两行代码是将(batch_size, sequence_length, feature_size)的张量转换为(sequence_length, batch_size, feature_size)的张量。
        
        '''

        compound_attention_output, _ = self.compound_attention(compound_embedded, compound_embedded,
                                                               compound_embedded)
        protein_attention_output, _ = self.protein_attention(protein_embedded, protein_embedded, protein_embedded)

        compound_attention_output = compound_attention_output.permute(1, 0, 2)
        protein_attention_output = protein_attention_output.permute(1, 0, 2)

        compound_output = self.activation(self.compound_fc(compound_attention_output))
        protein_output = self.activation(self.protein_fc(protein_attention_output))

        com_att = torch.unsqueeze(torch.mean(compound_output,1),1)
        pro_att = torch.unsqueeze(torch.mean(protein_output,1),1)
        return com_att,pro_att


        # 对mean使用权重参数矩阵
        # compound_mean = torch.mean(compound_output,1)
        # protein_mean = torch.mean(protein_output,1)
        #
        # com_att = torch.unsqueeze(torch.tanh(torch.mm(compound_mean,self.weight_matrix)),1)
        # pro_att = torch.unsqueeze(torch.tanh(torch.mm(protein_mean,self.weight_matrix)),1)
        # return com_att,pro_att



# NTN  (tensor_network)

class TensorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """

    def __init__(self,k_feature,hid_dim,k_dim):
        super(TensorNetworkModule, self).__init__()
        self.k_feature = k_feature
        self.hid_dim = hid_dim
        self.k_dim = k_dim

        self.setup_weights()
        self.init_parameters()

        self.fc1 = nn.Linear(hid_dim,k_dim)
        self.fc2 = nn.Linear(k_dim, hid_dim)


    def setup_weights(self):
        """
        Defining weights.  k_feature = args.filters_3   args.tensor_neurons = k_dim
        """
        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(
                self.k_feature, self.k_feature, self.k_dim
            )
        )                                                             # (16,16,16)
        self.weight_matrix_block = torch.nn.Parameter(
            torch.Tensor(self.k_dim, 2 * self.k_feature)
        )                                                             # (16,32)
        self.bias = torch.nn.Parameter(torch.Tensor(self.k_dim, 1))   # (16,1)

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.   com_att
        :param embedding_2: Result of the 2nd embedding after attention.   pro_att
        :return scores: A similarity score vector.
        """
        embedding_1 = torch.squeeze(embedding_1, dim=1)   # (1,1,64)——>(1,64)   (batch_size,1,64)——>(batch_size,64)
        embedding_1 = self.fc1(embedding_1)  # (1,64)——>(1,16)
        embedding_2 = torch.squeeze(embedding_2, dim=1)
        embedding_2 = self.fc1(embedding_2)

        batch_size = len(embedding_1)
        # print(self.weight_matrix.view(self.k_feature, -1).shape) # 原始输入的两个实体都是16维向量，现在用256维表示他们的某种关系
        scoring = torch.matmul(
            embedding_1, self.weight_matrix.view(self.k_feature, -1)
        )
        # print(scoring.shape)
        scoring = scoring.view(batch_size, self.k_feature, -1).permute([0, 2, 1]) #filters_3可以理解成找多少种关系
        # print(scoring.shape)
        scoring = torch.matmul(
            scoring, embedding_2.view(batch_size, self.k_feature, 1)
        ).view(batch_size, -1)
        # print(scoring.shape)
        combined_representation = torch.cat((embedding_1, embedding_2), 1)
        # print(combined_representation.shape)
        block_scoring = torch.t(
            torch.mm(self.weight_matrix_block, torch.t(combined_representation))
        )                                                                               # torch.t:转置
        # print(block_scoring.shape)
        scores = F.relu(scoring + block_scoring + self.bias.view(-1))
        # print(scores.shape)    # (1,16)——(batch_size,16)
        # scores = torch.unsqueeze(scores, 1)  # (1,16) ——> (1,1,16)
        scores = self.fc2(scores)    # (1,1,16) ——> (1,1,64)
        return scores




class AMMVF(nn.Module):
    def __init__(self, n_output, protein_dim=100, atom_dim=34, **config):
        super().__init__()

        
        n_layers = config['n_layers']
        hid_dim = config['hid_dim']
        finger_dict = pickle.load(open(config['feat_root'] + '/atom_dict', 'rb'))
        prot_embedding_matrix = pd.read_csv(config['feat_root'] + f'/embedding_3_100d.csv')
        prot_embedding_weight = prot_embedding_matrix.values
        self.prot_embed = nn.Embedding.from_pretrained(torch.FloatTensor(prot_embedding_weight), freeze=True)
        
        self.embed_fingerprint = nn.Embedding(len(finger_dict), atom_dim)
        self.gat = GAT(atom_dim, hid_dim, config['gat_heads'], config['dropout'], config['alpha'], config['n_layers'])
        self.Bert = BERT(len(prot_embedding_matrix))
        self.inter_att = InteractionModel(hid_dim, config['n_heads'])
        self.tensor_network = TensorNetworkModule(config['k_feature'], hid_dim, config['k_dim'])

        self.decoder = Decoder(atom_dim, hid_dim, n_layers, config['n_heads'], config['pf_dim'], DecoderLayer, SelfAttention, PositionwiseFeedforward, config['dropout'])
        self.weight = nn.Parameter(torch.FloatTensor(34, 34))
        self.init_weight()


        self.protein_dim = protein_dim
        self.hid_dim = hid_dim
        self.atom_dim = atom_dim
        self.fc1 = nn.Linear(self.protein_dim, self.hid_dim)
        self.fc2 = nn.Linear(self.atom_dim, self.hid_dim)

        self.W_attention = nn.Linear(self.hid_dim, self.hid_dim)

        self.out = nn.Sequential(
            nn.Linear(self.hid_dim * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_output)
        )

    def init_weight(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, data):
        data = data[0]
        compound1, compound2, adj, protein = data.x_atom, data.x, data.adj, data.prot
        compound1_mask, compound2_mask, prot_mask = data.x_atom_mask, data.x_mask, data.prot_mask

        protein1 = self.prot_embed(protein)
        protein1 = self.fc1(protein1)                  # (1,804,64)     # protein1 =[ batch size=1,protein len, hid_dim]

        compound1 = self.fc2(compound1.float())                # (1,1,64)       # compound1 = [batch size=1 ,atom_num, hid_dim]

        protein1_c, compound1_p = self.decoder(protein1, compound1, prot_mask, compound1_mask)     # protein1_c:(1,1,64)    # compound1_p:(1,1,64) 交互信息，decoder为MINN中的Interformer模块,都使用了mean
        # protein1_c, compound1_p = self.decoder(protein1, compound1)
        compound_vectors = self.embed_fingerprint(compound2)            # compound_vectors:(78,34)

        compound2 = self.gat(compound_vectors, adj)  # 1,78,34 (取消了mean)
        compound2 = self.fc2(compound2)                                 # compound_vectors:(1,1,64)，其中第二维度的1是当前原子数量，只是此时数量刚好为1

        protein2 = self.Bert(protein)                     # protein2:(1,806,64)  # [batch size, protein len, hid dim]

        com_att,pro_att = self.inter_att(compound2, protein2)

        scores = self.tensor_network(com_att, pro_att)

        out_fc = torch.cat((scores, compound1_p, protein1_c), 1)
        out = self.out(out_fc)    # out = [batch size, 2]

        return out
