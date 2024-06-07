import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd

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
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(Q.device)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = self.do(F.softmax(energy, dim=-1))
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        return x

class Encoder(nn.Module):
    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size , dropout):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"
        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.scale = torch.sqrt(torch.FloatTensor([0.5]))
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers
        self.do = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.gn = nn.GroupNorm(8, hid_dim * 2)
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self, protein):
        conv_input = self.fc(protein)
        conv_input = conv_input.permute(0, 2, 1)
        for i, conv in enumerate(self.convs):
            conved = conv(self.do(conv_input))
            conved = F.glu(conved, dim=1)
            conved = (conved + conv_input) * self.scale.to(protein.device)
            conv_input = conved

        conved = conved.permute(0, 2, 1)
        conved = self.ln(conved)
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
        x = x.permute(0, 2, 1)
        x = self.do(F.relu(self.fc_1(x)))
        x = self.fc_2(x)
        x = x.permute(0, 2, 1)
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
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))
        trg = self.ln(trg + self.do(self.pf(trg)))
        return trg

class Decoder(nn.Module):
    def __init__(self, embed_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout):
        super(Decoder, self).__init__()
        self.ft = nn.Linear(embed_dim, hid_dim)
        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout)
             for _ in range(n_layers)])
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg = self.do(self.ft(trg))
        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)
        return trg #[bs, seq_len, hid_dim]

class TextCNN(nn.Module):
    def __init__(self, embed_dim, hid_dim, kernels=[3, 5, 7], dropout_rate=0.5):
        super(TextCNN, self).__init__()
        padding1 = (kernels[0] - 1) // 2
        padding2 = (kernels[1] - 1) // 2
        padding3 = (kernels[2] - 1) // 2

        self.conv1 = nn.Sequential(
            nn.Conv1d(embed_dim, hid_dim, kernel_size=kernels[0], padding=padding1),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=kernels[0], padding=padding1),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(embed_dim, hid_dim, kernel_size=kernels[1], padding=padding2),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=kernels[1], padding=padding2),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(embed_dim, hid_dim, kernel_size=kernels[2], padding=padding3),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=kernels[2], padding=padding3),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
        )

        self.conv = nn.Sequential(
            nn.Linear(hid_dim*len(kernels), hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim),
        )
    def forward(self, protein):
        protein = protein.permute([0, 2, 1])  #[bs, hid_dim, seq_len]
        features1 = self.conv1(protein)
        features2 = self.conv2(protein)
        features3 = self.conv3(protein)
        features = torch.cat((features1, features2, features3), 1)  #[bs, hid_dim*3, seq_len]
        features = features.max(dim=-1)[0]  #[bs, hid_dim*3]
        return self.conv(features)

class IIFDTI(nn.Module):
    def __init__(self, n_output, hid_dim, n_layers, kernel_size, n_heads, pf_dim, dropout, atom_dim=34, protein_dim=100, **config):
        super(IIFDTI, self).__init__()

        self.do = nn.Dropout(dropout)
        prot_embedding_matrix = pd.read_csv(config['feat_root'] + f'/embedding_3_100d.csv')
        prot_embedding_weight = prot_embedding_matrix.values
        self.prot_embed = nn.Embedding.from_pretrained(torch.FloatTensor(prot_embedding_weight), freeze=True)

        self.prot_embed2 = nn.Embedding(26, protein_dim)
        smi_embedding_matrix = pd.read_csv(config['feat_root'] + f'/smile_embedding_1_100d.csv')
        smi_embedding_weight = smi_embedding_matrix.values
        self.smi_embed = nn.Embedding.from_pretrained(torch.FloatTensor(smi_embedding_weight), freeze=True)

        # protein encoding, target decoding
        self.enc_prot = Encoder(len(prot_embedding_weight[0]), hid_dim, n_layers, kernel_size, dropout)
        self.dec_smi = Decoder(len(smi_embedding_weight[0]), hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout)

        # target encoding, protein decoding
        self.enc_smi = Encoder(len(smi_embedding_weight[0]), hid_dim, n_layers, kernel_size, dropout)
        self.dec_prot = Decoder(len(prot_embedding_weight[0]), hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout)

        self.prot_textcnn = TextCNN(100, hid_dim)
        self.W_gnn = nn.ModuleList([nn.Linear(atom_dim, atom_dim),
                                    nn.Linear(atom_dim, atom_dim),
                                    nn.Linear(atom_dim, atom_dim)])
        self.W_gnn_trans = nn.Linear(atom_dim, hid_dim)
        self.out = nn.Sequential(
            nn.Linear(hid_dim * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_output)
        )
        # GAT
        self.dropout = dropout
        self.atom_dim = atom_dim
        self.compound_attn = nn.ParameterList(
            [nn.Parameter(torch.randn(size=(2 * atom_dim, 1))) for _ in range(len(self.W_gnn))])

    def gnn(self, xs, A):
        for i in range(len(self.W_gnn)):
            h = torch.relu(self.W_gnn[i](xs))
            size = h.size()[0]
            N = h.size()[1]
            # a_input 这个代码就比较nb了，  构造了 一个 [N*N, 2*F]  -->  [N, N, 2*F]  self.a --> [2*F, 1]  e--> [N,N]
            a_input = torch.cat([h.repeat(1, 1, N).view(size, N * N, -1), h.repeat(1, N, 1)], dim=2).view(size, N, -1,
                                                                                                          2 * self.atom_dim)
            e = F.leaky_relu(torch.matmul(a_input, self.compound_attn[i]).squeeze(3))
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(A > 0, e, zero_vec)  # 保证softmax 不为 0
            attention = F.softmax(attention, dim=2)
            attention = F.dropout(attention, self.dropout, training=self.training)
            h_prime = torch.matmul(attention, h)
            xs = xs + h_prime
        xs = self.do(F.relu(self.W_gnn_trans(xs)))
        return xs

    # compound = [bs,atom_num, atom_dim]
    # adj = [bs, atom_num, atom_num]
    # protein = [bs, protein_len, 100]
    # smi_ids = [bs, smi_len]
    # prot_ids = [bs, prot_len]
    def forward(self, data):
        data = data[0]
        compound, adj, protein, smi_ids, prot_ids = data.x, data.adj, data.prot_ids, data.smile, data.prot_ngram 
        cmp_gnn_out = self.gnn(compound.float(), adj)   # [bs, new_len, hid_dim]
        pro_textcnn_out = self.prot_textcnn(self.prot_embed2(protein)) # [bs, prot_len, hid_dim]

        smi_mask = torch.where(smi_ids == 0, torch.tensor(0, device=compound.device), torch.tensor(1, device=compound.device))
        prot_mask = torch.where(prot_ids == 0, torch.tensor(0, device=compound.device), torch.tensor(1, device=compound.device))

        out_enc_prot = self.enc_prot(self.prot_embed(prot_ids)) #[bs, prot_len, hid_dim]
        out_dec_smi = self.dec_smi(self.smi_embed(smi_ids), out_enc_prot, smi_mask.unsqueeze(1).unsqueeze(3), prot_mask.unsqueeze(1).unsqueeze(2))  # [bs, smi_len, hid_dim]

        out_enc_smi = self.enc_smi(self.smi_embed(smi_ids))  # [bs, smi_len, hid_dim]
        out_dec_prot = self.dec_prot(self.prot_embed(prot_ids), out_enc_smi, prot_mask.unsqueeze(1).unsqueeze(3), smi_mask.unsqueeze(1).unsqueeze(2)) # # [bs, prot_len, hid_dim]

        is_max = False
        if is_max:
            cmp_gnn_out = cmp_gnn_out.max(dim=1)[0]
            if pro_textcnn_out.ndim>=3: pro_textcnn_out = pro_textcnn_out.max(dim=1)[0]
            out_dec_smi = out_dec_smi.max(dim=1)[0]
            out_dec_prot = out_dec_prot.max(dim=1)[0]
        else:
            cmp_gnn_out = cmp_gnn_out.mean(dim=1)
            if pro_textcnn_out.ndim>=3: pro_textcnn_out = pro_textcnn_out.mean(dim=1)
            out_dec_smi = out_dec_smi.mean(dim=1)
            out_dec_prot = out_dec_prot.mean(dim=1)
        out_fc = torch.cat([cmp_gnn_out, pro_textcnn_out, out_dec_smi, out_dec_prot], dim=-1)
        return self.out(out_fc)