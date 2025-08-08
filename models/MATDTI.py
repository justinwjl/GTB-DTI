import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.modules.batchnorm import _BatchNorm
import torch_geometric.nn as gnn
from torch import Tensor
from torch_geometric.utils import to_dense_batch
from test_model.MGIN import GraphDenseGinNet
from test_model.MAT import GraphTransformer
import math
from models.MRBDTA import Encoder as MRB_Encoder
from models.MGraphDTA import GraphDenseNet

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
        # query = key = value [batch size, sent len, hid dim]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Q, K, V = [batch size, sent len, hid dim]
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(Q.device)

        # energy = [batch size, n heads, sent len_Q, sent len_K]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))
        # attention = [batch size, n heads, sent len_Q, sent len_K]
        x = torch.matmul(attention, V)
        # x = [batch size, n heads, sent len_Q, hid dim // n heads]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, sent len_Q, n heads, hid dim // n heads]
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        # x = [batch size, src sent len_Q, hid dim]
        x = self.fc(x)
        # x = [batch size, sent len_Q, hid dim]
        return x

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        self.query1 = nn.Linear(hidden_dim, hidden_dim)
        self.query2 = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim * 2, hidden_dim)

    def apply_mask(self, energy, mask):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # Reshape mask
            energy = energy.masked_fill(mask == 0, float("-1e20"))  # Apply mask
        return energy

    def forward(self, drug_feat_graph, drug_feat_smiles, target_feat, mask_graph=None, mask_smiles=None, mask_target=None):
        batch_size = drug_feat_graph.shape[0]

        # Prepare the query, key, value
        query_graph = self.query1(drug_feat_graph).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        query_smiles = self.query2(drug_feat_smiles).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(target_feat).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(target_feat).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention mechanism for both drug representations against the target
        energy_graph = torch.matmul(query_graph, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        energy_smiles = torch.matmul(query_smiles, key.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply masks
        energy_graph = self.apply_mask(energy_graph, mask_graph)
        energy_smiles = self.apply_mask(energy_smiles, mask_smiles)
        energy_target = self.apply_mask(energy_graph, mask_target)  # This mask is applied to the target feature

        attention_graph = torch.softmax(energy_graph, dim=-1)
        attention_smiles = torch.softmax(energy_smiles, dim=-1)

        out_graph = torch.matmul(attention_graph, value).transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        out_smiles = torch.matmul(attention_smiles, value).transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)

        # Concatenate and pass through final linear layer
        out = torch.cat((out_graph, out_smiles), dim=-1)
        out = self.fc_out(out)

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
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )
    
    def forward(self, x):

        return self.inc(x)

class LinearReLU(nn.Module):
    def __init__(self,in_features, out_features, bias=True):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
            nn.ReLU()
        )

    def forward(self, x):
        
        return self.inc(x)

class StackCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.inc = nn.Sequential(OrderedDict([('conv_layer0', Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))]))
        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1), Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))

        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):

        return self.inc(x).squeeze(-1)
    
class TargetRepresentation(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_num, CNN_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNN(block_idx+1, embedding_num, CNN_size, 3)
            )

        self.linear = nn.Linear(block_num * CNN_size, CNN_size)
        
    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        feats = [block(x) for block in self.block_list]
        x = torch.cat(feats, -1)
        x = self.linear(x)

        return x
    

class ScaleNorm(nn.Module):
    """ScaleNorm"""
    "All g’s in SCALE NORM are initialized to sqrt(d)"
    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(math.sqrt(scale)))
        self.eps = eps
        
    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
    
    
    
class MATDTI7(nn.Module):
    def __init__(self, d_atom, char_dim = 64, conv = 40, drug_kernel = [4, 6, 8], protein_kernel = [4, 8, 12], n_output = 1, protein_MAX_LENGH = 1000, drug_MAX_LENGH = 100, dropout = 0.0, d_model=256, n_generator_layers=3, scale_norm=False, aggregation_type='mean', **conifg):
        super(MATDTI7, self).__init__()

        self.dim = char_dim
        self.conv = conv
        self.drug_MAX_LENGH = drug_MAX_LENGH
        self.drug_kernel = drug_kernel
        self.protein_MAX_LENGH = protein_MAX_LENGH
        self.protein_kernel = protein_kernel

        self.protein_embed = nn.Embedding(26, self.dim,padding_idx=0)
        self.drug_embed = nn.Embedding(65, self.dim,padding_idx=0)
        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels= int(d_model/4),  kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=int(d_model/4), out_channels= int(d_model/2),  kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=int(d_model/2), out_channels= d_model,  kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(self.drug_MAX_LENGH-self.drug_kernel[0]-self.drug_kernel[1]-self.drug_kernel[2]+3)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4, kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )
        self.Protein_max_pool = nn.MaxPool1d(self.protein_MAX_LENGH - self.protein_kernel[0] - self.protein_kernel[1] - self.protein_kernel[2] + 3)


        if n_generator_layers == 1:
            self.proj = nn.Linear(2*d_model, n_output)
        else:
            self.proj = []
            for i in range(n_generator_layers-1):
                self.proj.append(nn.Linear(2*d_model, 2*d_model))
                self.proj.append(nn.LeakyReLU(0.1))
                self.proj.append(ScaleNorm(2*d_model) if scale_norm else LayerNorm(2*d_model))
                self.proj.append(nn.Dropout(dropout))
            self.proj.append(nn.Linear(2*d_model, n_output))
            self.proj = torch.nn.Sequential(*self.proj)
        self.aggregation_type = aggregation_type
        self.ligand_encoder = GraphTransformer(d_atom=d_atom, d_model=d_model, N=8, h=16, N_dense=1, lambda_attention=0.33, lambda_distance=0.33, dense_output_nonlinearity='relu',distance_matrix_kernel='exp', dropout=0.0, aggregation_type=self.aggregation_type)
        self.protein_encoder = TargetRepresentation(block_num=3, vocab_size=26, embedding_num=128, CNN_size=d_model)
        self.CNN_proj = nn.Linear(d_model, d_model)
        self.GNN_proj = nn.Linear(d_model, d_model)
        self.attn_proj = nn.Linear(d_model, d_model)
        
    def smile_cnn(self, drug):
        drugembed = self.drug_embed(drug)
        drugembed = drugembed.permute(0, 2, 1)
        drugConv = self.Drug_CNNs(drugembed)

        return drugConv
    
    def prot_cnn(self, prot):
        proteinembed = self.protein_embed(prot)
        proteinembed = proteinembed.permute(0, 2, 1)
        proteinConv = self.Protein_CNNs(proteinembed)
        return proteinConv
    
    
    def forward(self, data):
        data = data[0]
        drug_feature, adj, dist, protein, drug_seq = data.x, data.adj, data.dist, data.target, data.smile
        mask = torch.sum(torch.abs(drug_feature), dim=-1) != 0
        drugGraph = self.ligand_encoder(drug_feature, mask, adj, dist, None)
        mask = mask.unsqueeze(-1).float()
        out_masked = drugGraph * mask
        if self.aggregation_type == 'mean':
            out_sum = out_masked.sum(dim=1)
            out_avg_pooling = out_sum / mask.sum(dim=(1))
        elif self.aggregation_type == 'sum':
            out_sum = out_masked.sum(dim=1)
            out_avg_pooling = out_sum
        elif self.aggregation_type == 'dummy_node':
            out_avg_pooling = out_masked[:,0]
        
        proteinConv = self.protein_encoder(protein)
        drugCNN = self.smile_cnn(drug_seq)
        drugCNN = self.Drug_max_pool(drugCNN)
        drugCNN = drugCNN.squeeze(2)
        attn = F.sigmoid(self.attn_proj(self.GNN_proj(out_avg_pooling) + self.CNN_proj(drugCNN)))
        drug_feature = out_avg_pooling*attn+drugCNN*(1-attn)
        x = torch.cat([drug_feature, proteinConv], dim=-1)
        out = self.proj(x)
        
        return out 
    
    