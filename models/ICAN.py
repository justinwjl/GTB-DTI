import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import copy


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1) # No one line is allowed.
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device) #to device
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        words_embeddings = self.word_embeddings(input_ids)    
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

def clones(module, N):
    return [copy.deepcopy(module) for _ in range(N)]

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) 
        scores = scores + attn_mask
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model_1, d_model_2, n_heads, d_dim):
        super(MultiHeadAttention, self).__init__()
        self.d_dim  = d_dim
        
        self.n_heads = n_heads
        self.d_model_1 = d_model_1
        self.d_model_2 = d_model_2
        
        self.W_Q_dense = nn.Linear(self.d_model_1, self.d_dim * self.n_heads, bias=False) 
        self.W_K_dense = nn.Linear(self.d_model_2, self.d_dim * self.n_heads, bias=False)
        self.W_V_dense = nn.Linear(self.d_model_2, self.d_dim * self.n_heads, bias=False)
        
        self.scale_product = ScaledDotProductAttention(self.d_dim)
        self.out_dense = nn.Linear(self.n_heads * self.d_dim, self.d_model_1, bias=False)  # self.n_heads * self.d_dim = const
        
    def forward(self, Q, K, V, attn_mask):
        Q_spare, batch_size = Q, Q.size(0)
       
        q_s = self.W_Q_dense(Q).view(batch_size, -1, self.n_heads, self.d_dim).transpose(1,2)
        k_s = self.W_K_dense(K).view(batch_size, -1, self.n_heads, self.d_dim).transpose(1,2)
        self.v_s = self.W_V_dense(V).view(batch_size, -1, self.n_heads, self.d_dim).transpose(1,2)

        context, self.attn = self.scale_product(q_s, k_s, self.v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_dim)
        context = self.out_dense(context)
        
        return context + Q_spare
    
class CNN(nn.Module):
    def __init__(self, features, time_size, n_output):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(features, 32, kernel_size=5, stride=1, padding = 2)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding = 2)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dense_1 = nn.Linear(32 * int(int(time_size/2)/2),128)
        self.dense_2 = nn.Linear(128, 32)
        self.dense_3 = nn.Linear(32, n_output)
        self.relu = nn.ReLU()
        # self.sigmoid_func = nn.Sigmoid()
        self.dropout = nn.Dropout(p = 0.4)

    def forward(self, emb_mat):
        output = torch.transpose(emb_mat, -1, -2)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.maxpool(output)
        output = self.dropout(output)
        
        output = self.conv2(output)
        output = self.relu(output)
        output = self.maxpool(output)
        output = self.dropout(output)
        
        output = output.view(-1, output.size(1) * output.size(2))
        
        #fully connected layer
        output = self.dense_1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.dense_2(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.dense_3(output)
        # output = self.sigmoid_func(output)
       
        return output


class ICAN_model(nn.Module):
    def __init__(self, feature_size = 384, n_heads = 4, d_dim = 32*4, feature = 128, pooling_dropout = 0.5, linear_dropout = 0.3, **config): # original d_dim = 32
        super(ICAN_model, self).__init__()

        self.max_d = config['max_drug_seq']
        self.max_p = config['max_protein_seq']
        self.emb_size = config['emb_size']
        self.dropout_rate = config['dropout_rate'] 
        self.input_dim_drug = config['input_dim_drug']
        self.input_dim_target = config['input_dim_target']

        n_output = config['n_output']

        self.demb = Embeddings(self.input_dim_drug, self.emb_size, self.max_d, self.dropout_rate)
        self.pemb = Embeddings(self.input_dim_target, self.emb_size, self.max_p, self.dropout_rate) 
   
        self.att_list_1 = MultiHeadAttention(feature_size, feature_size, n_heads, d_dim)   
        self.att_list_2 = MultiHeadAttention(feature_size, feature_size, n_heads, d_dim)  
        
        self.dense_1 = nn.Linear(feature_size * 2, 64)
        self.dense_2 = nn.Linear(64, 16)
        self.dense_3 = nn.Linear(16, 1)
       
        self.dropout_layer_pool = nn.Dropout(pooling_dropout)
        self.dropout_layer_linear = nn.Dropout(linear_dropout)
        self.sigmoid_func = nn.Sigmoid()
        self.relu_func = nn.ReLU()

        #self.CNN = CNN(feature_size, self.max_d+self.max_p)
        self.CNN = CNN(feature_size, self.max_p, n_output)
        #self.CNN = CNN(feature_size, self.max_d)
        #self.CNN = CNN(self.max_d, self.max_p) 
 
    def forward(self, data):
        data = data[0]
        input_1, input_2, attn_mask_1, attn_mask_2 = data.d_v, data.p_v, data.d_mask, data.p_mask
        attn_mask_1 = attn_mask_1.unsqueeze(1).unsqueeze(2)
        attn_mask_2 = attn_mask_2.unsqueeze(1).unsqueeze(2)
        attn_mask_1 = (1.0 - attn_mask_1) * -10000.0
        attn_mask_2 = (1.0 - attn_mask_2) * -10000.0

        self.h_out_1 = self.demb(input_1) # batch_size x seq_length x embed_size
        self.h_out_2 = self.pemb(input_2)

        # cross attention 
        out_1_q, out_2_k, out_2_v = self.h_out_1, self.h_out_2, self.h_out_2
        out_2_q, out_1_k, out_1_v = self.h_out_2, self.h_out_1, self.h_out_1
        self.out_1_temp = self.att_list_1(out_1_q, out_2_k, out_2_v, attn_mask_2)
        self.out_2_temp = self.att_list_2(out_2_q, out_1_k, out_1_v, attn_mask_1)

        """ 
        # self-attention
        out_1_q, out_1_k, out_1_v = self.h_out_1, self.h_out_1, self.h_out_1
        out_2_q, out_2_k, out_2_v = self.h_out_2, self.h_out_2, self.h_out_2
        self.out_1_temp = self.att_list_1(out_1_q, out_1_k, out_1_v, attn_mask_1)
        self.out_2_temp = self.att_list_2(out_2_q, out_2_k, out_2_v, attn_mask_2)
        """

        out_1 = self.dropout_layer_pool(self.out_1_temp)
        out_2 = self.dropout_layer_pool(self.out_2_temp)

        """
        # FCL
        #print('out_1 {}'.format(out_1.shape)) # [128, 50, 384]
        #print('out_2 {}'.format(out_2.shape))  # [128, 545, 384]
         # neural network sequential combination for cross attention
        self.out_1, _ = torch.max(out_1, dim = 1) #[128, 384]
        self.out_2, _ = torch.max(out_2, dim = 1) #[128, 384]
        self.out = torch.cat((self.out_1, self.out_2), dim = 1) 
        #print('self.out {}'.format(self.out.shape))  # [128, 768]
        out = self.dense_1(self.out)
        out = self.relu_func(out)
        out = self.dropout_layer_linear(out)
        out = self.dense_2(out)
        out = self.relu_func(out)
        out = self.dropout_layer_linear(out)
        out = self.dense_3(out)
        out = self.sigmoid_func(out)
        """
        
        """
        # Notice: reset of self.CNN = CNN(feature_size, self.max_d+self.max_p)
        # CNN sequential combination  for cross attention
        out = torch.cat((out_1, out_2), dim = 1) 
        #print('self.out {}'.format(self.out.shape))  # [128, 595, 384]
        out = self.CNN(out)
        """    
        
        # CNN for protein for cross attention
        out = self.CNN(out_2) #protein       
        #out = self.CNN(out_1) #drug

        return out