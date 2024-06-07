import torch
from torch_geometric.utils import to_dense_adj
import torch_sparse
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from torchdrug.layers import MLP

class DeepGLSTM(nn.Module):
    def __init__(self, k1=1, k2=2, k3=3, embed_dim=128, lstm_layer=1, num_feature_xd=78, n_output=1, num_feature_xt=25, output_dim=128, dropout=0.2, **config):
        super(DeepGLSTM, self).__init__()
        self.model_name = 'DeepGLSTM'
        # Smile graph branch
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.embed_dim = embed_dim
        self.lstm_layer = lstm_layer
        self.Conv1 = GCNConv(num_feature_xd, num_feature_xd)
        self.Conv2 = GCNConv(num_feature_xd, num_feature_xd*2)
        self.Conv3 = GCNConv(num_feature_xd*2, num_feature_xd*4)
        self.relu = nn.ReLU()
        self.fc_g1 = nn.Linear(546, 1024)
        self.fc_g2 = nn.Linear(1024, output_dim)
        self.dropout = nn.Dropout(dropout)

        #protien sequence branch (LSTM)
        self.embedding_xt = nn.Embedding(num_feature_xt+1, embed_dim)
        self.LSTM_xt_1 = nn.LSTM(self.embed_dim, self.embed_dim, self.lstm_layer, batch_first = True, bidirectional=True)
        self.fc_xt = nn.Linear(1000*256, output_dim)

        #combined layers
        self.final = MLP(
            input_dim=2*output_dim,
            hidden_dims=[1024, 512, n_output],
            dropout=dropout,
            activation="relu",
        )

    def forward(self, data):
        data = data[0]
        x, edge_index, batch, target = data.x, data.edge_index, data.batch, data.target
        hidden, cell = self.init_hidden(target.size(0), x.device)
        adj = to_dense_adj(edge_index)

        if self.k1 == 1:
            h1 = self.Conv1(x, edge_index)
            h1 = self.relu(h1)
            h2 = self.Conv2(h1, edge_index)
            h2 = self.relu(h2)
            h3 = self.Conv3(h2, edge_index)
            h3 = self.relu(h3)

        if self.k2 == 2:
            edge_index_square,_ = torch_sparse.spspmm(edge_index, None, edge_index, None, adj.shape[1], adj.shape[1], adj.shape[1], coalesced=True)
            h4 = self.Conv1(x, edge_index_square)
            h4 = self.relu(h4)
            h5 = self.Conv2(h4, edge_index_square)
            h5 = self.relu(h5)

        if self.k3 == 3:
            edge_index_cube,_ = torch_sparse.spspmm(edge_index_square, None, edge_index, None, adj.shape[1], adj.shape[1], adj.shape[1], coalesced=True)
            h6 = self.Conv1(x, edge_index_cube)
            h6 = self.relu(h6)

        concat = torch.cat([h3, h5, h6], dim=1)

        x = gmp(concat, batch) #global_max_pooling

        #flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        drug_feature = self.dropout(x)

        # LSTM layer
        embedded_xt = self.embedding_xt(target)
        LSTM_xt, (hidden, cell) = self.LSTM_xt_1(embedded_xt, (hidden, cell))
        xt = LSTM_xt.contiguous().view(-1, 1000*256)
        protein_feature = self.fc_xt(xt)

        cat_feature = torch.cat((drug_feature, protein_feature), 1)
        out = self.final(cat_feature)
        return out
    
    def init_hidden(self, batch_size=32, device=torch.device('cpu')):
        hidden = torch.zeros(2, batch_size, self.embed_dim).to(device)
        cell = torch.zeros(2, batch_size, self.embed_dim).to(device)
        return hidden, cell
    
    