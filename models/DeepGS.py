
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import pandas as pd
from torch_geometric.data import Data, Batch
class GATLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(3), self.alpha)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, Wh)

        return F.elu(h_prime) if self.concat else h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        b = Wh.size()[0]
        N = Wh.size()[1]

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat_interleave(N, dim=0).view(b, N*N, self.out_features)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)

        return all_combinations_matrix.view(b, N, N, 2 * self.out_features)


class DeepGS(nn.Module):
    def __init__(self, n_output, dim, layer_gnn, side, layer_cnn, layer_out, **config):
        super(DeepGS, self).__init__()
        # dim=10
        finger_dict = pickle.load(open(config['feat_root'] + '/atom_dict', 'rb'))
        self.layer_cnn = layer_cnn
        self.layer_gnn = layer_gnn
        self.layer_out = layer_out
        self.embed_fingerprint = nn.Embedding(len(finger_dict), dim)
        # predefined word embedding
        self.embed_word = nn.Embedding(10000, 100)
        embedding_matrix = pd.read_csv(config['feat_root'] + f'/embedding_3_100d.csv')
        pro_embedding_matrix = embedding_matrix.values
        self.embed_word.weight = nn.Parameter(torch.tensor(pro_embedding_matrix, dtype=torch.float32))
        self.embed_word.weight.requires_grad = False

        embedding_matrix = pd.read_csv(config['feat_root'] + f'/smile_embedding_1_100d.csv')
        smi_embedding_matrix = embedding_matrix.values
        self.embed_smile = nn.Embedding(100, 100)
        self.embed_smile.weight = nn.Parameter(torch.tensor(smi_embedding_matrix, dtype=torch.float32))
        self.embed_smile.weight.requires_grad = False

        # define 3 dense layer
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        #GCN
        self.gcn1 = GATConv(dim, dim, heads=10, dropout=0.2)
        self.gcn2 = GATConv(dim * 10, 128, dropout=0.2)
        self.gat_layers = [GATLayer(dim, dim, dropout=0.2, alpha=0.1, concat=True)
                           for _ in range(10)]
        for i, layer in enumerate(self.gat_layers):
            self.add_module('gat_layer_{}'.format(i), layer)
        self.gat_out = GATLayer(10 * 10, 128, dropout=0.2, alpha=0.1, concat=False)
        self.W_comp = nn.Linear(128, 128)
        self.fc_g1 = nn.Linear(128, 128)
        
        window = 2 * side + 1
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*window+1,
                     stride=1, padding=window) for _ in range(layer_cnn)])
        
#         self.W_rnn = nn.GRU(bidirectional=False, num_layers=2, input_size=100, hidden_size=100)
        self.W_rnn = nn.GRU(bidirectional=True, num_layers=1, input_size=100, hidden_size=100)

        self.W_attention = nn.Linear(dim, 100)
        self.P_attention = nn.Linear(100, 100)
#         self.W_out = nn.ModuleList([nn.Linear(2*100+128, 2*100+128)
#                                     for _ in range(layer_output)])
#         # self.W_interaction = nn.Linear(2*dim, 2)
#         self.W_interaction = nn.Linear(2*100+128, 1)
        self.W_out = nn.ModuleList([nn.Linear(100+200+128, 100+200+128)
                                    for _ in range(layer_out)])
        self.W_interaction = nn.Linear(100+200+128, n_output)
        
        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def attention_cnn(self, xs, layer, xs_mask):
        """The attention mechanism is applied to the last layer of CNN."""

        xs = torch.unsqueeze(xs, 1)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(xs, 1)
        xs = torch.sum(xs, 1) / (torch.sum(xs_mask.unsqueeze(-1), 1)+1e-8)

        return xs
    
    def rnn(self, xs, xs_mask):

        xs, h = self.W_rnn(xs)
        xs = torch.relu(xs)
        # xs = torch.mean(xs, 1)
        xs = xs * xs_mask.unsqueeze(-1)
    
        sum_xs = torch.sum(xs, dim=1)
        sum_mask = torch.sum(xs_mask, dim=1, keepdim=True)
        
        epsilon = 1e-8
        avg_xs = sum_xs / (sum_mask + epsilon)
        return avg_xs

    
    def forward(self, data):
        data = data[0]
        fingerprints, words, smiles, edge_index = data.x, data.prot, data.smile, data.edge_index
        prot_mask, smile_mask = data.prot_mask, data.smile_mask
        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)

        x = F.dropout(fingerprint_vectors, p=0.2, training=self.training)
        x = F.relu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        batch = data.batch
        x = gmp(x, batch)          # global max pooling
        x = self.fc_g1(x)
        compound_vector = self.relu(x)
        

        """Protein vector with attention-CNN."""
        word_vectors = self.embed_word(words)
        protein_vector = self.attention_cnn(word_vectors, self.layer_cnn, prot_mask)
#         protein_vector = self.rnn(word_vectors, layer_cnn)

        """smile vector with attention-CNN."""
        # add the feature of word embedding of SMILES string
        smile_vectors = self.embed_smile(smiles)
        after_smile_vectors = self.rnn(smile_vectors, smile_mask)

        """Concatenate the above two vectors and output the interaction."""
        # concatenate with three types of features
        cat_vector = torch.cat((compound_vector, protein_vector, after_smile_vectors), 1)
        for j in range(self.layer_out):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)

        return interaction