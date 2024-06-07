import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool
from torchdrug.layers import MLP
import pandas as pd
    

class GraphCPI_GATGCN(nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2, **config):

        super(GraphCPI_GATGCN, self).__init__()

        self.n_output = n_output
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.fc_g1 = nn.Linear(num_features_xd*10*2, 1500)
        self.fc_g2 = nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1D convolution on protein sequence
        embedding_matrix = pd.read_csv(config['feat_root'] + f'/embedding_3_100d.csv')
        embedding_weight = embedding_matrix.values
        self.embedding_xt = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_weight), freeze=False)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*93, output_dim)

        # combined layers
        self.final = MLP(
            input_dim=2*output_dim,
            hidden_dims=[1024, 512, self.n_output],
            dropout=dropout,
            activation="relu",
        )
        
    def _forward_drug(self, molecules, edge_index, batch):
        x = self.conv1(molecules, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        return x
    
    def _forward_protein(self, molecules):
        embedded_mol = self.embedding_xt(molecules)
        conv_xt = self.conv_xt_1(embedded_mol)
        # flatten
        xt = conv_xt.view(-1, 32 * 93)
        feature = self.fc1_xt(xt)
        return feature
    
    def forward(self, data):
        data = data[0]
        x, edge_index, batch, target = data.x, data.edge_index, data.batch, data.target

        drug_feature = self._forward_drug(x, edge_index, batch)
        protein_feature = self._forward_protein(target)
        hidden = torch.cat((drug_feature, protein_feature), 1)
        out = self.final(hidden)
        return out
    
    
class GraphCPI_GAT(nn.Module):
    def __init__(self, num_features_xd=78, n_output=1, num_features_xt=25,
                     n_filters=32, embed_dim=128, output_dim=128, dropout=0.2, **config):
        super(GraphCPI_GAT, self).__init__()

        self.n_output = n_output
        # graph layers
        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        # 1D convolution on protein sequence
        embedding_matrix = pd.read_csv(config['feat_root'] + f'/embedding_3_100d.csv')
        embedding_weight = embedding_matrix.values
        self.embedding_xt = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_weight), freeze=False)
        self.conv_xt1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*93, output_dim)

        # combined layers
        self.final = MLP(
            input_dim=2*output_dim,
            hidden_dims=[1024, 512, self.n_output],
            dropout=dropout,
            activation="relu",
        )

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def _forward_drug(self, molecules, edge_index, batch):
        x = F.dropout(molecules, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)          # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)
        return x
    
    def _forward_protein(self, molecules):
        embedded_mol = self.embedding_xt(molecules)
        conv_xt = self.conv_xt1(embedded_mol)
        conv_xt = self.relu(conv_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 93)
        feature = self.fc1_xt(xt)
        return feature
    
    def forward(self, data):
        data = data[0]
        # graph the_input feed-forward
        x, edge_index, batch, target = data.x, data.edge_index, data.batch, data.target

        drug_feature = self._forward_drug(x, edge_index, batch)
        protein_feature = self._forward_protein(target)
        hidden = torch.cat((drug_feature, protein_feature), 1)
        out = self.final(hidden)
        return out

    
class GraphCPI_GCN(nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128,num_features_xd=78, num_features_xt=25, output_dim=128, dropout=0.2, **config):

        super(GraphCPI_GCN, self).__init__()
        # 78 91 102 106 109 110
        # num_features_xd = 110
        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # protein sequence branch (1d conv)
        embedding_matrix = pd.read_csv(config['feat_root'] + f'/embedding_3_100d.csv')
        # embedding_matrix = pd.read_csv('/disk3/jyxie/code/zxn/Antibody-Benchmark/data/Davis/regression_random_42/GraphCPI/embedding_3_100d.csv')
        embedding_weight = embedding_matrix.values
        self.embedding_xt = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_weight), freeze=True)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*93, output_dim)

        # combined layers
        self.final = MLP(
            input_dim=2*output_dim,
            hidden_dims=[1024, 512, self.n_output],
            dropout=dropout,
            activation="relu",
        )

    def _forward_drug(self, molecules, edge_index, batch):
        x = self.conv1(molecules, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)       # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)
        return x
    
    def _forward_protein(self, molecules):
        embedded_mol = self.embedding_xt(molecules)
        conv_xt = self.conv_xt_1(embedded_mol)
        # flatten
        xt = conv_xt.view(-1, 32 * 93)
        feature = self.fc1_xt(xt)
        return feature
    
    def forward(self, data):
        data = data[0]
        # get graph the_input
        x, edge_index, batch, target = data.x, data.edge_index, data.batch, data.target

        drug_feature = self._forward_drug(x, edge_index, batch)
        protein_feature = self._forward_protein(target)
        hidden = torch.cat((drug_feature, protein_feature), 1)
        out = self.final(hidden)
        return out
    
    
class GraphCPI_GIN(nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2, **config):

        super(GraphCPI_GIN, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = nn.Sequential(nn.Linear(num_features_xd, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = nn.BatchNorm1d(dim)

        nn2 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = nn.BatchNorm1d(dim)

        nn3 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = nn.BatchNorm1d(dim)

        nn4 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = nn.BatchNorm1d(dim)

        nn5 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = nn.BatchNorm1d(dim)

        self.fc1_xd = nn.Linear(dim, output_dim)

        # 1D convolution on protein sequence
        embedding_matrix = pd.read_csv(config['feat_root'] + f'/embedding_3_100d.csv')
        embedding_weight = embedding_matrix.values
        self.embedding_xt = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_weight), freeze=False)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*93, output_dim)

        # combined layers
        self.final = MLP(
            input_dim=2*output_dim,
            hidden_dims=[1024, 512, self.n_output],
            dropout=dropout,
            activation="relu",
        )     

    def _forward_drug(self, molecules, edge_index, batch):
        
        x = F.relu(self.conv1(molecules, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)
        return x
    
    def _forward_protein(self, molecules):
        embedded_mol = self.embedding_xt(molecules)
        conv_xt = self.conv_xt_1(embedded_mol)
        # flatten
        xt = conv_xt.view(-1, 32 * 93)
        feature = self.fc1_xt(xt)
        return feature
    
    def forward(self, data):
        data = data[0]
        x, edge_index, batch, target = data.x, data.edge_index, data.batch, data.target

        drug_feature = self._forward_drug(x, edge_index, batch)
        protein_feature = self._forward_protein(target)
        hidden = torch.cat((drug_feature, protein_feature), 1)
        out = self.final(hidden)
        return out
    