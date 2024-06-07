# GCN based model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp
import numpy as np

class EmbedDTI_Ori(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128, num_features_xd=101, num_features_xt=25, num_features_xc=35, output_dim=128, dropout=0.2, **config):

        super(EmbedDTI_Ori, self).__init__()
        self.model_name = 'EmbedDTI'
        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # clique graph branch
        self.c_conv1 = GCNConv(num_features_xc, num_features_xc)
        self.c_conv2 = GCNConv(num_features_xc, num_features_xc*2)
        self.c_conv3 = GCNConv(num_features_xc*2, num_features_xc * 4)
        self.c_g1 = torch.nn.Linear(num_features_xc*4, 1024)
        self.c_g2 = torch.nn.Linear(1024, output_dim)

        # protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)


        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=256, kernel_size=8)
        self.conv_xt_2 = nn.Conv1d(in_channels=256,out_channels=n_filters,kernel_size=8)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters,out_channels=n_filters,kernel_size=3)
        self.fc1_xt = nn.Linear(32*112, output_dim)


        # combined layers
        self.fc1 = nn.Linear(3*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        # get graph input
        # x 为 features
        atom_data, cli_data = data
        x, edge_index, batch = atom_data.x, atom_data.edge_index, atom_data.batch
        # get clique input
        clique_x, clique_edge, cli_batch = cli_data.x, cli_data.edge_index, cli_data.batch
        # get protein input
        target = atom_data.target

        # SMILES graph embedding
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x) # 得到node-level feature vectors
        x = gmp(x, batch)       # global max pooling
        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # clique graph embedding
        xq = self.c_conv1(clique_x, clique_edge)
        xq = self.relu(xq)

        xq = self.c_conv2(xq, clique_edge)
        xq = self.relu(xq)

        xq = self.c_conv3(xq, clique_edge)
        xq = self.relu(xq) # 得到node-level feature vectors
        xq = gmp(xq, cli_batch)       # global max pooling

        # flatten
        xq = self.relu(self.c_g1(xq))
        xq = self.dropout(xq)
        xq = self.c_g2(xq)
        xq = self.dropout(xq)

        # protein target embedding
        # 1d conv layers
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)

        # flatten
        xt = conv_xt.view(-1, 32 * 112)
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt, xq), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
    
class EmbedDTI_Pre(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128, num_features_xd=101, num_features_xt=25, num_features_xc=35, output_dim=128, dropout=0.2, **config):

        super(EmbedDTI_Pre, self).__init__()
        self.model_name = 'EmbedDTI'
        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # clique graph branch
        self.c_conv1 = GCNConv(num_features_xc, num_features_xc)
        self.c_conv2 = GCNConv(num_features_xc, num_features_xc*2)
        self.c_conv3 = GCNConv(num_features_xc*2, num_features_xc * 4)
        self.c_g1 = torch.nn.Linear(num_features_xc*4, 1024)
        self.c_g2 = torch.nn.Linear(1024, output_dim)

        # protein sequence branch (1d conv)
        # self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        pretrained_embeddings = np.load('models/1mer.npy')
        self.embedding_weights = torch.tensor(pretrained_embeddings,dtype = torch.float32)
        self.embedding_xt = nn.Embedding.from_pretrained(self.embedding_weights)
        # requires_grad指定是否在训练过程中对词向量的权重进行微调
        self.embedding_xt.weight.requires_grad = True

        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=256, kernel_size=8)
        self.conv_xt_2 = nn.Conv1d(in_channels=256,out_channels=n_filters,kernel_size=8)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters,out_channels=n_filters,kernel_size=3)
        self.fc1_xt = nn.Linear(32*112, output_dim)


        # combined layers
        self.fc1 = nn.Linear(3*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data_atom,data_clique):
        # get graph input
        # x 为 features
        x, edge_index, batch = data_atom.x, data_atom.edge_index, data_atom.batch
        # get clique input
        clique_x,clique_edge, clique_batch = data_clique.x, data_clique.edge_index, data_clique.batch
        # get protein input
        target = data_atom.target

        # SMILES graph embedding
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x) # 得到node-level feature vectors
        x = gmp(x, batch)       # global max pooling
        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # clique graph embedding
        xq = self.c_conv1(clique_x, clique_edge)
        xq = self.relu(xq)

        xq = self.c_conv2(xq, clique_edge)
        xq = self.relu(xq)

        xq = self.c_conv3(xq, clique_edge)
        xq = self.relu(xq) # 得到node-level feature vectors
        xq = gmp(xq, clique_batch)       # global max pooling

        # flatten
        xq = self.relu(self.c_g1(xq))
        xq = self.dropout(xq)
        xq = self.c_g2(xq)
        xq = self.dropout(xq)

        # protein target embedding
        # 1d conv layers
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)

        # flatten
        xt = conv_xt.view(-1, 32 * 112)
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt, xq), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
    
class EmbedDTI_Attn(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128, num_features_xd=101, num_features_xt=25, num_features_xc=35, output_dim=128, dropout=0.2, **config):

        super(EmbedDTI_Attn, self).__init__()
        self.model_name = 'EmbedDTI'
        pretrained_embeddings = np.load('featurize/EmbedDTI_encoding/1mer.npy')
        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # clique graph branch
        self.c_conv1 = GCNConv(num_features_xc, num_features_xc)
        self.c_conv2 = GCNConv(num_features_xc, num_features_xc*2)
        self.c_conv3 = GCNConv(num_features_xc*2, num_features_xc * 4)
        self.c_g1 = torch.nn.Linear(num_features_xc*4, 1024)
        self.c_g2 = torch.nn.Linear(1024, output_dim)

        # protein sequence branch (1d conv)
        # self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)

        self.embedding_weights = torch.tensor(pretrained_embeddings, dtype = torch.float32)
        self.embedding_xt = nn.Embedding.from_pretrained(self.embedding_weights)
        # requires_grad指定是否在训练过程中对词向量的权重进行微调
        self.embedding_xt.weight.requires_grad = True

        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=256, kernel_size=8)
        self.conv_xt_2 = nn.Conv1d(in_channels=256,out_channels=n_filters,kernel_size=8)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters,out_channels=n_filters,kernel_size=3)
        self.fc1_xt = nn.Linear(32*112, output_dim)


        # combined layers
        self.fc1 = nn.Linear(3*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        data_atom, data_clique = data
        # get graph input
        # x 为 features
        x, edge_index, batch = data_atom.x, data_atom.edge_index, data_atom.batch
        
        device = x.device
        w_atom = torch.Tensor(x.size()[0],1).to(device)
        w_atom = torch.nn.init.normal_(w_atom,mean=0,std=1)
        # get clique input
        clique_x,clique_edge, clique_batch = data_clique.x, data_clique.edge_index, data_clique.batch
        w_clique = torch.Tensor(clique_x.size()[0],1).to(device)
        w_clique = torch.nn.init.normal_(w_clique,mean=0,std=1)
        # get protein input
        target = data_atom.target

        # SMILES graph embedding
        x = torch.mul(x,w_atom)
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x) # 得到node-level feature vectors
        x = gmp(x, batch)       # global max pooling
        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # clique graph embedding
        xq = torch.mul(clique_x,w_clique)
        xq = self.c_conv1(xq, clique_edge)
        xq = self.relu(xq)

        xq = self.c_conv2(xq, clique_edge)
        xq = self.relu(xq)

        xq = self.c_conv3(xq, clique_edge)
        xq = self.relu(xq) # 得到node-level feature vectors
        xq = gmp(xq, clique_batch)       # global max pooling

        # flatten
        xq = self.relu(self.c_g1(xq))
        xq = self.dropout(xq)
        xq = self.c_g2(xq)
        xq = self.dropout(xq)

        # protein target embedding
        # 1d conv layers
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)

        # flatten
        xt = conv_xt.view(-1, 32 * 112)
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt, xq), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out