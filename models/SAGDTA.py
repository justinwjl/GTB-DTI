# This is the implementation for the SAGDTA.py of global architecture, i.e., SAGDTA.py(Glob_Pool).
#
# Author: Shugang Zhang


import torch
import torch.nn as nn
from torch_geometric.nn import SAGPooling, GCNConv, global_max_pool as gmp, global_mean_pool as gap


# 3DGCN model
class SAGNet(nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128, num_features_xd=78, num_features_xt=25, output_dim=128,
                 dropout=0.1, **config):
        super(SAGNet, self).__init__()
        
        self.model_name = 'SAGDTA'
        self.pooling_ratio = 1.0

        # SMILES graph branch
        self.n_output = n_output

        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.bn1 = nn.BatchNorm1d(num_features_xd)

        self.conv2 = GCNConv(num_features_xd, num_features_xd)
        self.bn2 = nn.BatchNorm1d(num_features_xd)

        self.conv3 = GCNConv(num_features_xd, num_features_xd)
        self.bn3 = nn.BatchNorm1d(num_features_xd)

        self.pool1 = SAGPooling(3 * num_features_xd, ratio=self.pooling_ratio, GNN=GCNConv, nonlinearity=nn.functional.tanh)

        self.fc_g1 = torch.nn.Linear(2 * 3 * num_features_xd, output_dim)  # 1024
        self.bn4 = nn.BatchNorm1d(output_dim)  # 1024
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.bn5 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=4 * n_filters, kernel_size=3, padding=1)
        self.bn_xt1 = nn.BatchNorm1d(4 * n_filters)
        self.conv_xt_2 = nn.Conv1d(in_channels=4 * n_filters, out_channels=2 * n_filters, kernel_size=3, padding=1)
        self.bn_xt2 = nn.BatchNorm1d(2 * n_filters)
        self.conv_xt_3 = nn.Conv1d(in_channels=2 * n_filters, out_channels=n_filters, kernel_size=3, padding=1)
        self.bn_xt3 = nn.BatchNorm1d(n_filters)
        self.fc1_xt = nn.Linear(32 * 128, output_dim)
        self.bn6 = nn.BatchNorm1d(output_dim)

        # combined layers.py
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        # self.fc1 = nn.Linear(output_dim, 1024)  # protein only
        self.bn7 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn8 = nn.BatchNorm1d(512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        data = data[0]
        # get graph the_input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # get protein the_input
        target = data.target  # 512x1000

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = x

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        x2 = x

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = self.relu(x)
        x3 = x

        x = torch.cat([x1, x2, x3], dim=1)  # 16571*234
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.fc_g1(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)

        # 1d conv layers.py
        embedded_xt = self.embedding_xt(target)  # 512*1000*128
        conv_xt = self.conv_xt_1(embedded_xt)  # 512*(32*4)*128
        conv_xt = self.bn_xt1(conv_xt)
        conv_xt = self.relu(conv_xt)
        conv_xt = self.conv_xt_2(conv_xt)  # 512*(32*2)*128
        conv_xt = self.bn_xt2(conv_xt)
        conv_xt = self.relu(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)  # 512*(32*1)*128
        conv_xt = self.bn_xt3(conv_xt)
        conv_xt = self.relu(conv_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 128)
        xt = self.fc1_xt(xt)
        xt = self.relu(xt)
        xt = self.bn6(xt)
        xt = self.dropout(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # xc = x  # protein only
        # add some dense layers.py
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.bn7(xc)
        xc = self.dropout(xc)

        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.bn8(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


# This is the implementation for the SAGDTA.py of hierarchical architecture, i.e., SAGDTA.py(HierPool).
class SAGNet_HIER(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128, num_features_xd=78, num_features_xt=25, output_dim=128,
                 dropout=0.1, **config):
        super(SAGNet_HIER, self).__init__()
        self.model_name = 'SAGDTA'
        self.pooling_ratio = 1.0

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.bn1 = nn.BatchNorm1d(num_features_xd)
        self.pool1 = SAGPooling(num_features_xd, ratio=self.pooling_ratio, GNN=GCNConv, nonlinearity=nn.functional.tanh)

        self.conv2 = GCNConv(num_features_xd, num_features_xd)
        self.bn2 = nn.BatchNorm1d(num_features_xd)
        self.pool2 = SAGPooling(num_features_xd, ratio=self.pooling_ratio, GNN=GCNConv, nonlinearity=nn.functional.tanh)  # ratio本身可理解为另一种形式的filter数

        self.conv3 = GCNConv(num_features_xd, num_features_xd)
        self.bn3 = nn.BatchNorm1d(num_features_xd)
        self.pool3 = SAGPooling(num_features_xd, ratio=self.pooling_ratio, GNN=GCNConv, nonlinearity=nn.functional.tanh)

        self.fc_g1 = torch.nn.Linear(num_features_xd * 2, output_dim)  # 1024
        self.bn4 = nn.BatchNorm1d(output_dim)  # 1024
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.bn5 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=4 * n_filters, kernel_size=3, padding=1)
        self.bn_xt1 = nn.BatchNorm1d(4 * n_filters)
        self.conv_xt_2 = nn.Conv1d(in_channels=4 * n_filters, out_channels=2 * n_filters, kernel_size=3, padding=1)
        self.bn_xt2 = nn.BatchNorm1d(2 * n_filters)
        self.conv_xt_3 = nn.Conv1d(in_channels=2 * n_filters, out_channels=n_filters, kernel_size=3, padding=1)
        self.bn_xt3 = nn.BatchNorm1d(n_filters)
        self.fc1_xt = nn.Linear(32 * 128, output_dim)
        self.bn6 = nn.BatchNorm1d(output_dim)

        # combined layers.py
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn8 = nn.BatchNorm1d(512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        data = data[0]
        # get graph the_input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # get protein the_input
        target = data.target  # 512x1000

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = self.relu(x)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # x = torch.cat([x1, x2, x3], dim=1)
        x = x1 + x2 + x3

        x = self.fc_g1(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        # x = self.fc_g2(x)
        # x = self.bn5(x)
        # x = self.relu(x)
        # x = self.dropout(x)

        # 1d conv layers.py
        embedded_xt = self.embedding_xt(target)  # 512*1000*128
        conv_xt = self.conv_xt_1(embedded_xt)  # 512*(32*4)*128
        conv_xt = self.bn_xt1(conv_xt)
        conv_xt = self.relu(conv_xt)
        conv_xt = self.conv_xt_2(conv_xt)  # 512*(32*2)*128
        conv_xt = self.bn_xt2(conv_xt)
        conv_xt = self.relu(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)  # 512*(32*1)*128
        conv_xt = self.bn_xt3(conv_xt)
        conv_xt = self.relu(conv_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 128)
        xt = self.fc1_xt(xt)
        xt = self.relu(xt)
        xt = self.bn6(xt)
        xt = self.dropout(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers.py
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.bn7(xc)
        xc = self.dropout(xc)

        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.bn8(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
