import dgl
import torch
import torch.nn as nn
# import pudb
from dgllife.model import GAT
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
from torch.nn.utils.rnn import pad_sequence


class PGraphDTA_CNN(nn.Module):
    """
    Custom Class for DTINetwork based on GraphDTA with CNN for protein sequence. For more information about GraphDTA, please refer to the following paper:
    GraphDTA: predicting drug–target binding affinity with graph neural networks.
    """

    def __init__(self,
                 n_output,
                 prot_dim=1024,
                 in_feats=74,
                 graph_hidden_feats=[74, 128],  # GraphDTA: [74, 128], Original: [32, 32]
                 graph_num_heads=[10, 1],  # GraphDTA: [10, 1], Original: [4, 4]
                 dense_hidden_feats=[1024, 256],  # GraphDTA: [1024, 256], Original: 64
                 dropout=0.2,  # GraphDTA: 0.2
                 verbose=False,
                 num_features_xt=25,
                 embed_dim=128,
                 n_filters=32, **config):

        """
        Args:
            prot_dim (int): Protein embedding dimension.
            in_feats (int): Input feature size for molecules.
            graph_hidden_feats (list): Hidden feature sizes for GAT layers.
            graph_num_heads (list): Number of attention heads for GAT layers.
            dense_hidden_feats (list): Hidden feature sizes for dense layers.
            dropout (float): Dropout rate.
            verbose (bool): Whether to print out information.
            num_features_xt (int): Number of features for protein sequence.
            embed_dim (int): Embedding dimension for protein sequence.
            n_filters (int): Number of filters for 1D CNN of protein sequence.
        """
        super(PGraphDTA_CNN, self).__init__()
        self.verbose = verbose
        self.prot_dim = prot_dim

        self.mol_model = GATEmbedding(in_feats=in_feats,
                                      hidden_feats=graph_hidden_feats,
                                      num_heads=graph_num_heads,
                                      dropouts=[dropout] * len(graph_hidden_feats))
        self.dense = nn.ModuleList()

        # CNN Module for protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt1 = nn.Conv1d(in_channels=1024, out_channels=n_filters, kernel_size=8)
        self.fc_xt1 = nn.Linear(32 * 121, 2 * self.mol_model.gnn_out_feats)
        self.prot_fc = nn.Linear(self.prot_dim, 2 * self.mol_model.gnn_out_feats)
        self.dense.append(nn.Linear(2 * 2 * self.mol_model.gnn_out_feats, dense_hidden_feats[0]))
        for i in range(1, len(dense_hidden_feats)):
            self.dense.append(nn.Linear(dense_hidden_feats[i - 1], dense_hidden_feats[i]))
        self.output = nn.Linear(dense_hidden_feats[-1], n_output)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # self.dataset = config['dataset']

    def forward(self, data):
        """
        Returns:
            torch.Tensor: A batch of predicted DTI binding affinity.
        """
        # mol_graphs (DGLGraph): A batch of DGLGraphs for molecules.
        # target (torch.Tensor): A batch of protein sequences.
        data = data[0]
        # if self.dataset == 'Human':
        #     mol_graphs, target = data.drug[0], data.protein[0]
        # else:
        #     mol_graphs, target = dgl.batch(data.drug), data.protein
        mol_graphs, target = dgl.batch(data.drug), data.protein

        mol_graphs = mol_graphs.to(target.device)
        x_mol = self.mol_model(mol_graphs, in_feats=mol_graphs.ndata['h'], readout=True)
        if self.verbose: print("Molecule tensor shape:", x_mol.shape)
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt1(embedded_xt)
        conv_xt = self.activation(conv_xt)

        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        x_prot = self.fc_xt1(xt)

        x = torch.cat((x_prot, x_mol), dim=1)  # axis=1
        for layer in self.dense:
            x = self.dropout(self.activation(layer(x)))
        return self.output(x)


class PGraphDTA_PLM(nn.Module):
    """
    Custom Class for DTINetwork based on GraphDTA with Protein Language Models (PLM) [SeqVec/DistilProtBERT/ProtBERT]
    for protein sequence. For more information about GraphDTA, please refer to the following paper:
    GraphDTA: predicting drug–target binding affinity with graph neural networks.
    """

    def __init__(self,
                 n_output,
                 prot_model='',
                 in_feats=74,
                 graph_hidden_feats=[74, 128],  # GraphDTA: [74, 128], Original: [32, 32]
                 graph_num_heads=[10, 1],  # GraphDTA: [10, 1], Original: [4, 4]
                 dense_hidden_feats=[1024, 256],  # GraphDTA: [1024, 256], Original: 64
                 dropout=0.2,  # GraphDTA: 0.2
                 verbose=False, **config):

        """
        Args:
            prot_model (string): Protein language model.
            prot_dim (int): Protein embedding dimension.
            in_feats (int): Input feature size for molecules.
            graph_hidden_feats (list): Hidden feature sizes for GAT layers.
            graph_num_heads (list): Number of attention heads for GAT layers.
            dense_hidden_feats (list): Hidden feature sizes for dense layers.
            dropout (float): Dropout rate.
            verbose (bool): Whether to print out information.
        """
        super(PGraphDTA_PLM, self).__init__()
        self.verbose = verbose
        # self.prot_model = prot_model
        self.prot_dim = config['MAX_PROT_LEN']
        self.mol_dim = config['MAX_MOLECULE_LEN']

        self.mol_model = GATEmbedding(in_feats=in_feats,
                                      hidden_feats=graph_hidden_feats,
                                      num_heads=graph_num_heads,
                                      dropouts=[dropout] * len(graph_hidden_feats))
        self.dense = nn.ModuleList()

        self.dist_fc = nn.Linear(self.mol_dim * self.mol_dim, 2 * self.mol_model.gnn_out_feats)
        self.prot_fc = nn.Linear(self.prot_dim, 2 * self.mol_model.gnn_out_feats)
        self.dense.append(nn.Linear(2 * 3 * self.mol_model.gnn_out_feats, dense_hidden_feats[0]))
        for i in range(1, len(dense_hidden_feats)):
            self.dense.append(nn.Linear(dense_hidden_feats[i - 1], dense_hidden_feats[i]))
        self.output = nn.Linear(dense_hidden_feats[-1], n_output)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        """
        Returns:
            torch.Tensor: A batch of predicted DTI binding affinity.
        """
        data = data[0]
        # mol_graphs (DGLGraph): A batch of DGLGraphs for molecules.
        # target (torch.Tensor): A batch of protein sequences.
        mol_graphs, target, dist_dicts = dgl.batch(data.drug), data.protein, data.dist
        x_mol = self.mol_model(mol_graphs, in_feats=mol_graphs.ndata['h'], readout=True)
        if self.verbose: print("Molecule tensor shape:", x_mol.shape)
        x_prot = self.prot_fc(target)
        # pu.db
        # # print("Protein tensor shape:", x_prot.shape)
        # print("Distance tensor shape:", dist_dicts.shape)
        # # print dtype of dist_dicts
        # print("Distance tensor dtype:", dist_dicts.dtype)
        # # print dist_fc nn layer
        # print("Distance fc layer shape:", self.dist_fc)
        # # print dtype of dist_fc nn layer
        # print("Distance fc layer dtype:", self.dist_fc.weight.dtype)
        x_dist = self.dist_fc(dist_dicts)

        x = torch.cat((x_prot, x_mol, x_dist), dim=1)  # axis=1
        for layer in self.dense:
            x = self.dropout(self.activation(layer(x)))
        return self.output(x).squeeze(dim=1)


class GATEmbedding(nn.Module):
    """
    GAT Network Class based on GraphDTA. For more information about GraphDTA, please refer to the following paper:
    GraphDTA: predicting drug–target binding affinity with graph neural networks.
    """

    def __init__(self,
                 in_feats,
                 hidden_feats,
                 num_heads,
                 dropouts):
        """
        Args:
            in_feats (int): Input feature size for molecules.
            hidden_feats (list): Hidden feature sizes for GAT layers.
            num_heads (list): Number of attention heads for GAT layers.
            dropouts (list): Dropout rate for GAT layers.
        """
        super(GATEmbedding, self).__init__()

        self.gnn = GAT(in_feats,
                       hidden_feats=hidden_feats,
                       num_heads=num_heads,
                       feat_drops=dropouts,
                       attn_drops=dropouts)
        if self.gnn.agg_modes[-1] == 'flatten':
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]
        self.gnn_out_feats = gnn_out_feats
        self.readout = WeightedSumAndMax(gnn_out_feats)

    def forward(self, g, in_feats, readout=False):
        """
        Args:
            g (DGLGraph): A DGLGraph for a molecule.
            in_feats (torch.Tensor): Input node features.
            readout (bool): Whether to perform readout.
        Returns:
            torch.Tensor: The output of GAT network.
        """
        node_feats = self.gnn(g, in_feats)
        if readout:
            return self.readout(g, node_feats)
        else:
            batch_num_nodes = g.batch_num_nodes().tolist()
            return pad_sequence(torch.split(node_feats, batch_num_nodes, dim=0), batch_first=True)
