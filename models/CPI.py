import torch
import torch.nn as nn
import pickle


class CPINet(nn.Module):
    def __init__(self, n_output, dim, layer_gnn, side, layer_cnn, layer_out, dropout=0.1, **config):
        super(CPINet, self).__init__()
        
        atom_dict = pickle.load(open(config['feat_root'] + '/atom_dict', 'rb'))
        amino_dict = pickle.load(open(config['feat_root'] + '/amino_dict', 'rb'))
        window = 2 * side + 1
        self.embed_fingerprint = nn.Embedding(len(atom_dict), dim)
        self.embed_word = nn.Embedding(len(amino_dict), dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*window+1,
                     stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim)
                                    for _ in range(layer_out)])
        # self.W_interaction = nn.Linear(2*dim, 2)
        self.W_interaction = nn.Linear(2*dim, n_output)
        
        self.layer_cnn = layer_cnn
        self.layer_gnn = layer_gnn
        self.layer_out = layer_out
        
        self.dropout = nn.Dropout(dropout)

    def gnn(self, xs, A, layer, atom_mask):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        atom_mask = atom_mask.unsqueeze(-1)
        xs = xs * atom_mask
        xs = torch.sum(xs, 1) / (torch.sum(atom_mask, 1) + 1e-6)
        # return torch.unsqueeze(torch.mean(xs, 1), 1)
        return xs.unsqueeze(1)

    def attention_cnn(self, x, xs, layer, mask):
        """The attention mechanism is applied to the last layer of CNN."""

        xs = torch.unsqueeze(xs, 1)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(xs, 1)

        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        
        mask = mask.unsqueeze(-1)
        hs = hs * mask
        weights = torch.tanh(torch.bmm(h, hs.transpose(1, 2)))
        ys = weights.transpose(1, 2) * hs

        ys = torch.sum(ys, 1) / (torch.sum(mask, 1) + 1e-6)
        return ys.unsqueeze(1)

    def forward(self, data):

        data = data[0]
        atoms, atoms_mask, adjacency, amino, amino_mask = data.x, data.x_mask, data.adj, data.prot, data.prot_mask
        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(atoms)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, self.layer_gnn, atoms_mask)

        """Protein vector with attention-CNN."""
        word_vectors = self.embed_word(amino)
        protein_vector = self.attention_cnn(compound_vector,
                                            word_vectors, self.layer_cnn, amino_mask)

        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 2)
        cat_vector = torch.squeeze(cat_vector, 1)
        for j in range(self.layer_out):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
            cat_vector = self.dropout(cat_vector)
        interaction = self.W_interaction(cat_vector)

        return interaction
