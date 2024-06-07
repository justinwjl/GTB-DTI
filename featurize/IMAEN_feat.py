import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data.data import Data
from featurize.base import one_of_k_encoding, one_of_k_encoding_unk

class IMAEN_featurize:
    def __init__(self, **config):
        self.feat_name = "IMAEN"
        self.seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
        self.seq_dict = {v: (i + 1) for i, v in enumerate(self.seq_voc)}
        self.seq_dict_len = len(self.seq_dict)
        self.max_seq_len = config['max_seq_len'] if 'max_seq_len' in config else 1000

    def atom_features(self, atom):
        return np.array(self.one_of_k_encoding_unk(atom.GetSymbol(),
                                                   ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                                                    'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag',
                                                    'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
                                                    'Cd',
                                                    'In', 'Mn',
                                                    'Zr',
                                                    'Cr',
                                                    'Pt', 'Hg', 'Pb', 'Unknown']) +
                        self.one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        self.one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        self.one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        [atom.GetIsAromatic()])

    def smile_to_graph(self, smile):
        mol = Chem.MolFromSmiles(smile)

        c_size = mol.GetNumAtoms()

        features = []
        for atom in mol.GetAtoms():
            feature = self.atom_features(atom)
            features.append(feature / sum(feature))

        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        g = nx.Graph(edges).to_directed()
        edge_index = []
        for e1, e2 in g.edges():
            edge_index.append([e1, e2])

        return c_size, features, edge_index

    def seq_cat(self, prot):
        x = np.zeros(self.max_seq_len)
        for i, ch in enumerate(prot[:self.max_seq_len]):
            x[i] = self.seq_dict[ch]
        return x

    def data_input(self, drugs, proteins, labels):

        targets = [self.seq_cat(t) for t in proteins]
        targets = np.asarray(targets)

        data_list = []
        for drug, protein, label in (
                list(zip(drugs, targets, labels))):
            d_size, features, edge_index = self.smile_to_graph(drug)

            data = Data(x=torch.Tensor(features),
                        edge_index=torch.LongTensor(edge_index).transpose(1, 0),  # 行列转置
                        y=torch.FloatTensor([label]))
            data.target = torch.LongTensor([protein])
            # data.__setitem__('c_size', torch.LongTensor([c_size]))
            data_list.append([data])
        return data_list
