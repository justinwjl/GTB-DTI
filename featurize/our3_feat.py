import os
import torch
from torch_geometric.data import Data
from rdkit import Chem
import networkx as nx
from rdkit import Chem
from rdkit import RDConfig
from featurize.base import *
from sklearn.metrics import pairwise_distances
from featurize.base import pad_or_truncate
from featurize.base import *
'''
https://github.com/guaguabujianle/MGraphDTA/tree/dev
'''


class our3_featurize:
    def __init__(self, max_prot_len=1000):
        self.feat_name = 'Our3'
        self.seq_dict = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                         "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                         "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                         "U": 19, "T": 20, "W": 21,
                         "V": 22, "Y": 23, "X": 24,
                         "Z": 25}
        
        self.smile_dict = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                           "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                           "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                           "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                           "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                           "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                           "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                           "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

        self.max_prot_len = max_prot_len
        self.max_smile_len = 100
        fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        self.chem_feature_factory = Chem.ChemicalFeatures.BuildFeatureFactory(fdef_name)
        self.adj_max = 100


    def get_atom_features(self, atom, one_hot_formal_charge=True):
        """Calculate atom features.

        Args:
            atom (rdchem.Atom): An RDKit Atom object.
            one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

        Returns:
            A 1-dimensional array (ndarray) of atom features.
        """
        attributes = []
        
        # atom_map = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
        #                  'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu',
        #                  'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
        # attributes += one_of_k_encoding_unk(atom.GetSymbol(), atom_map) 
            
        attributes += self.one_hot_vector(
            atom.GetAtomicNum(),
            [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]
        )

        attributes += self.one_hot_vector(
            len(atom.GetNeighbors()),
            [0, 1, 2, 3, 4, 5]
        )

        attributes += self.one_hot_vector(
            atom.GetTotalNumHs(),
            [0, 1, 2, 3, 4]
        )

        # attributes += self.one_hot_vector(
        #     atom.GetSymbol(),
        #     ['H', 'C', 'N', 'O', 'F', 'Cl', 'S', 'Br', 'I', ]
        # )
        
        attributes += self.one_hot_vector(
            atom.GetHybridization(),
            [Chem.rdchem.HybridizationType.SP,
                            Chem.rdchem.HybridizationType.SP2,
                            Chem.rdchem.HybridizationType.SP3]
        )
        
        if one_hot_formal_charge:
            attributes += self.one_hot_vector(
                atom.GetFormalCharge(),
                [-1, 0, 1]
            )
            attributes += self.one_hot_vector(
                atom.GetExplicitValence(),
                [1, 2, 3, 4, 5]
            )
            # attributes += self.one_hot_vector(
            #     atom.GetImplicitValence(),
            #     [0, 1]
            # )
            attributes += self.one_hot_vector(
                atom.GetNumExplicitHs(),
                [0, 1, 2, 3]
            )
            # attributes += self.one_hot_vector(
            #     atom.GetNumRadicalElectrons(),
            #     [0, 1]
            # )
        else:
            attributes.append(atom.GetFormalCharge())
            attributes.append(atom.GetExplicitValence())
            # attributes.append(atom.GetImplicitValence())
            attributes.append(atom.GetNumExplicitHs())
            # attributes.append(atom.GetNumRadicalElectrons())

        attributes.append(atom.IsInRing())
        attributes.append(atom.GetIsAromatic())
        
        attributes.append(0)
        attributes.append(0)

        return np.array(attributes, dtype=np.float32)

    def statis_test(self, smiles):
        attributes1, attributes2, attributes3, attributes4, attributes5 = [], [], [], [], []
        for smile in smiles:
            mol = Chem.MolFromSmiles(smile)
            try:
                mol = Chem.AddHs(mol)
                Chem.AllChem.EmbedMolecule(mol, maxAttempts=5000)
                Chem.AllChem.UFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)
            except:
                Chem.AllChem.Compute2DCoords(mol)
            for atom in mol.GetAtoms():
                attributes1.append(atom.GetFormalCharge())
                attributes2.append(atom.GetExplicitValence())
                attributes3.append(atom.GetImplicitValence())
                attributes4.append(atom.GetNumExplicitHs())
                attributes5.append(atom.GetNumRadicalElectrons())
            
        from collections import Counter
        print(Counter(attributes1))
        print(Counter(attributes2))
        print(Counter(attributes3))
        print(Counter(attributes4))
        print(Counter(attributes5))

    def one_hot_vector(self, val, lst):
        """Converts a value to a one-hot vector based on options in lst"""
        if val not in lst:
            val = lst[-1]
        return map(lambda x: x == val, lst)
    
    def featurize_mol(self, mol, add_dummy_node, one_hot_formal_charge):
        """Featurize molecule.

        Args:
            mol (rdchem.Mol): An RDKit Mol object.
            add_dummy_node (bool): If True, a dummy node will be added to the molecular graph.
            one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

        Returns:
            A tuple of molecular graph descriptors (node features, adjacency matrix, distance matrix).
        """
        feats = self.chem_feature_factory.GetFeaturesForMol(mol)
        node_features = np.array([self.get_atom_features(atom, one_hot_formal_charge)
                                for atom in mol.GetAtoms()])
        
        for i in range(len(feats)):
            if feats[i].GetFamily() == 'Donor':
                node_list = feats[i].GetAtomIds()
                for n in node_list:
                    node_features[n][-1] = 1.0
            elif feats[i].GetFamily() == 'Acceptor':
                node_list = feats[i].GetAtomIds()
                for n in node_list:
                    node_features[n][-2] = 1.0

        adj_matrix = np.eye(mol.GetNumAtoms())
        for bond in mol.GetBonds():
            begin_atom = bond.GetBeginAtom().GetIdx()
            end_atom = bond.GetEndAtom().GetIdx()
            adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1

        conf = mol.GetConformer()
        pos_matrix = np.array([[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                            for k in range(mol.GetNumAtoms())])
        dist_matrix = pairwise_distances(pos_matrix)

        if add_dummy_node:
            m = np.zeros((node_features.shape[0] + 1, node_features.shape[1] + 1))
            m[1:, 1:] = node_features
            m[0, 0] = 1.
            node_features = m

            m = np.zeros((adj_matrix.shape[0] + 1, adj_matrix.shape[1] + 1))
            m[1:, 1:] = adj_matrix
            adj_matrix = m

            m = np.full((dist_matrix.shape[0] + 1, dist_matrix.shape[1] + 1), 1e6)
            m[1:, 1:] = dist_matrix
            dist_matrix = m

        return node_features, adj_matrix, dist_matrix

    def graph_build(self, smile, add_dummy_node=True, one_hot_formal_charge=False):

        mol = Chem.MolFromSmiles(smile)
        try:
            mol = Chem.AddHs(mol)
            Chem.AllChem.EmbedMolecule(mol, maxAttempts=5000)
            Chem.AllChem.UFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
        except:
            Chem.AllChem.Compute2DCoords(mol)

        afm, adj, dist = self.featurize_mol(mol, add_dummy_node, one_hot_formal_charge)

        return afm, adj, dist
    
    def get_nodes(self, g):
        feat = []
        for n, d in g.nodes(data=True):
            h_t = []
            h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F', 'Cl', 'S', 'Br', 'I', ]]
            h_t.append(d['a_num'])
            h_t.append(d['acceptor'])
            h_t.append(d['donor'])
            h_t.append(int(d['aromatic']))
            h_t += [int(d['hybridization'] == x)
                    for x in (Chem.rdchem.HybridizationType.SP,
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3)]
            h_t.append(d['num_h'])
            # 5 more
            h_t.append(d['ExplicitValence'])
            h_t.append(d['FormalCharge'])
            h_t.append(d['ImplicitValence'])
            h_t.append(d['NumExplicitHs'])
            h_t.append(d['NumRadicalElectrons'])
            feat.append((n, h_t))
        feat.sort(key=lambda item: item[0])
        node_attr = torch.FloatTensor([item[1] for item in feat])

        return node_attr

    def get_edges(self, g):
        e = {}
        for n1, n2, d in g.edges(data=True):
            e_t = [int(d['b_type'] == x)
                   for x in (Chem.rdchem.BondType.SINGLE,
                             Chem.rdchem.BondType.DOUBLE,
                             Chem.rdchem.BondType.TRIPLE,
                             Chem.rdchem.BondType.AROMATIC)]

            e_t.append(int(d['IsConjugated'] == False))
            e_t.append(int(d['IsConjugated'] == True))
            e[(n1, n2)] = e_t

        if len(e) == 0:
            return torch.LongTensor([[0], [0]]), torch.FloatTensor([[0, 0, 0, 0, 0, 0]])

        edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
        edge_attr = torch.FloatTensor(list(e.values()))
        return edge_index, edge_attr

    def mol2graph(self, mol):
        if mol is None:
            return None
        feats = self.chem_feature_factory.GetFeaturesForMol(mol)
        g = nx.DiGraph()

        # Create nodes
        for i in range(mol.GetNumAtoms()):
            atom_i = mol.GetAtomWithIdx(i)
            g.add_node(i,
                       a_type=atom_i.GetSymbol(),
                       a_num=atom_i.GetAtomicNum(),
                       acceptor=0,
                       donor=0,
                       aromatic=atom_i.GetIsAromatic(),
                       hybridization=atom_i.GetHybridization(),
                       num_h=atom_i.GetTotalNumHs(),

                       # 5 more node features
                       ExplicitValence=atom_i.GetExplicitValence(),
                       FormalCharge=atom_i.GetFormalCharge(),
                       ImplicitValence=atom_i.GetImplicitValence(),
                       NumExplicitHs=atom_i.GetNumExplicitHs(),
                       NumRadicalElectrons=atom_i.GetNumRadicalElectrons(),
                       )

        for i in range(len(feats)):
            if feats[i].GetFamily() == 'Donor':
                node_list = feats[i].GetAtomIds()
                for n in node_list:
                    g.nodes[n]['donor'] = 1
            elif feats[i].GetFamily() == 'Acceptor':
                node_list = feats[i].GetAtomIds()
                for n in node_list:
                    g.nodes[n]['acceptor'] = 1

        # Read Edges
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                e_ij = mol.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    g.add_edge(i, j,
                               b_type=e_ij.GetBondType(),
                               # 1 more edge features 2 dim
                               IsConjugated=int(e_ij.GetIsConjugated()),
                               )

        node_attr = self.get_nodes(g)
        edge_index, edge_attr = self.get_edges(g)

        return node_attr, edge_index, edge_attr



    def drug_encoding(self, smile):
        smile = normalize_smile(smile)
        mol = Chem.MolFromSmiles(smile)

        node_attr, edge_index, edge_attr = self.mol2graph(mol)
        node_attr = (node_attr - node_attr.min()) / (node_attr.max() - node_attr.min())

        return node_attr, edge_index, edge_attr

    def protein_encoding(self, prot):
        return str2int(prot, self.seq_dict, self.max_prot_len)

    def smile_seq_encoding(self, smile):
        return str2int(smile, self.smile_dict, self.max_smile_len)
    
    def pad_array(self, array, shape, dtype=np.float32):
        """Pad a 2-dimensional array with zeros.

        Args:
            array (ndarray): A 2-dimensional array to be padded.
            shape (tuple[int]): The desired shape of the padded array.
            dtype (data-type): The desired data-type for the array.

        Returns:
            A 2-dimensional array of the given shape padded with zeros.
        """
        padded_array = np.zeros(shape, dtype=dtype)
        padded_array[:array.shape[0], :array.shape[1]] = array
        return padded_array
    
    def data_input(self, drugs, proteins, labels):
        data_list = []
        
        drug_dict = dict()
        prot_dict = dict()
        
        for drug in set(drugs):
            afm, adj, dist = self.graph_build(drug)
            afm_pad = pad_or_truncate(afm, max_len=self.adj_max, pad_2d=False)
            adj_pad = pad_or_truncate(adj, max_len=self.adj_max, pad_2d=True)
            dist_pad = pad_or_truncate(dist, max_len=self.adj_max, pad_2d=True)
            smile_seq = self.smile_seq_encoding(drug)
            drug_dict[drug] = (afm_pad, adj_pad, dist_pad, smile_seq)
        
        for prot in set(proteins):
            prot_dict[prot] = self.protein_encoding(prot)
            
        for smile, protein, label in list(zip(drugs, proteins, labels)):
            # afm = smile_graph[smile]
            # adj = smile_graph_adj[smile]
            # dist = smile_graph_dist[smile]
            # afm_pad = pad_or_truncate(afm, max_len=self.adj_max, pad_2d=False)
            # adj_pad = pad_or_truncate(adj, max_len=self.adj_max, pad_2d=True)
            # dist_pad = pad_or_truncate(dist, max_len=self.adj_max, pad_2d=True)
            
            # smile_seq_input = smile_seq[smile]
            afm_pad, adj_pad, dist_pad, smile_seq = drug_dict[smile]
            protein_input = prot_dict[protein]
            data = Data(
                x=torch.FloatTensor([afm_pad]),
                # edge_index=edge_index,
                # edge_attr=edge_attr,
                adj = torch.FloatTensor([adj_pad]),
                dist = torch.FloatTensor([dist_pad]),
                y=torch.FloatTensor([label]),
            )
            # protein_input = self.protein_encoding(protein)
            data.target = torch.LongTensor([protein_input])
            data.smile = torch.LongTensor([smile_seq])
            data_list.append([data])

        return data_list
