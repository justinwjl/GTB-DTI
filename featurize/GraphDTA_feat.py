import numpy as np
from rdkit import Chem
import networkx as nx
from rdkit import RDConfig
from featurize.base import *
import torch
from torch_geometric.data import Data
import os

class GraphDTA_featurize:
    def __init__(self, max_prot_len=1000, **config):
        self.feat_name = 'GraphDTA'
        self.atom_map = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                         'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu',
                         'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
        self.degree_map = list(range(0, 11))
        self.numH_map = list(range(0, 11))
        self.implicit_map = list(range(0, 11))

        self.seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
        self.seq_dict = {v: (i + 1) for i, v in enumerate(self.seq_voc)}
        self.max_prot_len = max_prot_len
        fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        self.chem_feature_factory = Chem.ChemicalFeatures.BuildFeatureFactory(fdef_name)
        try:
            self.atom_property = config['atom_property']
            self.hydrogen_info = config['hydrogen_info']
            self.electron_p = config['electron_p']
            self.stereochemistry = config['stereochemistry']
            self.structural = config['structural']
            self.edge = config['edge']
            
            self.feat_name += f'a{int(self.atom_property)}_h{int(self.hydrogen_info)}_e{int(self.electron_p)}'\
                f'_ste{int(self.stereochemistry)}_str{int(self.structural)}_ed{int(self.edge)}'
            self.use_pretrain = True if config['use_pretrain'] else False
            if self.use_pretrain:
                self.feat_name += '_pretrain'
                import esm
                self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
                self.batch_converter = self.alphabet.get_batch_converter()
        except:
            self.atom_property = False
            self.hydrogen_info = False
            self.electron_p = False
            self.stereochemistry = False
            self.structural = False
            self.edge = False
            self.use_pretrain = False
            
    def one_hot_vector(self, val, lst):
        """Converts a value to a one-hot vector based on options in lst"""
        if val not in lst:
            val = lst[-1]
        return map(lambda x: x == val, lst)

    def atom_features(self, atom):
        attributes = one_of_k_encoding_unk(atom.GetSymbol(), self.atom_map) +\
            one_of_k_encoding(atom.GetDegree(), self.degree_map) +\
            one_of_k_encoding_unk(atom.GetTotalNumHs(), self.numH_map) +\
            one_of_k_encoding_unk(atom.GetImplicitValence(), self.implicit_map) +\
            [atom.GetIsAromatic()]
        
        if self.atom_property:
            attributes += self.one_hot_vector(
                atom.GetFormalCharge(),
                [-1, 0, 1]
            )
            attributes += self.one_hot_vector(
                atom.GetExplicitValence(),
                [1, 2, 3, 4, 5]
            )
            attributes += self.one_hot_vector(
            atom.GetHybridization(),
            [Chem.rdchem.HybridizationType.SP,
                            Chem.rdchem.HybridizationType.SP2,
                            Chem.rdchem.HybridizationType.SP3,Chem.rdchem.HybridizationType.SP3D,
                        Chem.rdchem.HybridizationType.SP3D2]
            )
        if self.hydrogen_info:
            attributes += self.one_hot_vector(
                atom.GetNumExplicitHs(),
                [0, 1, 2, 3]
            )
            attributes += one_of_k_encoding_unk(atom.GetNumImplicitHs(), list(range(0, 7)))
        if self.electron_p:
            attributes += self.one_hot_vector(
                atom.GetNumRadicalElectrons(),
                [0, 1]
            )
            attributes.append(0)
            attributes.append(0)
        if self.stereochemistry:
            try:
                attributes += one_of_k_encoding_unk(
                            atom.GetProp('_CIPCode'),
                            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                attributes += [0, 0] + [atom.HasProp('_ChiralityPossible')]
            
        if self.structural:
            attributes += [atom.IsInRing()]

        return np.array(attributes, dtype=np.float32)

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

    def drug_encoding(self, smile):
        smile = normalize_smile(smile)
        mol = Chem.MolFromSmiles(smile)
        c_size = mol.GetNumAtoms()
        feats = self.chem_feature_factory.GetFeaturesForMol(mol)
        features = []
        node_features = []
        for atom in mol.GetAtoms():
            feature = self.atom_features(atom)
            node_features.append(feature)
            # features.append(feature / sum(feature))

        if self.electron_p:
            if not self.stereochemistry and not self.structural:
                donor, acceptor = -1, -2
            elif not self.stereochemistry and self.structural:
                donor, acceptor = -2, -3
            elif self.stereochemistry and not self.structural:
                donor, acceptor = -4, -5
            else:
                donor, acceptor = -5, -6
            
            for i in range(len(feats)):
                if feats[i].GetFamily() == 'Donor':
                    node_list = feats[i].GetAtomIds()
                    for n in node_list:
                        node_features[n][donor] = 1.0
                elif feats[i].GetFamily() == 'Acceptor':
                    node_list = feats[i].GetAtomIds()
                    for n in node_list:
                        node_features[n][acceptor] = 1.0
                    
        for feature in node_features:
            features.append(feature / sum(feature))
        
        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        g = nx.Graph(edges).to_directed()
        edge_index = []
        for e1, e2 in g.edges:
            edge_index.append([e1, e2])

        return c_size, features, edge_index

    def protein_encoding(self, prot):
        if not self.use_pretrain:
            return str2int(prot, self.seq_dict, self.max_prot_len)
        else:
            batch_labels, batch_strs, batch_tokens = self.batch_converter([('prot', prot)])
            # pad_prot = np.pad(batch_tokens[:, :self.max_prot_len], ((0, 0), (0, max(self.max_prot_len-batch_tokens.shape[1], 0))), mode='constant', constant_values=self.alphabet.padding_idx)
            with torch.no_grad():
                results = self.model(torch.LongTensor(batch_tokens), repr_layers=[6], return_contacts=False)
            token_representations = results["representations"][6]
            seq_representations = token_representations[0, 1:].mean(0)
            return seq_representations

    def data_input(self, drugs, proteins, labels):
        
        data_list = []

        drug_dict = dict()
        prot_dict = dict()
        
        for smile in set(drugs):
            drug_dict[smile] = self.drug_encoding(smile)
            
        for prot in set(proteins):
            protein_input = self.protein_encoding(prot)
            if self.use_pretrain:
                prot_dict[prot] = protein_input.unsqueeze(0)
            else:
                prot_dict[prot] = torch.LongTensor([protein_input])
            
        for smile, protein, label in list(zip(drugs, proteins, labels)):

            c_size, features, edge_index = drug_dict[smile]
            if len(edge_index) == 0:
                edge_index = torch.empty((0, 2), dtype=torch.long)
            data = Data(
                x=torch.Tensor(features),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                y=torch.FloatTensor([label])
            )

            data.target = prot_dict[protein]
            data.__setitem__('c_size', torch.LongTensor([c_size]))
            
            data_list.append([data])
        return data_list

