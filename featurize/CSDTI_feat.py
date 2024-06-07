import os.path as osp
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
import networkx as nx
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from torch_geometric.data import Data
from featurize.base import *

fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)


class CSDTI_featurize:
    def __init__(self):
        self.feat_name = 'CSDTI'
        self.protein_vocab = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                              "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                              "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                              "U": 19, "T": 20, "W": 21,
                              "V": 22, "Y": 23, "X": 24,
                              "Z": 25}
        self.max_prot_len=1200
        

    def protein_encoding(self, prot):
        return str2int(prot, self.protein_vocab, self.max_prot_len)
    
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

    def get_2hop(self, g):
        As = nx.adjacency_matrix(g)
        As = As.dot(As)
        edge_index2 = torch.LongTensor(As.nonzero())
        return edge_index2

    def mol2graph(self, mol):
        if mol is None:
            return None
        feats = chem_feature_factory.GetFeaturesForMol(mol)
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
                               IsConjugated=int(e_ij.GetIsConjugated()),
                               )

        node_attr = self.get_nodes(g)
        edge_index, edge_attr = self.get_edges(g)
        edge_index2 = self.get_2hop(g)

        return node_attr, edge_index, edge_index2, edge_attr
    
    def data_input(self, drugs, proteins, labels):
        
        data_list = []

        drug_dict = dict()
        prot_dict = dict()
        
        for drug in set(drugs):
            mol = Chem.MolFromSmiles(drug)
            drug_dict[drug] = self.mol2graph(mol)
        
        for prot in set(proteins):
            prot_dict[prot] = self.protein_encoding(prot)
        
        for smile, protein, label in list(zip(drugs, proteins, labels)):
            x, edge_index, edge_index2, edge_attr = drug_dict[smile]
            x = (x - x.min()) / (x.max() - x.min())
            target = prot_dict[protein]
            data = Data(
                x=torch.FloatTensor(x),
                edge_index=edge_index,
                edge_index2=edge_index2,
                edge_attr=edge_attr,
                y=torch.FloatTensor([label]),
            )
            data.target = torch.LongTensor([target])
            data_list.append([data])
        return data_list