import os
import random
import networkx as nx
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDConfig
from torch_geometric.data import Data
from featurize.base import str2int, one_of_k_encoding, one_of_k_encoding_unk

class coldDTA_featurize:
    def __init__(self, max_prot_len=1200):
        self.feat_name = 'coldDTA'
        self.seq_dict = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                         "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                         "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                         "U": 19, "T": 20, "W": 21,
                         "V": 22, "Y": 23, "X": 24,
                         "Z": 25}

        self.max_prot_len = max_prot_len
        fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        self.chem_feature_factory = Chem.ChemicalFeatures.BuildFeatureFactory(fdef_name)

    def atom_features(self, atom):
        encoding = one_of_k_encoding_unk(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])
        encoding += one_of_k_encoding(atom.GetDegree(), [0,1,2,3,4,5,6,7,8,9,10]) + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0,1,2,3,4,5,6,7,8,9,10]) 
        encoding += one_of_k_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5,6,7,8,9,10]) 
        encoding += one_of_k_encoding_unk(atom.GetHybridization(), [
                        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                        Chem.rdchem.HybridizationType.SP3D2, 'other']) 
        encoding += [atom.GetIsAromatic()]

        try:
            encoding += one_of_k_encoding_unk(
                        atom.GetProp('_CIPCode'),
                        ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            encoding += [0, 0] + [atom.HasProp('_ChiralityPossible')]
        
        return np.array(encoding)


    def remove_subgraph(self, Graph, center, percent):
        assert percent <= 1
        G = Graph.copy()
        num = int(np.floor(len(G.nodes())*percent))
        removed = []
        temp = [center]
        while len(removed) < num and temp:
            neighbors = []

            try:
                for n in temp:
                    neighbors.extend([i for i in G.neighbors(n) if i not in temp])      
            except Exception as e:
                print(e)
                return None, None

            for n in temp:
                if len(removed) < num:
                    G.remove_node(n)
                    removed.append(n)
                else:
                    break

            temp = list(set(neighbors))

        return G, removed

    # data augmentation
    def mol_to_graph(self, mol, times=2):
        start_list = random.sample(list(range(mol.GetNumAtoms())), times)

        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

        molGraph = nx.Graph(edges)
        percent = 0.2
        removed_list = []
        for i in range(1, times + 1):
            G, removed = self.remove_subgraph(molGraph, start_list[i - 1], percent)
            removed_list.append(removed)

        for removed_i in removed_list:
            if not removed_i:
                return None, None, None

        features_list = []
        for i in range(times):
            features_list.append([])

        for index, atom in enumerate(mol.GetAtoms()):
            for i, removed_i in enumerate(removed_list):
                if index not in removed_i:
                    feature = self.atom_features(atom)
                    features_list[i].append(feature/np.sum(feature))

        edges_list = []
        for i in range(times):
            edges_list.append([])

        g = nx.DiGraph()
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                e_ij = mol.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    g.add_edge(i, j,
                                b_type=e_ij.GetBondType(),
                                IsConjugated=int(e_ij.GetIsConjugated()),
                                )
        
        edge_attr_list = []
        for i in range(times):
            edge_attr_list.append([])
        for i, removed_i in enumerate(removed_list):
            e = {}
            for n1, n2, d in g.edges(data=True):
            
                if n1 not in removed_i and n2 not in removed_i:
                    start_i = n1 - sum(num < n1 for num in removed_i)
                    e_i = n2 - sum(num < n2 for num in removed_i)
                    edges_list[i].append([start_i, e_i])
                    e_t = [int(d['b_type'] == x)
                            for x in (Chem.rdchem.BondType.SINGLE, \
                                        Chem.rdchem.BondType.DOUBLE, \
                                        Chem.rdchem.BondType.TRIPLE, \
                                        Chem.rdchem.BondType.AROMATIC)]
                    e_t.append(int(d['IsConjugated'] == False))
                    e_t.append(int(d['IsConjugated'] == True))
                    e[(n1, n2)] = e_t
            edge_attr = list(e.values())
            edge_attr_list[i] = edge_attr

        if len(e) == 0:
            return features_list, torch.LongTensor([[0], [0]]), torch.FloatTensor([[0, 0, 0, 0, 0, 0]])

        edge_index_list = []
        for i in range(times):
            edge_index_list.append([])
        for i, edges_i in enumerate(edges_list):
            g_i = nx.Graph(edges_i).to_directed()
            for e1, e2 in g_i.edges():
                edge_index_list[i].append([e1, e2])

        return features_list, edge_index_list, edge_attr_list


    def mol_to_graph_without_rm(self, mol):
        features = []
        for atom in mol.GetAtoms():
            feature = self.atom_features(atom)
            features.append(feature/np.sum(feature))

        g = nx.DiGraph()
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                e_ij = mol.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    g.add_edge(i, j,
                                b_type=e_ij.GetBondType(),
                                IsConjugated=int(e_ij.GetIsConjugated()),
                                )
        e = {}
        for n1, n2, d in g.edges(data=True):
            e_t = [int(d['b_type'] == x)
                    for x in (Chem.rdchem.BondType.SINGLE, \
                                Chem.rdchem.BondType.DOUBLE, \
                                Chem.rdchem.BondType.TRIPLE, \
                                Chem.rdchem.BondType.AROMATIC)]
            e_t.append(int(d['IsConjugated'] == False))
            e_t.append(int(d['IsConjugated'] == True))
            e[(n1, n2)] = e_t

        if len(e) == 0:
            return features, torch.LongTensor([[0], [0]]), torch.FloatTensor([[0, 0, 0, 0, 0, 0]])

        edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
        edge_attr = torch.FloatTensor(list(e.values()))

        return features, edge_index, edge_attr
    
    def protein_encoding(self, prot):
        return str2int(prot, self.seq_dict, self.max_prot_len)

    def data_input(self, drugs, proteins, labels):
        
        data_list = []
        
        drug_dict = dict()
        prot_dict = dict()
        
        for drug in set(drugs):
            mol = Chem.MolFromSmiles(drug)
            drug_dict[drug] = self.mol_to_graph_without_rm(mol)
            
        for prot in set(proteins):
            prot_dict[prot] = self.protein_encoding(prot)
            
        for smile, protein, label in list(zip(drugs, proteins, labels)):
            x, edge_index, edge_attr = drug_dict[smile]
            prot_input = prot_dict[protein]
            data = Data(
                x=torch.FloatTensor(x),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.FloatTensor([label]),
                target=torch.LongTensor([prot_input])
            )
            data_list.append([data])
            
        return data_list