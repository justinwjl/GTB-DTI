from collections import defaultdict
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
import os
from utils import dump_dictionary
from keras.preprocessing import text, sequence
import pandas as pd
from utils import w2v_train
from featurize.base import *


class AMMVF_featurize:
    def __init__(self, **config):
        
        radius = config['radius']
        ngram = config['ngram']
        self.root = config['root']
        self.cfg = config
        self.feat_name = 'AMMVF-{}_ngram-{}'.format(radius, ngram)
        self.atom_dict = defaultdict(lambda: len(self.atom_dict))
        self.bond_dict = defaultdict(lambda: len(self.bond_dict))
        self.edge_dict = defaultdict(lambda: len(self.edge_dict))
        self.fingerprint_dict = defaultdict(lambda: len(self.fingerprint_dict))
        self.fingerprint_dict['<pad>'] = 0
        self.radius = radius
        self.ngram = ngram
        self.prot_max_len = 1000
        self.vector_size = 100
        self.num_atom_feat = 34
        self.fg_max = 100
        self.atom_max = 100
        
    def atom_features(self, atom,explicit_H=False, use_chirality=True):
        """Generate atom features including atom symbol(10),degree(7),formal charge,
        radical electrons,hybridization(6),aromatic(1),Chirality(3)
        """
        symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']  # 10-dim
        degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
        hybridizationType = [Chem.rdchem.HybridizationType.SP,
                                Chem.rdchem.HybridizationType.SP2,
                                Chem.rdchem.HybridizationType.SP3,
                                Chem.rdchem.HybridizationType.SP3D,
                                Chem.rdchem.HybridizationType.SP3D2,
                                'other']   # 6-dim
        results = one_of_k_encoding_unk(atom.GetSymbol(),symbol) + \
                    one_of_k_encoding(atom.GetDegree(),degree) + \
                    [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                    one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]  # 10+7+2+6+1=26

        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if not explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                        [0, 1, 2, 3, 4])   # 26+5=31
        if use_chirality:
            try:
                results = results + one_of_k_encoding_unk(
                        atom.GetProp('_CIPCode'),
                        ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 31+3 =34
        return results


    def mol_features(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
        except:
            raise RuntimeError("SMILES cannot been parsed!")
        #mol = Chem.AddHs(mol)
        atom_feat = np.zeros((mol.GetNumAtoms(), self.num_atom_feat))
        for atom in mol.GetAtoms():
            atom_feat[atom.GetIdx(), :] = self.atom_features(atom)
        return atom_feat

    def create_atoms(self, mol):
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        for a in mol.GetAromaticAtoms():
            i = a.GetIdx()
            atoms[i] = (atoms[i], 'aromatic')
        atoms = [self.atom_dict[a] for a in atoms]
        return np.array(atoms)

    def create_ijbonddict(self, mol):
        """Create a dictionary, which each key is a node ID
        and each value is the tuples of its neighboring node
        and bond (e.g., single and double) IDs."""
        i_jbond_dict = defaultdict(lambda: [])
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bond = self.bond_dict[str(b.GetBondType())]
            i_jbond_dict[i].append((j, bond))
            i_jbond_dict[j].append((i, bond))
        return i_jbond_dict


    def extract_fingerprints(self, mol):
        """Extract the r-radius subgraphs (i.e., fingerprints)
        from a molecular graph using Weisfeiler-Lehman algorithm."""
        # mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # Consider hydrogens.
        atoms = self.create_atoms(mol)
        i_jbond_dict = self.create_ijbonddict(mol)

        if (len(atoms) == 1) or (self.radius == 0):
            fingerprints = [self.fingerprint_dict[a] for a in atoms]

        else:
            nodes = atoms
            i_jedge_dict = i_jbond_dict

            for _ in range(self.radius):

                """Update each node ID considering its neighboring nodes and edges
                (i.e., r-radius subgraphs or fingerprints)."""
                fingerprints = []
                for i, j_edge in i_jedge_dict.items():
                    neighbors = [(nodes[j], edge) for j, edge in j_edge]
                    fingerprint = (nodes[i], tuple(sorted(neighbors)))
                    fingerprints.append(self.fingerprint_dict[fingerprint])
                nodes = fingerprints

                """Also update each edge ID considering two nodes
                on its both sides."""
                _i_jedge_dict = defaultdict(lambda: [])
                for i, j_edge in i_jedge_dict.items():
                    for j, edge in j_edge:
                        try:
                            both_side = tuple(sorted((nodes[i], nodes[j])))
                            edge = self.edge_dict[(both_side, edge)]
                            _i_jedge_dict[i].append((j, edge))
                        except:
                            continue
                i_jedge_dict = _i_jedge_dict

        return np.array(fingerprints)
    
    def create_adjacency(self, mol):
        adjacency = Chem.GetAdjacencyMatrix(mol)
        return np.array(adjacency)

    
    def w2v_pad(self, proteins):

        #keras API 
        prot_split = [seq_to_kmers(protein, ngram=self.ngram) for protein in proteins]
        tokenizer = text.Tokenizer(num_words=10000, lower=False, filters=" ")
        tokenizer.fit_on_texts(prot_split)
        protein_ = sequence.pad_sequences(tokenizer.texts_to_sequences(prot_split), maxlen=self.prot_max_len, padding='post')
        protein_mask = np.where(protein_ != 0, 1, 0)

        word_index = tokenizer.word_index
        nb_words = len(word_index)
        print(nb_words)

        if not os.path.exists(os.path.join(self.root, self.feat_name)):
            os.mkdir(os.path.join(self.root, self.feat_name))
        saved_path = os.path.join(self.root, self.feat_name, f"word2vec_{self.ngram}_{self.vector_size}d.model")
        
        w2v_model = w2v_train(saved_path, proteins, ngram=self.ngram)
        embedding_matrix = np.zeros((nb_words + 1, self.vector_size))
        for word, i in word_index.items():
            embedding_glove_vector=w2v_model.wv[word] if word in w2v_model.wv.index2word else None
            if embedding_glove_vector is not None:
                embedding_matrix[i] = embedding_glove_vector
            else:
                unk_vec = np.random.random(self.vector_size) * 0.5
                unk_vec = unk_vec - unk_vec.mean()
                embedding_matrix[i] = unk_vec

        pd_embedding_matrix = pd.DataFrame(embedding_matrix)
        pd_embedding_matrix.to_csv(os.path.join(self.root, self.feat_name, f"embedding_{self.ngram}_{self.vector_size}d.csv"), index=False)
        
        return protein_, protein_mask

    
    def data_input(self, drugs, proteins, labels):
        
        data_list = []
        proteins_, proteins_mask = self.w2v_pad(proteins)

        drug_dict = dict()
        
        for drug in set(drugs):
            mol = Chem.AddHs(Chem.MolFromSmiles(drug))
            atom_feature = self.mol_features(drug)
            fingerprints = self.extract_fingerprints(mol)
            adj = self.create_adjacency(mol)
            drug_dict[drug] = (atom_feature, fingerprints, adj)
            
        for i, (smile, protein, label) in enumerate(list(zip(drugs, proteins, labels))):
            atom_feature, fingerprints, adj = drug_dict[smile]
            fingerprints, fingerprints_mask = pad_or_truncate(fingerprints, self.fg_max, return_mask=True)
            atom_feature, atom_feature_mask = pad_or_truncate(atom_feature, max_len=self.atom_max, return_mask=True)
            adj = pad_or_truncate(adj, self.fg_max, pad_2d=True)
            words = proteins_[i, :] 
            
            data = Data(
                x=torch.LongTensor([fingerprints]),
                x_atom=torch.LongTensor([atom_feature]),
                x_atom_mask=torch.FloatTensor([atom_feature_mask]),
                adj=torch.FloatTensor([adj]),
                y=torch.FloatTensor([label]),
            )
            data.prot=torch.LongTensor([words])
            data.x_mask=torch.FloatTensor([fingerprints_mask])
            data.prot_mask = torch.FloatTensor([proteins_mask[i, :]])
            data_list.append([data])
        self.save_file()
        
        return data_list

    def save_file(self):
        if not os.path.exists(os.path.join(self.root, self.feat_name)):
            os.mkdir(os.path.join(self.root, self.feat_name))
        if not os.path.exists(os.path.join(self.root, self.feat_name, 'atom_dict')):
            dump_dictionary(self.fingerprint_dict, os.path.join(self.root, self.feat_name, 'atom_dict'))