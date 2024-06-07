from collections import defaultdict
import scipy.sparse as sp
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
import os
from utils import dump_dictionary
from featurize.base import normalize_smile, pad_or_truncate
from tqdm import tqdm


class CPI_featurize:
    def __init__(self, **config):

        radius = config['radius']
        ngram = config['ngram']
        self.root = config['root']
        self.cfg = config
        self.feat_name = 'CPI_radius-{}_ngram-{}'.format(radius, ngram)
        self.atom_dict = defaultdict(lambda: len(self.atom_dict))
        self.bond_dict = defaultdict(lambda: len(self.bond_dict))
        self.fingerprint_dict = defaultdict(lambda: len(self.fingerprint_dict))
        self.edge_dict = defaultdict(lambda: len(self.edge_dict))
        self.word_dict = defaultdict(lambda: len(self.word_dict))
        self.word_dict['<pad>'] = 0
        self.fingerprint_dict['<pad>'] = 0
        self.radius = radius
        self.ngram = ngram
        self.prot_maxlen = 1000

        self.fg_maxlen = 100

    def create_atoms(self, mol):
        """Create a list of atom (e.g., hydrogen and oxygen) IDs
        considering the aromaticity."""
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

    def split_sequence(self, sequence):
        sequence = '-' + sequence + '='
        words = [self.word_dict[sequence[i: i + self.ngram]]
                 for i in range(len(sequence) - self.ngram + 1)]
        return np.array(words)

    # def get_padding(self, input, max_len):
    #     vec = np.zeros(max_len)
    #     mask = np.zeros(max_len)
    #     for i in range(len(input)):
    #         if i < max_len:
    #             vec[i] = input[i]
    #             mask[i] = 1
    #     return vec, mask

    # def get_padding_dim2(self, input, max_len):
    #     vec = np.zeros((max_len, max_len))
    #     N = len(input)
    #     N = min(max_len, N)
    #     vec[: N, :N] = input[: N, :N]
    #     return vec
    def drug_encoding(self, smile):
        mol = Chem.AddHs(Chem.MolFromSmiles(smile))
        fingerprints = self.extract_fingerprints(mol)
        adj = self.create_adjacency(mol)
        fingerprints, fingerprints_mask = pad_or_truncate(fingerprints, max_len=self.fg_maxlen, return_mask=True)
        adj = pad_or_truncate(adj, max_len=self.fg_maxlen, pad_2d=True)
        return fingerprints, fingerprints_mask, adj
    
    def prot_encoding(self, prot):
        words = self.split_sequence(prot)
        words, w_mask = pad_or_truncate(words, max_len=self.prot_maxlen, return_mask=True)
        return words, w_mask
    
    def data_input(self, drugs, proteins, labels):

        data_list = []

        drug_dict = dict()
        prot_dict = dict()
        
        for drug in set(drugs):
            drug_dict[drug] = self.drug_encoding(drug)
        
        for prot in set(proteins):
            prot_dict[prot] = self.prot_encoding(prot)
            
        for smile, protein, label in tqdm(list(zip(drugs, proteins, labels))):

            fingerprints, fingerprints_mask, adj = drug_dict[smile]
            words, words_mask = prot_dict[protein]

            data = Data(
                x=torch.LongTensor([fingerprints]),
                adj=torch.FloatTensor([adj]),
                y=torch.FloatTensor([label]),
            )
            data.prot = torch.LongTensor([words])
            data.prot_mask = torch.FloatTensor([words_mask])
            data.x_mask = torch.FloatTensor([fingerprints_mask])
            data_list.append([data])
        self.save_file()

        return data_list

    def save_file(self):
        if not os.path.exists(os.path.join(self.root, self.feat_name)):
            os.mkdir(os.path.join(self.root, self.feat_name))
        if not os.path.exists(os.path.join(self.root, self.feat_name, 'atom_dict')):
            dump_dictionary(self.fingerprint_dict, os.path.join(self.root, self.feat_name, 'atom_dict'))
        if not os.path.exists(os.path.join(self.root, self.feat_name, 'amino_dict')):
            dump_dictionary(self.word_dict, os.path.join(self.root, self.feat_name, 'amino_dict'))
