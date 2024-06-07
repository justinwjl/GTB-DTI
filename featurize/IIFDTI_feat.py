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
from featurize.base import one_of_k_encoding, one_of_k_encoding_unk, str2int, pad_or_truncate

class IIFDTI_featurize:
    def __init__(self, **config):
        
        ngram = config['ngram']
        self.root = config['root']
        self.cfg = config
        self.feat_name = 'IIFDTI_ngram-{}'.format(ngram)
        self.ngram = ngram
        self.prot_max_len = 1000
        self.max_prot_len = 1000
        self.vector_size = 100
        self.smile_max_len = 200
        self.num_atom_feat = 34
        self.seq_dict = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                         "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                         "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                         "U": 19, "T": 20, "W": 21,
                         "V": 22, "Y": 23, "X": 24,
                         "Z": 25}
        self.adj_max = 100
        
    def atom_features(self, atom, explicit_H=False, use_chirality=True):
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


    def adjacent_matrix(self, mol):
        adjacency = Chem.GetAdjacencyMatrix(mol)
        return np.array(adjacency)+np.eye(adjacency.shape[0])

    def mol_features(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
        except:
            raise RuntimeError("SMILES cannot been parsed!")
        atom_feat = np.zeros((mol.GetNumAtoms(), self.num_atom_feat))
        for atom in mol.GetAtoms():
            atom_feat[atom.GetIdx(), :] = self.atom_features(atom)
        adj_matrix = self.adjacent_matrix(mol)
        return atom_feat, adj_matrix

    def seq_to_kmers(self, seq, ngram):
        """ Divide a string into a list of kmers strings.

        Parameters:
            seq (string)
        Returns:
            List containing a list of kmers.
        """
        N = len(seq)
        return [seq[i: i+ngram] for i in range(N - ngram + 1)]
    
    def w2v_pad(self, proteins):

        #keras API 
        prot_split = [self.seq_to_kmers(protein, ngram=self.ngram) for protein in proteins]
        tokenizer = text.Tokenizer(num_words=10000, lower=False, filters=" ")
        tokenizer.fit_on_texts(prot_split)
        protein_ = sequence.pad_sequences(tokenizer.texts_to_sequences(prot_split), maxlen=self.prot_max_len, padding='post')

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
        
        return protein_

    def protein_encoding(self, prot):
        return str2int(prot, self.seq_dict, self.max_prot_len)
    
    def smile_w2v_pad(self, smiles):

        #keras API
        smile_split = [self.seq_to_kmers(smile, ngram=1) for smile in smiles]
        tokenizer = text.Tokenizer(num_words=100, lower=False, filters=" ")
        tokenizer.fit_on_texts(smile_split)
        smile_ = sequence.pad_sequences(tokenizer.texts_to_sequences(smile_split), maxlen=self.smile_max_len)

        word_index = tokenizer.word_index
        nb_words = len(word_index)
        print(nb_words)
        
        if not os.path.exists(os.path.join(self.root, self.feat_name)):
            os.mkdir(os.path.join(self.root, self.feat_name))
        saved_path = os.path.join(self.root, self.feat_name, f"smile2vec_{1}_{self.vector_size}d.model")
        
        w2v_model = w2v_train(saved_path, smiles, ngram=1)
        
        embedding_matrix = np.zeros((nb_words + 1, self.vector_size))
        for word, i in word_index.items():
            embedding_glove_vector=w2v_model.wv[word] if word in w2v_model.wv.index2word else None
            if embedding_glove_vector is not None:
                embedding_matrix[i] = embedding_glove_vector
            else:
                unk_vec = np.random.random(self.vector_size) * 0.5
                unk_vec = unk_vec - unk_vec.mean()
                embedding_matrix[i] = unk_vec

        print(embedding_matrix.shape)
        pd_embedding_matrix = pd.DataFrame(embedding_matrix)
        pd_embedding_matrix.to_csv(os.path.join(self.root, self.feat_name, f"smile_embedding_{1}_{self.vector_size}d.csv"), index=False)
        return smile_


    def padding_atom(self, atom_feature, max_len):

        pad_row = max_len - len(atom_feature)
        mask = np.ones(max_len)
        mask[max_len-pad_row:] = 0.0
        atom_feature_padding = np.pad(atom_feature, ((0, pad_row), (0, 0)), 
                  mode='constant', constant_values=0)
        return atom_feature_padding, mask
    
    def padding_dim2(self, input, max_len):

        pad_len = max_len - len(input)
        return np.pad(input, ((0, pad_len), (0, pad_len)), mode='constant', constant_values=0)
    
    
    def data_input(self, drugs, proteins, labels):
        
        data_list = []
        proteins_ = self.w2v_pad(proteins)
        smiles_ = self.smile_w2v_pad(drugs)
        
        smile_set = set(drugs)
        smile_dict = dict()
        smile_edge = dict()
        max_N = -1
        for smile in smile_set:
            atom_feature, adj = self.mol_features(smile)
            smile_edge[smile] = adj
            smile_dict[smile] = atom_feature
            if len(atom_feature) > max_N:
                max_N = len(atom_feature)
        
        for i, (smile, protein, label) in enumerate(list(zip(drugs, proteins, labels))):

            atom_feature = smile_dict[smile]
            atom_feature, atom_mask = pad_or_truncate(atom_feature, max_len=self.adj_max, 
                                                      pad_2d=False, return_mask=True)
            adj = smile_edge[smile]
            # adj = self.padding_dim2(adj, max_N)
            adj = pad_or_truncate(adj, max_len=self.adj_max, pad_2d=True)
            
            data = Data(
                x=torch.LongTensor([atom_feature]),
                adj = torch.LongTensor([adj]),
                y=torch.FloatTensor([label]),
            )
            prot_ids = self.protein_encoding(protein)
            data.prot_ngram = torch.LongTensor([proteins_[i, :]])
            data.prot_ids = torch.LongTensor([prot_ids])
            data.smile = torch.LongTensor([smiles_[i, :]])
            data.x_mask=torch.FloatTensor([atom_mask])
            data_list.append([data])
        
        return data_list
