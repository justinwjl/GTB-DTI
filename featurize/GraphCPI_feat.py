import numpy as np
from rdkit import Chem
import networkx as nx
from featurize.base import *
import torch
from torch_geometric.data import Data
from gensim.models import Word2Vec
import pandas as pd
import os
from utils import w2v_train
from keras.preprocessing import text, sequence
from rdkit import RDConfig
class GraphCPI_featurize:
    def __init__(self, max_prot_len=1000, **config):
        self.feat_name = 'GraphCPI'
        self.atom_map = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                         'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu',
                         'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
        self.degree_map = list(range(0, 11))
        self.numH_map = list(range(0, 11))
        self.implicit_map = list(range(0, 11))
        
        self.root = config['root']
        self.max_prot_len = max_prot_len
        self.vector_size = 100
        self.ngram = config['ngram']
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

    def split_ngrams(self, seq):
        """
        'AGAMQSASM' => [['AGA', 'MQS', 'ASM'], ['GAM','QSA'], ['AMQ', 'SAS']]
        """
        a, b, c = zip(*[iter(seq)]*3), zip(*[iter(seq[1:])]*3), zip(*[iter(seq[2:])]*3)
        str_ngrams = ""
        for ngrams in [a,b,c]:
            x = ""
            for ngram in ngrams:
                x +="".join(ngram) + " "
            str_ngrams += x + " "
        return str_ngrams

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
        protein_ = sequence.pad_sequences(tokenizer.texts_to_sequences(prot_split), maxlen=self.max_prot_len, padding='post')

        word_index = tokenizer.word_index
        nb_words = len(word_index)
        print(nb_words)
        name = 'GraphCPI'
        # if not os.path.exists(os.path.join(self.root, self.feat_name)):
        #     os.mkdir(os.path.join(self.root, self.feat_name))
        # saved_path = os.path.join(self.root, self.feat_name, f"word2vec_{self.ngram}_{self.vector_size}d.model")
        if not os.path.exists(os.path.join(self.root, name)):
            os.mkdir(os.path.join(self.root, name))
        saved_path = os.path.join(self.root, name, f"word2vec_{self.ngram}_{self.vector_size}d.model")
        
        
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
        pd_embedding_matrix.to_csv(os.path.join(self.root, name, f"embedding_{self.ngram}_{self.vector_size}d.csv"), index=False)
        
        return protein_

    def protein_encoding(self, prot, dict):
        split_prot = self.split_ngrams(prot).strip().split(' ')
        split_prot = [word for word in split_prot if len(word) == 3]
        x = np.ones(self.max_prot_len) * int(dict['<pad>'])
        for i, ch in enumerate(split_prot[: self.max_prot_len]):
            x[i] = int(dict[ch])
        return x

    def data_input(self, drugs, proteins, labels):
        
        proteins_ = self.w2v_pad(proteins)
        
        data_list = []

        drug_dict = dict()
        
        for drug in set(drugs):
            c_size, atom_feature, adj = self.drug_encoding(drug)
            drug_dict[drug] = (atom_feature, adj)

        for i, (smile, protein, label) in enumerate(list(zip(drugs, proteins, labels))):
            features, edge_index = drug_dict[smile]
            if len(edge_index) == 0:
                edge_index = torch.empty((0, 2), dtype=torch.long)
            data = Data(
                x=torch.Tensor(features),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                y=torch.FloatTensor([label])
            )
            protein_input = proteins_[i, :]
            data.target = torch.LongTensor([protein_input])
            data_list.append([data])
        return data_list

