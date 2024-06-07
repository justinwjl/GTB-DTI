import numpy as np
from rdkit import Chem
import torch
from featurize.base import *
from keras.preprocessing import text, sequence
from utils import w2v_train
import os
from torch_geometric.data import Data
from featurize.base import pad_or_truncate
from tqdm import tqdm
class TransCPI_featurize:
    def __init__(self, **config):
        
        self.num_atom_feat = 34
        self.ngram = config['ngram']
        self.root = config['root']
        self.feat_name = 'TransCPI_ngram-{}'.format(self.ngram)
        self.prot_max = 1000
        self.vector_size = 100
        self.adj_max = 100
        self.ngram = 3
        
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


    def adjacent_matrix(self, mol):
        adjacency = Chem.GetAdjacencyMatrix(mol)
        return np.array(adjacency)+np.eye(adjacency.shape[0])


    def mol_features(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
        except:
            raise RuntimeError("SMILES cannot been parsed!")
        #mol = Chem.AddHs(mol)
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
        protein_ = sequence.pad_sequences(tokenizer.texts_to_sequences(prot_split), maxlen=self.prot_max, padding='post')
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
        
        proteins_, proteins_mask = self.w2v_pad(proteins)
        
        drug_dict = dict()
        for drug in set(drugs):
            drug_dict[drug] = self.mol_features(drug)
            
        data_list = []

        for i, (smile, protein, label) in enumerate(tqdm(list(zip(drugs, proteins, labels)))):
            atom_feature, adj = drug_dict[smile]
            atom_feature, atom_mask = pad_or_truncate(atom_feature, max_len=self.adj_max, 
                                                      pad_2d=False, return_mask=True)
            adj = pad_or_truncate(adj, self.adj_max, pad_2d=True)
            data = Data(
                x=torch.FloatTensor([atom_feature]),
                adj=torch.FloatTensor([adj]),
                y=torch.FloatTensor([label]),
            )
            data.prot = torch.LongTensor([proteins_[i, :]])
            data.prot_mask = torch.FloatTensor([proteins_mask[i, :]])
            data.comp_mask = torch.FloatTensor([atom_mask])
            data_list.append([data])
        return data_list