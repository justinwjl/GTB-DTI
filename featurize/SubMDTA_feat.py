from torch_geometric.data.data import Data
from rdkit import Chem
import networkx as nx
from utils import *
from collections import defaultdict
from featurize.base import one_of_k_encoding, one_of_k_encoding_unk, pad_or_truncate

class SubMDTA_featurize:
    def __init__(self, **config):
        self.feat_name = "SubMDTA"
        self.seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
        self.seq_dict = {v: (i + 1) for i, v in enumerate(self.seq_voc)}
        self.seq_dict_len = len(self.seq_dict)
        self.max_seq_len = config["max_seq_len"] if "max_seq_len" in config else 1000
        self.word_dict_2 = defaultdict(lambda: len(self.word_dict_2))
        self.word_dict_3 = defaultdict(lambda: len(self.word_dict_3))
        self.word_dict_4 = defaultdict(lambda: len(self.word_dict_4))
        self.word_dict_2['<pad>'] = 0
        self.word_dict_3['<pad>'] = 0
        self.word_dict_4['<pad>'] = 0
        self.root = config['root']

    def atom_features(self, atom):
        return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                            ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                            'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                            'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                            'Pt', 'Hg', 'Pb', 'Unknown']) +
                        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
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
        for e1, e2 in g.edges:
            edge_index.append([e1, e2])

        return c_size, np.array(features), np.array(edge_index)
    

    def seq_cat(self, prot):
        x = np.zeros(self.max_seq_len)
        for i, ch in enumerate(prot[:self.max_seq_len]):
            x[i] = self.seq_dict[ch]
        return x

    def split_sequence(self, sequence, ngram):
        sequence = '-' + sequence + '='

        if ngram == 2:
            words = [self.word_dict_2[sequence[i:i + ngram]]
                     for i in range(len(sequence) - ngram + 1)]
        elif ngram == 3:
            words = [self.word_dict_3[sequence[i:i + ngram]]
                     for i in range(len(sequence) - ngram + 1)]
        else:
            words = [self.word_dict_4[sequence[i:i + ngram]]
                     for i in range(len(sequence) - ngram + 1)]
        # return np.array(words)
        return words


    def prot_encoding(self, prot, ngram):
        prot_ngram = self.split_sequence(prot, ngram=ngram)
        prot_encode = pad_or_truncate(np.array(prot_ngram), max_len=self.max_seq_len)
        return prot_encode

    def data_input(self, drugs, proteins, labels):
        # Drug
        data_list = []
        
        drug_dict = dict()
        prot_dict = dict()
        
        for drug in set(drugs):
            drug_dict[drug] = self.smile_to_graph(drug)
            
        for prot in set(proteins):
            prot_dict[prot] = (self.prot_encoding(prot, 2), self.prot_encoding(prot, 3), self.prot_encoding(prot, 4))
            
        for smile, protein, label in list(zip(drugs, proteins, labels)):
            c_size, features, edge_index = drug_dict[smile]
            target_2, target_3, target_4 = prot_dict[protein]
            if len(edge_index) == 0:
                edge_index = torch.empty((0, 2), dtype=torch.long)
            data = Data(x=torch.Tensor(features),
                        edge_index=torch.LongTensor(edge_index).transpose(1, 0),  # 行列转置
                        y=torch.FloatTensor([label]))
            data.target_2 = torch.LongTensor([target_2])
            data.target_3 = torch.LongTensor([target_3])
            data.target_4 = torch.LongTensor([target_4])
            
            data_list.append([data])
        self.save_file()
        return data_list
    
    def save_file(self):
        if not os.path.exists(os.path.join(self.root, self.feat_name)):
            os.mkdir(os.path.join(self.root, self.feat_name))
        if not os.path.exists(os.path.join(self.root, self.feat_name, 'word_dict2')):
            dump_dictionary(self.word_dict_2, os.path.join(self.root, self.feat_name, 'word_dict2'))
        if not os.path.exists(os.path.join(self.root, self.feat_name, 'word_dict3')):
            dump_dictionary(self.word_dict_3, os.path.join(self.root, self.feat_name, 'word_dict3'))
        if not os.path.exists(os.path.join(self.root, self.feat_name, 'word_dict4')):
            dump_dictionary(self.word_dict_4, os.path.join(self.root, self.feat_name, 'word_dict4'))
