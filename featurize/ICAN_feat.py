import numpy as np
from featurize.base import *
import torch
import pandas as pd
import codecs
from subword_nmt.apply_bpe import BPE
from torch_geometric.data import Data
import selfies as sf

class ICAN_featurize:
    def __init__(self, **config):
        self.feat_name = 'ICAN'
        vocab_path = 'featurize/MolTrans_encoding/protein_codes_uniprot.txt'
        bpe_codes_protein = codecs.open(vocab_path)
        self.pbpe = BPE(bpe_codes_protein, merges=-1, separator='')

        sub_csv = pd.read_csv('featurize/MolTrans_encoding/subword_units_map_uniprot.csv')
        idx2word_p = sub_csv['index'].values
        self.words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

        vocab_path = 'featurize/MolTrans_encoding/drug_codes_chembl.txt'
        bpe_codes_drug = codecs.open(vocab_path)
        self.dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

        sub_csv = pd.read_csv('featurize/MolTrans_encoding/subword_units_map_chembl.csv')
        idx2word_d = sub_csv['index'].values
        self.words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

        self.max_prot_len = config['max_prot_len']
        self.max_smile_len = config['max_smile_len']
        self.encode_method = config['encode_method']

    def drug_encoding(self, smile):
        if self.encode_method == 'smiles':
            t1 = smile
        elif self.encode_method == 'selfies':
            t1 = sf.encoder(smile).replace("][", "],[").split(',') 
        else:
            t1 = self.dbpe.process_line(smile).split()  # split
            
        try:
            i1 = np.asarray([self.words2idx_d[i] for i in t1])  # index
        except:
            i1 = np.array([0])

        i1_l = len(i1)

        if i1_l < self.max_smile_len:
            i = np.pad(i1, (0, self.max_smile_len - i1_l), 'constant', constant_values=0)
            input_mask = ([1] * i1_l) + ([0] * (self.max_smile_len - i1_l))

        else:
            i = i1[:self.max_smile_len]
            input_mask = [1] * self.max_smile_len

        return i, np.asarray(input_mask)

    def protein_encoding(self, prot):
        if self.encode_method == 'smiles':
            t1 = prot
        elif self.encode_method == 'selfies':
            t1 = prot
        else:            
            t1 = self.pbpe.process_line(prot).split()  # split
            
        try:
            i1 = np.asarray([self.words2idx_p[i] for i in t1])  # index
        except:
            i1 = np.array([0])

        i1_l = len(i1)

        if i1_l < self.max_prot_len:
            i = np.pad(i1, (0, self.max_prot_len - i1_l), 'constant', constant_values=0)
            input_mask = ([1] * i1_l) + ([0] * (self.max_prot_len - i1_l))
        else:
            i = i1[:self.max_prot_len]
            input_mask = [1] * self.max_prot_len

        return i, np.asarray(input_mask)

    def data_input(self, drugs, proteins, labels):
        
        data_list = []
        
        drug_dict = dict()
        prot_dict = dict()
        
        for drug in set(drugs):
            drug_dict[drug] = self.drug_encoding(drug)
            
        for prot in set(proteins):
            prot_dict[prot] = self.protein_encoding(prot)
        
        for smile, protein, label in list(zip(drugs, proteins, labels)):
            d_v, d_mask = drug_dict[smile]
            data = Data(
                d_v=torch.LongTensor([d_v]),
                y=torch.FloatTensor([label]),
            )
            p_v, p_mask = prot_dict[protein]
            data.p_v = torch.LongTensor([p_v])
            data.d_mask = torch.LongTensor([d_mask])
            data.p_mask = torch.LongTensor([p_mask])
            data_list.append([data])

        return data_list
