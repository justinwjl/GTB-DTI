import torch
from torch_geometric.data import Data
from featurize.base import str2int

class MRBDTA_featurize:
    def __init__(self, max_prot_len=1000, max_drug_len=100):
        self.feat_name = 'MRBDTA'
        self.seq_dict = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                         "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                         "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                         "U": 19, "T": 20, "W": 21,
                         "V": 22, "Y": 23, "X": 24,
                         "Z": 25}
        self.drugseq_dict = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6,
			      ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12,
			      "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18,
			      "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
			      "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30,
			      "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36,
			      "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42,
			      "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48,
			      "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54,
			      "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
			      "t": 61, "y": 62, "@": 63, '/': 64, '\\': 65}

        self.max_prot_len = max_prot_len
        self.max_drug_len = max_drug_len

    def drug_encoding(self, smile):
        return str2int(smile, self.drugseq_dict, self.max_drug_len)

    def protein_encoding(self, prot):
        return str2int(prot, self.seq_dict, self.max_prot_len)

    def data_input(self, drugs, proteins, labels):

        data_list = []
        
        drug_dict = dict()
        prot_dict = dict()
        
        for drug in set(drugs):
            drug_dict[drug] = self.drug_encoding(drug)
        
        for prot in set(proteins):
            prot_dict[prot] = self.protein_encoding(prot)
            
        for smile, protein, label in list(zip(drugs, proteins, labels)):
            drug_input = drug_dict[smile]
            data = Data(
                x=torch.LongTensor([drug_input]),
                y=torch.FloatTensor([label]),
            )
            protein_input = prot_dict[protein]
            data.target = torch.LongTensor([protein_input])
            data_list.append([data])

        return data_list
