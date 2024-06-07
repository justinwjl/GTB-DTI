from featurize.base import *
import torch
from torch_geometric.data import Data


class Hyperattention_featurize:
    def __init__(self, max_smile_len=100, max_prot_len=1000):
        self.feat_name = 'hyperattention'
        self.smile_dict = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                           "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                           "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                           "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                           "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                           "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                           "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                           "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
        self.max_smile_len = max_smile_len

        self.seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
        self.seq_dict = {v: (i + 1) for i, v in enumerate(self.seq_voc)}
        self.max_prot_len = max_prot_len

    def drug_encoding(self, smile):
        return str2int(smile, self.smile_dict, self.max_smile_len)

    def protein_encoding(self, prot):
        return str2int(prot, self.seq_dict, self.max_prot_len)

    def data_input(self, drugs, proteins, labels):
        
        data_list = []
        for smile, protein, label in list(zip(drugs, proteins, labels)):
            
            drug_input = self.drug_encoding(smile)
            data = Data(
                x=torch.LongTensor([drug_input]),
                y=torch.FloatTensor([label]),
            )
            protein_input = self.protein_encoding(protein)
            data.target = torch.LongTensor([protein_input])

            data_list.append([data])
        return data_list
