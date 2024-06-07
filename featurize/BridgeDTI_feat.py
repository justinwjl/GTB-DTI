import numpy as np
from tqdm import tqdm
import os, logging, pickle, random, torch, gc, deepchem
from torch_geometric.data.data import Data
from deepchem.feat import graph_features
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn.feature_extraction.text import CountVectorizer
from featurize.base import one_of_k_encoding, one_of_k_encoding_unk


class BridgeDTI_featurize:
    def __init__(self, **config):
        self.feat_name = 'BridgeDTI'
        self.atNum = None
        self.amNum = None
        self.p2id, self.id2p = {}, []
        self.d2id, self.id2d = {}, []
        self.am2id, self.id2am = {"<UNK>": 0, "<EOS>": 1}, ["<UNK>", "<EOS>"]
        self.at2id, self.id2at = {"<UNK>": 0, "<EOS>": 1}, ["<UNK>", "<EOS>"]

        self.pSeqMaxLen = config['pSeqMaxLen'] if 'pSeqMaxLen' in config else 1024
        self.dSeqMaxLen = config['dSeqMaxLen'] if 'dSeqMaxLen' in config else 128
        self.cSize = config['cSize'] if 'cSize' in config else 8450  # 8422
        # pSeqMaxLen = 1024, dSeqMaxLen = 128, kmers = -1, validSize = 0.2, sep = ' '

    def atom_features(self, atom, bool_id_feat=False, explicit_H=False, use_chirality=False):

        results = one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd',
                'Co', 'Se', 'Ti', 'Zn',
                'H',  # H?
                'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
                'Unknown'
            ]) + one_of_k_encoding(atom.GetDegree(),
                                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
                  one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                  ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if not explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False
                                     ] + [atom.HasProp('_ChiralityPossible')]

        return np.array(results)

    def data_input(self, drugs, proteins, labels):
        pCnt, dCnt = 0, 0
        pSeqData, dMolData, dSeqData, dFeaData, dFinData = [], [], [], [], []
        eSeqData = []

        for drug, protein, lab in list(zip(drugs, proteins, labels)):
            if protein not in self.p2id:
                pSeqData.append(protein)
                self.p2id[protein] = pCnt
                self.id2p.append(protein)
                pCnt += 1

            if drug not in self.d2id:
                mol = Chem.MolFromSmiles(drug)
                if mol is None:
                    continue
                dSeqData.append([a.GetSymbol() for a in mol.GetAtoms()])
                dMolData.append(mol)
                dFeaData.append([self.atom_features(a) for a in mol.GetAtoms()])
                tmp = np.ones((1,))
                DataStructs.ConvertToNumpyArray(AllChem.GetHashedMorganFingerprint(mol, 2, nBits=1024), tmp)
                dFinData.append(tmp)
                self.d2id[drug] = dCnt
                self.id2d.append(drug)
                dCnt += 1

            eSeqData.append([self.p2id[protein], self.d2id[drug], lab])

        amCnt = 2
        for pSeq in tqdm(pSeqData):
            for am in pSeq:
                if am not in self.am2id:
                    self.am2id[am] = amCnt
                    self.id2am.append(am)
                    amCnt += 1
        self.amNum = amCnt

        atCnt = 2
        for dSeq in tqdm(dSeqData):
            for at in dSeq:
                if at not in self.at2id:
                    self.at2id[at] = atCnt
                    self.id2at.append(at)
                    atCnt += 1
        self.atNum = atCnt

        # Tokenized protein data
        pSeqTokenized = []
        # pSeqLen = []
        for pSeq in tqdm(pSeqData):
            pSeq = [self.am2id[am] for am in pSeq]
            # pSeqLen.append(min(len(pSeq), self.pSeqMaxLen))
            pSeqTokenized.append(pSeq[:self.pSeqMaxLen] + [1] * max(self.pSeqMaxLen - len(pSeq), 0))

        pSeqTokenized = np.array(pSeqTokenized, dtype=np.int32)

        ctr = CountVectorizer(ngram_range=(1, 3), analyzer='char')
        pContFeat = ctr.fit_transform([''.join(i) for i in pSeqData]).toarray().astype('float32')
        # k1, k2, k3 = [len(i) == 1 for i in ctr.get_feature_names()], [len(i) == 2 for i in ctr.get_feature_names()], [
        #     len(i) == 3 for i in ctr.get_feature_names()]
        k1, k2, k3 = ([len(i) == 1 for i in ctr.get_feature_names_out()],
                      [len(i) == 2 for i in ctr.get_feature_names_out()],
                      [len(i) == 3 for i in ctr.get_feature_names_out()])

        pContFeat[:, k1] = (pContFeat[:, k1] - pContFeat[:, k1].mean(axis=1).reshape(-1, 1)) / (
                pContFeat[:, k1].std(axis=1).reshape(-1, 1) + 1e-8)
        pContFeat[:, k2] = (pContFeat[:, k2] - pContFeat[:, k2].mean(axis=1).reshape(-1, 1)) / (
                pContFeat[:, k2].std(axis=1).reshape(-1, 1) + 1e-8)
        pContFeat[:, k3] = (pContFeat[:, k3] - pContFeat[:, k3].mean(axis=1).reshape(-1, 1)) / (
                pContFeat[:, k3].std(axis=1).reshape(-1, 1) + 1e-8)
        mean, std = pContFeat.mean(axis=0), pContFeat.std(axis=0) + 1e-8
        pContFeat = (pContFeat - mean) / std
        pContFeat = self.pad_or_truncate_second_dim(pContFeat, self.cSize)

        dGraphFeat = np.array([i[:self.dSeqMaxLen] + [[0] * 75] * (self.dSeqMaxLen - len(i)) for i in dFeaData],
                              dtype=np.int8)
        dFinprFeat = np.array(dFinData, dtype=np.float32)
        # eSeqData = np.array(eSeqData, dtype=np.int32)
        eSeqData = np.array(eSeqData, dtype=np.float32)

        data_list = []
        # for smile, protein, label in eSeqData:
        for index in eSeqData:
            # eSeqData.append([self.p2id[protein], self.d2id[drug], lab])
            dTokenizedName = (index[1]).astype(np.int32)
            pTokenizedName = (index[0]).astype(np.int32)
            label = index[2]

            data = Data(
                y=torch.FloatTensor([label])
            )
            # cSize = pContFeat.shape[1]
            # "aminoSeq": torch.tensor(self.pSeqTokenized[pTokenizedNames], dtype=torch.long).to(device), \
            # "aminoCtr": torch.tensor(self.pContFeat[pTokenizedNames], dtype=torch.float32).to(device), \
            # "atomFea": torch.tensor(self.dGraphFeat[dTokenizedNames], dtype=torch.float32).to(device), \
            # "atomFin": torch.tensor(self.dFinprFeat[dTokenizedNames], dtype=torch.float32).to(device), \
            data.aminoSeq = torch.LongTensor([pSeqTokenized[pTokenizedName]])  # , dtype=torch.long)
            data.aminoCtr = torch.FloatTensor([pContFeat[pTokenizedName]])  # , dtype=torch.float32)
            data.atomFea = torch.FloatTensor([dGraphFeat[dTokenizedName]])  # , dtype=torch.float32)
            data.atomFin = torch.FloatTensor([dFinprFeat[dTokenizedName]])  # , dtype=torch.float32)
            data_list.append([data])
        return data_list  # , 'cSize', cSize

    @staticmethod
    def padding_second_dim(input, max_len):
        pad_len = max_len - input.shape[1]
        return np.pad(input, ((0, 0), (0, pad_len)), mode='constant', constant_values=0)

    @staticmethod
    def pad_or_truncate_second_dim(input_tensor, target_len):
        if input_tensor.shape[1] > target_len:
            return input_tensor[:, :target_len]
        else:
            return np.pad(input_tensor, ((0, 0), (0, target_len - input_tensor.shape[1])), mode='constant',
                          constant_values=0)
