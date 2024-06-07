import torch
import os
from torchvision import transforms
from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from torch_geometric.data import Data
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from featurize.base import pad_or_truncate

class MCLDTI_featurize:
    def __init__(self, **config):
        self.feat_name = 'MCLDTI'
        self.pic_size = 256
        self.transformImg = transforms.Compose([
            transforms.ToTensor()])

        self.protein_dim = 256
        self.smile_dim = 512
        self.ngram = 1
        self.prot_dict = defaultdict(lambda: len(self.prot_dict))
        self.smile_dict = defaultdict(lambda: len(self.smile_dict))
        self.prot_dict['pad'] = 0
        self.smile_dict['pad'] = 0

    def split_sequence(self, sequence, word_dict):
        words = [word_dict[sequence[i: i+self.ngram]]
                for i in range(len(sequence)- self.ngram + 1)]
        return np.array(words)

    def data_input(self, drugs, proteins, labels):
        
        data_list = []

        prot_dict = dict()
        smile_dict = dict()
        
        for prot in set(proteins):
            prot_seq = self.split_sequence(prot, self.prot_dict)
            prot_pad = pad_or_truncate(prot_seq, max_len=self.protein_dim)
            prot_dict[prot] = prot_pad

        for drug in set(drugs):
            smile_seq = self.split_sequence(drug, self.smile_dict)
            smile_pad = pad_or_truncate(smile_seq, max_len=self.smile_dim)
            raw_img = self.smile2img(drug)
            image = raw_img.convert('RGB')
            image = self.transformImg(image)
            smile_dict[drug] = (smile_pad, image)

        for smile, protein, label in tqdm(list(zip(drugs, proteins, labels))):

            # image = smile_img_dict[smile]
            smile_seq, image = smile_dict[smile]
            prot_seq = prot_dict[protein]

            # Pack -----------------------------------------
            data = Data(
                y=torch.FloatTensor([label])
            )
            data.image = torch.FloatTensor([image.cpu().detach().numpy()])
            data.drug = torch.LongTensor([smile_seq])
            data.protein = torch.LongTensor([prot_seq])
            data_list.append([data])
        return data_list

    @staticmethod
    def smile2feature(smile):
        fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
        mol = Chem.MolFromSmiles(smile)
        feats = factory.GetFeaturesForMol(mol)
        return feats

    def smile2img(self, smile):
        mol = Chem.MolFromSmiles(smile)
        canonical_smi = Chem.MolToSmiles(mol)
        canonical_mol = Chem.MolFromSmiles(canonical_smi)
        img = Draw.MolToImage(mol, size=(self.pic_size, self.pic_size), wedgeBonds=False)
        return img