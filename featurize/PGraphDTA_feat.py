import torch
from torch_geometric.data.data import Data
from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer, EsmModel
import dgl
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph
import numpy as np
from tqdm import tqdm
import re
from rdkit import Chem
from featurize.base import str2int


class PGraphDTA_PLM_featurize:

    def __init__(self, **config):
        self.feat_name = 'PGraphDTA_PLM'
        model_choice = config['model_choice']
        prot_models = ["Rostlab/prot_bert", "yarongef/DistilProtBert", "facebook/esm2_t33_650M_UR50D"]
        PROT_MODEL = prot_models[model_choice]
        model_names = ["ProtBERT", "DistilProtBERT", "ESM-2"]
        self.model_name = model_names[model_choice]

        self.device = torch.device("cuda:0")
        if model_choice == 2:
            self.prot_tokenizer = AutoTokenizer.from_pretrained(PROT_MODEL, do_lower_case=False)
            self.prot_model = EsmModel.from_pretrained(PROT_MODEL)
        else:
            self.prot_tokenizer = BertTokenizer.from_pretrained(PROT_MODEL, do_lower_case=False)
            self.prot_model = BertModel.from_pretrained(PROT_MODEL)
        self.prot_model.to(self.device)

        self.MAX_PROT_LEN = config['MAX_PROT_LEN']
        self.MAX_MOLECULE_LEN = config['MAX_MOLECULE_LEN']

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer()

    def data_input(self, drugs, proteins, labels):
        data_list = []
        targets = [self.preprocess_protein(target) for target in proteins]

        tmp_batch_size = 8
        target_batches = [targets[i:i + tmp_batch_size] for i in range(0, len(targets), tmp_batch_size)]
        protein_embs = []
        for tmp_batch in tqdm(target_batches):
            encoded_proteins = self.prot_tokenizer(tmp_batch,
                                                   return_tensors='pt',
                                                   max_length=self.MAX_PROT_LEN,
                                                   truncation=True,
                                                   padding=True)

            encoded_proteins = encoded_proteins.to(self.device)
            with torch.no_grad():
                target_embeddings = self.prot_model(**encoded_proteins).last_hidden_state[:, 0, :]
            for target_embedding in target_embeddings.cpu().detach().numpy():
                protein_embs.append(target_embedding)

        # Distances are 1D np array of size 100 * 100 each value being 0 initially
        dist = np.zeros(self.MAX_MOLECULE_LEN * self.MAX_MOLECULE_LEN)
        for smile in drugs:
            self.extract_complex_distances(smile, dist)

        for smile, protein, label in list(zip(drugs, protein_embs, labels)):
            # Drug
            mol_graph = smiles_to_bigraph(smile,
                                          node_featurizer=self.atom_featurizer,
                                          edge_featurizer=self.bond_featurizer,
                                          )
            drug = dgl.add_self_loop(mol_graph)

            data = Data(
                y=torch.FloatTensor([label])
            )
            data.drug = drug
            data.protein = torch.Tensor([protein])
            data.dist = torch.Tensor([dist])
            data_list.append([data])
        return data_list

    def extract_complex_distances(self, smile, distances):
        mol = Chem.MolFromSmiles(smile)
        # Tries to add hydrogens to the molecule mol.
        # Utilizes AllChem.EmbedMolecule to generate 3D coordinates for the structure of mol
        # Applies AllChem.UFFOptimizeMolecule to optimize the energy and geometry of mol.
        # Removes hydrogens from the optimized mol.
        # This function primarily aims to optimize the structure of the mol molecule
        # and calculate its 2D coordinates when necessary
        try:
            mol = Chem.AddHs(mol)
            Chem.AllChem.EmbedMolecule(mol, maxAttempts=5000)
            Chem.AllChem.UFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
        except:
            Chem.AllChem.Compute2DCoords(mol)

        MAX_MOLECULE_LEN = self.MAX_MOLECULE_LEN
        # Get the coordinates of each atom
        if mol is not None:
            conf = mol.GetConformer()
            coords = [(atom.GetAtomMapNum(), conf.GetAtomPosition(atom.GetIdx())) for atom in mol.GetAtoms()]
            for i in range(len(coords)):
                for j in range(len(coords)):
                    distance = (coords[i][1] - coords[j][1]).Length()
                    index = i * self.MAX_MOLECULE_LEN + j
                    if index < MAX_MOLECULE_LEN * MAX_MOLECULE_LEN and distance < 10:
                        distances[index] = 1

    @staticmethod
    def preprocess_protein(sequence):
        processProtein = [aa for aa in sequence]  # aa is a single amino acid
        processProtein = " ".join(processProtein)
        processProtein = re.sub(r"[UZOB]", "X", processProtein)
        return processProtein


class PGraphDTA_CNN_featurize:

    def __init__(self, **config):
        self.feat_name = 'PGraphDTA_CNN'

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer()

        self.seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
        self.seq_dict = {v: (i + 1) for i, v in enumerate(self.seq_voc)}
        self.seq_dict_len = len(self.seq_dict)
        self.max_prot_len = config['MAX_PROT_LEN']

    def drug_encoding(self, smile):
        mol_graph = smiles_to_bigraph(smile,
                                      node_featurizer=self.atom_featurizer,
                                      edge_featurizer=self.bond_featurizer,
                                      )
        drug = dgl.add_self_loop(mol_graph)
        return drug

    def prot_encoding(self, prot):
        return str2int(prot, self.seq_dict, self.max_prot_len)

    def data_input(self, drugs, proteins, labels):

        data_list = []

        drug_dict = dict()
        prot_dict = dict()

        for drug in set(drugs):
            drug_dict[drug] = self.drug_encoding(drug)

        for prot in set(proteins):
            prot_dict[prot] = self.prot_encoding(prot)

        for smile, protein, label in list(zip(drugs, proteins, labels)):
            drug = drug_dict[smile]
            prot = prot_dict[protein]

            data = Data(
                y=torch.FloatTensor([label])
            )
            data.drug = drug
            data.protein = torch.LongTensor([prot])
            data_list.append([data])
        return data_list
