import numpy as np
from featurize.base import str2int
import torch
import pandas as pd
from torch_geometric.data.data import Data
import os
import torch.utils.data as data
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
import random
import logging


class DrugBAN_featurize:
    def __init__(self, **config):
        self.feat_name = 'DrugBAN'

        self.max_drug_nodes = 290
        # self.max_drug_nodes = 290  # DrugBank Max: 551
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)
        self.drug_enc_dict = {}
        self.CHARPROTSET = {
            "A": 1,
            "C": 2,
            "B": 3,
            "E": 4,
            "D": 5,
            "G": 6,
            "F": 7,
            "I": 8,
            "H": 9,
            "K": 10,
            "M": 11,
            "L": 12,
            "O": 13,
            "N": 14,
            "Q": 15,
            "P": 16,
            "S": 17,
            "R": 18,
            "U": 19,
            "T": 20,
            "W": 21,
            "V": 22,
            "Y": 23,
            "X": 24,
            "Z": 25,
        }

        self.CHARPROTLEN = 25

    def integer_label_protein(self, sequence, max_length=1200):
        """
        Integer encoding for protein string sequence.
        Args:
            sequence (str): Protein string sequence.
            max_length: Maximum encoding length of input protein string.
        """
        encoding = np.zeros(max_length)
        for idx, letter in enumerate(sequence[:max_length]):
            try:
                letter = letter.upper()
                encoding[idx] = self.CHARPROTSET[letter]
            except KeyError:
                logging.warning(
                    f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
                )
        return encoding

    def drug_encoding(self, smile):

        v_d = self.fc(smiles=smile, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        if self.max_drug_nodes > num_actual_nodes:
            num_virtual_nodes = self.max_drug_nodes - num_actual_nodes

            # Add one column of zeros
            virtual_node_bit = torch.zeros([num_actual_nodes, 1])
            actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
            v_d.ndata['h'] = actual_node_feats
            virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
            v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        else:
            num_virtual_nodes = 0
            degrees = v_d.out_degrees().numpy()
            # 找到度数最低的节点，直到图的节点数降至max_nodes以下
            nodes_to_remove = degrees.argsort()[:v_d.number_of_nodes() - self.max_drug_nodes]
            # 删除这些节点
            virtual_node_bit = torch.zeros([num_actual_nodes, 1])
            actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
            mask = torch.ones(actual_node_feats.size(0), dtype=torch.bool)
            mask[nodes_to_remove] = False

            actual_node_feats = actual_node_feats[mask, :]
            v_d.remove_nodes(nodes_to_remove)
            v_d.ndata['h'] = actual_node_feats

        v_d = v_d.add_self_loop()
        return v_d

    def data_input(self, drugs, proteins, labels):

        data_list = []

        drug_dict = dict()
        prot_dict = dict()

        for smile in set(drugs):
            drug_feature = self.drug_encoding(smile)
            drug_dict[smile] = drug_feature

        for prot in set(proteins):
            prot_feature = self.integer_label_protein(prot)
            prot_dict[prot] = prot_feature

        for smile, protein, label in list(zip(drugs, proteins, labels)):
            # Drug
            v_d = drug_dict[smile]
            protein_input = prot_dict[protein]

            data = Data(
                y=torch.FloatTensor([label])
            )
            data.drug = v_d
            data.protein = torch.Tensor([protein_input])
            data_list.append([data])

        # print("max_drug_nodes: ", self.max_drug_nodes)
        return data_list
