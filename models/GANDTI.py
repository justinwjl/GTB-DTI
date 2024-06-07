# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 14:18:27 2020

@author: shuyu
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

class GanDTI(nn.Module):
    def __init__(self, n_output, features, GNN_depth, MLP_depth, dropout, **config):
        super(GanDTI, self).__init__()
        
        atom_dict = pickle.load(open(config['feat_root'] + '/atom_dict', 'rb'))
        amino_dict = pickle.load(open(config['feat_root'] + '/amino_dict', 'rb'))
        self.embed_compound = nn.Embedding(len(atom_dict), features)
        self.embed_protein = nn.Embedding(len(amino_dict), features)
        self.GNN_depth = GNN_depth
        self.GNN = nn.ModuleList(nn.Linear(features, features) for i in range(GNN_depth))
        self.W_att = nn.Linear(features, features)
        self.MLP_depth = MLP_depth
        self.MLP = nn.ModuleList(nn.Linear(features*2, features*2) for i in range(self.MLP_depth))
        self.finnal_out = nn.Linear(features*2, n_output)
        self.dropout = nn.Dropout(dropout)
        
    def Attention(self, compound, protein, compound_mask, protein_mask):
        compound_h = torch.relu(self.W_att(compound))
        protein_h = torch.relu(self.W_att(protein))
        # mult = compound @ protein_h.transpose(2, 1)
        protein_mask = protein_mask.unsqueeze(-1)
        protein_h = protein_h * protein_mask
        mult = torch.bmm(compound.unsqueeze(1), protein_h.transpose(2, 1))
        weights = torch.tanh(mult)
        protein = weights.transpose(2, 1) * protein_h
        # protein_vector = torch.mean(protein, 1)
        protein_vector = torch.sum(protein, 1) / (torch.sum(protein_mask, 1)+1e-6)
        return protein_vector
        
    def GraphNeuralNet(self, compound, A, GNN_depth, mask):
        residual = compound
        for i in range(GNN_depth):
            compound_h = F.leaky_relu(self.GNN[i](compound))
            compound = compound + torch.matmul(A, compound_h)
        
        compound = compound + residual
        # compound_vector = torch.unsqueeze(torch.mean(compound, 1), 1)
        mask = mask.unsqueeze(-1)
        compound = compound * mask
        compound_vector = torch.sum(compound, 1) / (torch.sum(mask, 1)+1e-6)
        return compound_vector
    
    def MLP_module(self, compound_protein, MLP_depth):
        for i in range(MLP_depth):
            compound_protein = torch.relu(self.MLP[i](compound_protein))
        compound_protein = self.dropout(compound_protein)
        out = self.finnal_out(compound_protein)
        
        return out
    
    def forward(self, data):
        data = data[0]
        compound, A, protein, compound_mask, prot_mask = data.x, data.adj, data.prot, data.x_mask, data.prot_mask
        
        compound_embed = self.embed_compound(compound)
        compound_vector = self.GraphNeuralNet(compound_embed, A, self.GNN_depth, compound_mask)
        
        protein_embed = self.embed_protein(protein)
        protein_vector = self.Attention(compound_vector, protein_embed, compound_mask, prot_mask)
        
        compound_protein = torch.cat((compound_vector, protein_vector), 1)
        out = self.MLP_module(compound_protein, self.MLP_depth)
        return out
        
        