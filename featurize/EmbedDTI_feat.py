import numpy as np
from featurize.base import *
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from scipy.sparse import csr_matrix
from utils import *
from collections import defaultdict
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from torch_geometric.data import Data

class EmbedDTI_featurize:
    def __init__(self, max_prot_len=1000, MST_MAX_WEIGHT=100):
        self.feat_name = 'EmbedDTI'
        self.atom_map = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                         'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu',
                         'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
        self.degree_map = list(range(0, 11))
        self.numH_map = list(range(0, 11))
        self.implicit_map = list(range(0, 11))
        self.totalval_map = list(range(0, 11))
        self.charge_map = list(range(0, 11))

        self.seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
        self.seq_dict = {v: (i + 1) for i, v in enumerate(self.seq_voc)}
        self.max_prot_len = max_prot_len
        self.MST_MAX_WEIGHT = MST_MAX_WEIGHT

    def atom_features(self, atom):
        return np.array(
            one_of_k_encoding_unk(atom.GetSymbol(), self.atom_map) +
            one_of_k_encoding(atom.GetDegree(), self.degree_map) +
            one_of_k_encoding_unk(atom.GetTotalNumHs(), self.numH_map) +
            one_of_k_encoding_unk(atom.GetImplicitValence(), self.implicit_map) +
            one_of_k_encoding_unk(atom.GetTotalValence(), self.totalval_map) +
            one_of_k_encoding_unk(atom.GetFormalCharge(), self.charge_map) +
            [atom.GetIsAromatic()] + 
            [atom.IsInRing()]
        )

    def smile_to_graph(self, smile):
        smile = normalize_smile(smile)
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

        return c_size, features, edge_index

    def get_mol(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: 
            return None
        try:
            Chem.Kekulize(mol)
        except:
            pass
        return mol

    # 构建邻接矩阵和特征矩阵
    def cluster_graph(self, mol, idx):
        n_atoms = mol.GetNumAtoms()
        # if n_atoms == 1: #special case
    #    	return [[0]], []
        cliques = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if not bond.IsInRing():
                cliques.append([a1,a2])

        ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
        cliques.extend(ssr)

        # nei_list为原子属于哪个基团
        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)

        #Merge Rings with intersection > 2 atoms
        for i in range(len(cliques)):
            if len(cliques[i]) <= 2: continue
            for atom in cliques[i]:
                for j in nei_list[atom]:
                    if i >= j or len(cliques[j]) <= 2: 
                        continue
                    inter = set(cliques[i]) & set(cliques[j])
                    if len(inter) > 2:
                        cliques[i].extend(cliques[j])
                        cliques[i] = list(set(cliques[i]))
                        cliques[j] = []
        
        cliques = [c for c in cliques if len(c) > 0]
        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)

        # Build edges and add singleton cliques
        edges = defaultdict(int)
        for atom in range(n_atoms):
            if len(nei_list[atom]) <= 1: 
                continue
            cnei = nei_list[atom]
            bonds = [c for c in cnei if len(cliques[c]) == 2]
            rings = [c for c in cnei if len(cliques[c]) > 4]
            
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1,c2 = cnei[i],cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1,c2)] < len(inter):
                        edges[(c1,c2)] = len(inter) # cnei[i] < cnei[j] by construction
                        edges[(c2,c1)] = len(inter)
        try:
            edges = [u + (self.MST_MAX_WEIGHT-v,) for u,v in edges.items()]
            row,col,data = zip(*edges)
            data = list(data)
            for i in range(len(data)):
                data[i] = 1
            data = tuple(data)
            n_clique = len(cliques)
            clique_graph = csr_matrix( (data,(row,col)), shape=(n_clique,n_clique) )
            edges = [[row[i],col[i]] for i in range(len(row))]
        except:
            edges = None
        
        return cliques, edges



    def clique_features(self, clique, edges, clique_idx, smile):
        NumAtoms = len(clique) # 基团中除去氢原子的原子数
        NumEdges = 0  # 与基团所连的边数
        if edges is None:
            pass
        else:
            for edge in edges:
                if clique_idx == edge[0] or clique_idx == edge[1]:
                    NumEdges += 1
        mol = Chem.MolFromSmiles(smile)
        NumHs = 0 # 基团中氢原子的个数
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() in clique:
                NumHs += atom.GetTotalNumHs()
        # 基团中是否包含环
        IsRing = 0
        if len(clique) > 2:
            IsRing = 1
        # 基团中是否有键
        IsBond = 0
        if len(clique) == 2:
            IsBond = 1

        return np.array(
            one_of_k_encoding_unk(NumAtoms,[0,1,2,3,4,5,6,7,8,9,10]) + 
            one_of_k_encoding_unk(NumEdges,[0,1,2,3,4,5,6,7,8,9,10]) + 
            one_of_k_encoding_unk(NumHs,[0,1,2,3,4,5,6,7,8,9,10]) + 
            [IsRing] + 
            [IsBond]
            )

    def protein_encoding(self, prot):
        return str2int(prot, self.seq_dict, self.max_prot_len)

    def data_input(self, drugs, proteins, labels):

        drug_dict = dict()
        prot_dict = dict()
        clique_dict = dict()
        cluster_dict = dict()
        
        for i, drug in enumerate(set(drugs)):
            drug_dict[drug] = self.smile_to_graph(drug)
            mol = self.get_mol(drug)
            cluster_dict[drug] = self.cluster_graph(mol, i)

        
        for drug in set(drugs):
            c_features = []
            clique, edge = cluster_dict[drug]
            for idx in range(len(clique)):
                cq_features = self.clique_features(clique[idx], edge, idx, drug)
                c_features.append( cq_features / sum(cq_features) )
            clique_dict[drug] = (len(clique), c_features, edge)
        
        for prot in set(proteins):
            prot_dict[prot] = self.protein_encoding(prot)
        
        data_list = []
        for smile, protein, label in list(zip(drugs, proteins, labels)):
            c_size, features, edge_index = drug_dict[smile]
            cli_size, cli_features, cli_edge_index = clique_dict[smile]
            if cli_edge_index is None:
                cli_edge_index = torch.empty((0, 2), dtype=torch.long)
            if edge_index is None or len(edge_index) == 0:
                edge_index = torch.empty((0, 2), dtype=torch.long)

            data = Data(
                x=torch.Tensor(features),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                y=torch.FloatTensor([label])
            )
            protein_input = prot_dict[protein]
            data.target = torch.LongTensor([protein_input])
            data.__setitem__('c_size', torch.LongTensor([c_size]))
            
            data_cli = Data(
                x=torch.Tensor(cli_features),
                edge_index=torch.LongTensor(cli_edge_index).transpose(1, 0),
                y=torch.FloatTensor([label])
            )
            data_cli.__setitem__('cli_size', torch.LongTensor([cli_size]))
            
            data_list.append([data, data_cli])
            
        return data_list
