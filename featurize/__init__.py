import warnings

warnings.filterwarnings("ignore")
from .GraphDTA_feat import GraphDTA_featurize
from .MGraphDTA_feat import MGraph_featurize
from .Hyperattention_feat import Hyperattention_featurize
from .MolTrans_feat import MolTrans_featurize
from .CPI_feat import CPI_featurize
from .CSDTI_feat import CSDTI_featurize
from .BridgeDTI_feat import BridgeDTI_featurize
from .EmbedDTI_feat import EmbedDTI_featurize
from .BACPI_feat import BACPI_featurize
from .TransCPI_feat import TransCPI_featurize
from .ICAN_feat import ICAN_featurize
from .GraphCPI_feat import GraphCPI_featurize
from .IIFDTI_feat import IIFDTI_featurize
from .MRBDTA_feat import MRBDTA_featurize
from .DrugBAN_feat import DrugBAN_featurize
from .AMMVF_feat import AMMVF_featurize
from .our3_feat import our3_featurize
from .PGraphDTA_feat import PGraphDTA_PLM_featurize, PGraphDTA_CNN_featurize
from .ColdDTA_feat import coldDTA_featurize
from .SubMDTA_feat import SubMDTA_featurize
# from .gt_feat import gt_featurize

def get_featurizer(name, **config):
    if name in ['GAT_GCN', 'GATNet', 'GINConvNet', 'GCNNet', 'DeepGLSTM', 'SAGNet', 'SAGNet_HIER', 'HGC', 'GEN',
                'GCNNet_pretrain', 'IMAEN', 'TDGraphDTA']:
        return GraphDTA_featurize(**config)
    elif name in ['AttentionDTI']:
        return Hyperattention_featurize()
    elif name in ['MolTransformer', 'FOTFCPI']:
        return MolTrans_featurize()
    elif name in ['MGraphDTA']:
        return MGraph_featurize()
    elif name in ['ColdDTA']:
        return coldDTA_featurize()
    elif name in ['CPINet', 'GanDTI']:
        return CPI_featurize(**config)
    elif name in ['CSDTI']:
        return CSDTI_featurize()
    elif name in ['BridgeDTI']:
        return BridgeDTI_featurize()
    elif name in ['EmbedDTI_Ori', 'EmbedDTI_Attn']:
        return EmbedDTI_featurize()
    elif name in ['BACPI']:
        return BACPI_featurize(**config)
    elif name in ['TransCPI']:
        return TransCPI_featurize(**config)
    elif name in ['ICAN_model']:
        return ICAN_featurize(**config)
    elif name in ['GraphCPI_GCN', 'GraphCPI_GIN', 'GraphCPI_GAT', 'GraphCPI_GATGCN']:
        return GraphCPI_featurize(**config)
    elif name in ['IIFDTI']:
        return IIFDTI_featurize(**config)
    elif name in ['MRBDTA']:
        return MRBDTA_featurize()
    elif name in ['AMMVF']:
        return AMMVF_featurize(**config)
    elif name in ['DrugBAN']:
        return DrugBAN_featurize()
    elif name in ['PGraphDTA_PLM']:
        return PGraphDTA_PLM_featurize(**config)
    elif name in ['PGraphDTA_CNN']:
        return PGraphDTA_CNN_featurize(**config)
    elif name in ['MATDTI7']:
        return our3_featurize()
    elif name in ['SubMDTA']:
        return SubMDTA_featurize(**config)
    else:
        print('Wrong Name!')
