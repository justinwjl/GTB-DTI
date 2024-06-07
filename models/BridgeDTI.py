import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F


class BridgeDTI(nn.Module):
    def __init__(self, n_output=2, outSize=128,
                 cHiddenSizeList=[1024],
                 fHiddenSizeList=[1024, 256],
                 fSize=1024, cSize=8500,
                 gcnHiddenSizeList=[128, 128], fcHiddenSizeList=[128], nodeNum=32, resnet=True,
                 hdnDropout=0.1, fcDropout=0.2, sampleType='CEL',
                 useFeatures={"kmers": True, "pSeq": True, "FP": True, "dSeq": True},
                 maskDTI=False, **config):
        super(BridgeDTI, self).__init__()

        cSize = config['cSize'] if 'cSize' in config else 8450  # 8422
        self.nodeEmbedding = TextEmbedding(
            torch.tensor(np.random.normal(size=(max(nodeNum, 0), outSize)), dtype=torch.float32), dropout=hdnDropout,
            name='nodeEmbedding')  # .to(device)

        self.amEmbedding = TextEmbedding(torch.eye(24), dropout=hdnDropout, freeze=True,
                                         name='amEmbedding')  # .to(device)
        self.pCNN = TextCNN(24, 64, [25], ln=True, name='pCNN')  # .to(device)
        self.pFcLinear = MLP(64, outSize, dropout=hdnDropout, bnEveryLayer=True, dpEveryLayer=True, outBn=True,
                             outAct=True, outDp=True, name='pFcLinear')  # .to(device)

        self.dCNN = TextCNN(75, 64, [7], ln=True, name='dCNN')  # .to(device)
        self.dFcLinear = MLP(64, outSize, dropout=hdnDropout, bnEveryLayer=True, dpEveryLayer=True, outBn=True,
                             outAct=True, outDp=True, name='dFcLinear')  # .to(device)

        self.fFcLinear = MLP(fSize, outSize, fHiddenSizeList, outAct=True, name='fFcLinear', dropout=hdnDropout,
                             dpEveryLayer=True, outDp=True, bnEveryLayer=True, outBn=True)  # .to(device)
        self.cFcLinear = MLP(cSize, outSize, cHiddenSizeList, outAct=True, name='cFcLinear', dropout=hdnDropout,
                             dpEveryLayer=True, outDp=True, bnEveryLayer=True, outBn=True)  # .to(device)

        self.nodeGCN = GCN(outSize, outSize, gcnHiddenSizeList, name='nodeGCN', dropout=hdnDropout, dpEveryLayer=True,
                           outDp=True, bnEveryLayer=True, outBn=True, resnet=resnet)  # .to(device)

        self.fcLinear = MLP(outSize, n_output, fcHiddenSizeList, dropout=fcDropout, bnEveryLayer=True,
                            dpEveryLayer=True)
        # .to(device)

        self.criterion = nn.BCEWithLogitsLoss()

        self.embModuleList = nn.ModuleList([])
        self.finetunedEmbList = nn.ModuleList([])
        self.moduleList = nn.ModuleList(
            [self.nodeEmbedding, self.cFcLinear, self.fFcLinear, self.nodeGCN, self.fcLinear,
             self.amEmbedding, self.pCNN, self.pFcLinear, self.dCNN, self.dFcLinear])
        self.sampleType = sampleType
        self.resnet = resnet
        self.nodeNum = nodeNum
        self.hdnDropout = hdnDropout
        self.useFeatures = useFeatures
        self.maskDTI = maskDTI

    def forward(self, data):
        unpack_data = data[0]
        aminoCtr = unpack_data.aminoCtr
        aminoSeq = unpack_data.aminoSeq
        atomFea = unpack_data.atomFea
        atomFin = unpack_data.atomFin
        device = aminoCtr.device
        
        Xam = (self.cFcLinear(aminoCtr).unsqueeze(1) if self.useFeatures['kmers'] else 0) + \
              (self.pFcLinear(self.pCNN(self.amEmbedding(aminoSeq))).unsqueeze(1) if self.useFeatures[
                  'pSeq'] else 0)  # => batchSize × 1 × outSize
        Xat = (self.fFcLinear(atomFin).unsqueeze(1) if self.useFeatures['FP'] else 0) + \
              (self.dFcLinear(self.dCNN(atomFea)).unsqueeze(1) if self.useFeatures[
                  'dSeq'] else 0)  # => batchSize × 1 × outSize

        if self.nodeNum > 0:
            node = self.nodeEmbedding.dropout2(self.nodeEmbedding.dropout1(self.nodeEmbedding.embedding.weight)).repeat(
                len(Xat), 1, 1)
            node = torch.cat([Xam, Xat, node], dim=1)  # => batchSize × nodeNum × outSize
            nodeDist = torch.sqrt(torch.sum(node ** 2, dim=2, keepdim=True) + 1e-8)  # => batchSize × nodeNum × 1

            cosNode = torch.matmul(node, node.transpose(1, 2)) / (
                    nodeDist * nodeDist.transpose(1, 2) + 1e-8)  # => batchSize × nodeNum × nodeNum
            # cosNode = cosNode*0.5 + 0.5
            # cosNode = F.relu(cosNode) # => batchSize × nodeNum × nodeNum
            cosNode[cosNode < 0] = 0
            cosNode[:, range(node.shape[1]), range(node.shape[1])] = 1  # => batchSize × nodeNum × nodeNum
            if self.maskDTI: cosNode[:, 0, 1] = cosNode[:, 1, 0] = 0
            D = torch.eye(node.shape[1], dtype=torch.float32).repeat(len(Xam), 1, 1).to(device)  # => batchSize × nodeNum × nodeNum
            D[:, range(node.shape[1]), range(node.shape[1])] = 1 / (torch.sum(cosNode, dim=2) ** 0.5)
            pL = torch.matmul(torch.matmul(D, cosNode), D)  # => batchSize × batchnodeNum × nodeNumSize
            node_gcned = self.nodeGCN(node, pL)  # => batchSize × nodeNum × outSize

            node_embed = node_gcned[:, 0, :] * node_gcned[:, 1, :]  # => batchSize × outSize
        else:
            node_embed = (Xam * Xat).squeeze(dim=1)  # => batchSize × outSize
        # if self.resnet:
        #    node_gcned += torch.cat([Xam[:,0,:],Xat[:,0,:]],dim=1)

        Y_pre = self.fcLinear(node_embed).squeeze(dim=1)
        return Y_pre
        # return torch.sigmoid(Y_pre)
        # return {"y_logit": self.fcLinear(node_embed).squeeze(dim=1)}  # , "loss":1*l2}


class TextSPP(nn.Module):
    def __init__(self, size=128, name='textSpp'):
        super(TextSPP, self).__init__()
        self.name = name
        self.spp = nn.AdaptiveAvgPool1d(size)

    def forward(self, x):
        return self.spp(x)


class TextSPP2(nn.Module):
    def __init__(self, size=128, name='textSpp2'):
        super(TextSPP2, self).__init__()
        self.name = name
        self.spp1 = nn.AdaptiveMaxPool1d(size)
        self.spp2 = nn.AdaptiveAvgPool1d(size)

    def forward(self, x):
        x1 = self.spp1(x).unsqueeze(dim=3)  # => batchSize × feaSize × size × 1
        x2 = self.spp2(x).unsqueeze(dim=3)  # => batchSize × feaSize × size × 1
        x3 = -self.spp1(-x).unsqueeze(dim=3)  # => batchSize × feaSize × size × 1
        return torch.cat([x1, x2, x3], dim=3)  # => batchSize × feaSize × size × 3


class TextEmbedding(nn.Module):
    def __init__(self, embedding, dropout=0.3, freeze=False, name='textEmbedding'):
        super(TextEmbedding, self).__init__()
        self.name = name
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding, dtype=torch.float32), freeze=freeze)
        self.dropout1 = nn.Dropout2d(p=dropout / 2)
        self.dropout2 = nn.Dropout(p=dropout / 2)
        self.p = dropout

    def forward(self, x):
        # x: batchSize × seqLen
        if self.p > 0:
            x = self.dropout2(self.dropout1(self.embedding(x)))
        else:
            x = self.embedding(x)
        return x


class ResDilaCNNBlock(nn.Module):
    def __init__(self, dilaSize, filterSize=64, dropout=0.15, name='ResDilaCNNBlock'):
        super(ResDilaCNNBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(filterSize, filterSize, kernel_size=3, padding=dilaSize, dilation=dilaSize),
            nn.InstanceNorm1d(filterSize),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(filterSize, filterSize, kernel_size=3, padding=dilaSize, dilation=dilaSize),
            nn.InstanceNorm1d(filterSize),
        )
        self.name = name

    def forward(self, x):
        # x: batchSize × filterSize × seqLen
        return x + self.layers(x)


class ResDilaCNNBlocks(nn.Module):
    def __init__(self, feaSize, filterSize, blockNum=10, dilaSizeList=[1, 2, 4, 8, 16], dropout=0.15,
                 name='ResDilaCNNBlocks'):
        super(ResDilaCNNBlocks, self).__init__()
        self.blockLayers = nn.Sequential()
        self.linear = nn.Linear(feaSize, filterSize)
        for i in range(blockNum):
            self.blockLayers.add_module(f"ResDilaCNNBlock{i}",
                                        ResDilaCNNBlock(dilaSizeList[i % len(dilaSizeList)], filterSize,
                                                        dropout=dropout))
        self.name = name

    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = self.linear(x)  # => batchSize × seqLen × filterSize
        x = self.blockLayers(x.transpose(1, 2)).transpose(1, 2)  # => batchSize × seqLen × filterSize
        return F.elu(x)  # => batchSize × seqLen × filterSize


class BatchNorm1d(nn.Module):
    def __init__(self, inSize, name='batchNorm1d'):
        super(BatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(inSize)
        self.name = name

    def forward(self, x):
        return self.bn(x)


class TextCNN(nn.Module):
    def __init__(self, featureSize, filterSize, contextSizeList, reduction='pool', actFunc=nn.ReLU, bn=False, ln=False,
                 name='textCNN'):
        super(TextCNN, self).__init__()
        moduleList = []
        bns, lns = [], []
        for i in range(len(contextSizeList)):
            moduleList.append(
                nn.Conv1d(in_channels=featureSize, out_channels=filterSize, kernel_size=contextSizeList[i],
                          padding=contextSizeList[i] // 2),
            )
            bns.append(nn.BatchNorm1d(filterSize))
            lns.append(nn.LayerNorm(filterSize))
        if bn:
            self.bns = nn.ModuleList(bns)
        if ln:
            self.lns = nn.ModuleList(lns)
        self.actFunc = actFunc()
        self.conv1dList = nn.ModuleList(moduleList)
        self.reduction = reduction
        self.batcnNorm = nn.BatchNorm1d(filterSize)
        self.bn = bn
        self.ln = ln
        self.name = name

    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = x.transpose(1, 2)  # => batchSize × feaSize × seqLen
        x = [conv(x).transpose(1, 2) for conv in self.conv1dList]  # => scaleNum * (batchSize × seqLen × filterSize)

        if self.bn:
            x = [b(i.transpose(1, 2)).transpose(1, 2) for b, i in zip(self.bns, x)]
        elif self.ln:
            x = [l(i) for l, i in zip(self.lns, x)]
        x = [self.actFunc(i) for i in x]

        if self.reduction == 'pool':
            x = [F.adaptive_max_pool1d(i.transpose(1, 2), 1).squeeze(dim=2) for i in x]
            return torch.cat(x, dim=1)  # => batchSize × scaleNum*filterSize
        elif self.reduction == 'None':
            return x  # => scaleNum * (batchSize × seqLen × filterSize)


class MLP(nn.Module):
    def __init__(self, inSize, outSize, hiddenList=[], dropout=0.0, bnEveryLayer=False, dpEveryLayer=False, outBn=False,
                 outAct=False, outDp=False, name='MLP', actFunc=nn.ReLU):
        super(MLP, self).__init__()
        self.name = name
        hiddens, bns = [], []
        for i, os in enumerate(hiddenList):
            hiddens.append(nn.Sequential(
                nn.Linear(inSize, os),
            ))
            bns.append(nn.BatchNorm1d(os))
            inSize = os
        bns.append(nn.BatchNorm1d(outSize))
        self.actFunc = actFunc()
        self.dropout = nn.Dropout(p=dropout)
        self.hiddens = nn.ModuleList(hiddens)
        self.bns = nn.ModuleList(bns)
        self.out = nn.Linear(inSize, outSize)
        self.bnEveryLayer = bnEveryLayer
        self.dpEveryLayer = dpEveryLayer
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp

    def forward(self, x):
        for h, bn in zip(self.hiddens, self.bns):
            x = h(x)
            if self.bnEveryLayer:
                x = bn(x) if len(x.shape) == 2 else bn(x.transpose(-1, -2)).transpose(-1, -2)
            x = self.actFunc(x)
            if self.dpEveryLayer:
                x = self.dropout(x)
        x = self.out(x)
        if self.outBn: x = self.bns[-1](x) if len(x.shape) == 2 else self.bns[-1](x.transpose(-1, -2)).transpose(-1, -2)
        if self.outAct: x = self.actFunc(x)
        if self.outDp: x = self.dropout(x)
        return x


class GCN(nn.Module):
    def __init__(self, inSize, outSize, hiddenSizeList=[], dropout=0.0, bnEveryLayer=False, dpEveryLayer=False,
                 outBn=False, outAct=False, outDp=False, resnet=False, name='GCN', actFunc=nn.ReLU):
        super(GCN, self).__init__()
        self.name = name
        hiddens, bns = [], []
        for i, os in enumerate(hiddenSizeList):
            hiddens.append(nn.Sequential(
                nn.Linear(inSize, os),
            ))
            bns.append(nn.BatchNorm1d(os))
            inSize = os
        bns.append(nn.BatchNorm1d(outSize))
        self.actFunc = actFunc()
        self.dropout = nn.Dropout(p=dropout)
        self.hiddens = nn.ModuleList(hiddens)
        self.bns = nn.ModuleList(bns)
        self.out = nn.Linear(inSize, outSize)
        self.bnEveryLayer = bnEveryLayer
        self.dpEveryLayer = dpEveryLayer
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp
        self.resnet = resnet

    def forward(self, x, L):
        # x: nodeNum × feaSize; L: batchSize × nodeNum × nodeNum
        for h, bn in zip(self.hiddens, self.bns):
            a = h(torch.matmul(L, x))  # => batchSize × nodeNum × os
            if self.bnEveryLayer:
                if len(L.shape) == 3:
                    a = bn(a.transpose(1, 2)).transpose(1, 2)
                else:
                    a = bn(a)
            a = self.actFunc(a)
            if self.dpEveryLayer:
                a = self.dropout(a)
            if self.resnet and a.shape == x.shape:
                a += x
            x = a
        a = self.out(torch.matmul(L, x))  # => batchSize × nodeNum × outSize
        if self.outBn:
            if len(L.shape) == 3:
                a = self.bns[-1](a.transpose(1, 2)).transpose(1, 2)
            else:
                a = self.bns[-1](a)
        if self.outAct: a = self.actFunc(a)
        if self.outDp: a = self.dropout(a)
        if self.resnet and a.shape == x.shape:
            a += x
        x = a
        return x
