"""
Provides some tools
"""
import logging
import os
import pickle
import time
import easydict
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import yaml
from gensim.models import Word2Vec
from tdc.multi_pred import bi_pred_dataset
from torch_geometric.data import Batch

MB = 1024 ** 2
GB = 1024 ** 3


def load_config(cfg_file):
    with open(cfg_file, "r", encoding='utf-8') as fin:
        raw_text = fin.read()
    cfg = yaml.load(raw_text, Loader=yaml.CLoader)
    cfg = easydict.EasyDict(cfg)
    return cfg


def read_csv(drug, protein, label, filename):
    df = pd.read_csv(filename)
    out = [df[drug].tolist(), df[protein].tolist(), df[label].tolist()]
    return out


def load_data(split_types=('train', 'val', 'test'), load_from_tdc=False, cfg=None, seed=None, task='regression'):
    data_path = os.path.join(cfg['path'], cfg['class'])
    if not load_from_tdc:
        if not os.path.exists(data_path):
            os.mkdir(data_path)
            raise NotImplementedError('Create data first')
        out_dict = {}
        out_dict['kfold'] = [[], [], []]
        for split in split_types:
            data_file = os.path.join(data_path, f'{split}.csv')
            if not os.path.exists(data_file):
                continue
            out_dict[split] = read_csv('SMILES', 'Target Sequence', 'Label', data_file)
            if split in ['train', 'val']:
                for i in range(3):
                    out_dict['kfold'][i].extend(out_dict[split][i])
        split_file = data_path
    else:
        from tdc.multi_pred import DTI
        if cfg['class'] in ['Human', 'C.elegans', 'Drugbank']:
            data = DTI_dataset(name=cfg['class'], path=cfg['path'])
        else:
            data = DTI(name=cfg['class'], path=cfg['path'])
            data.harmonize_affinities(mode='max_affinity')
            if task == 'classification':
                data.binarize(threshold=cfg['threshold'])
            else:
                data.convert_to_log(form='binding')
        split_file = os.path.join(cfg['path'], cfg['class'], task + '_' + cfg['split'] + '_' + str(seed))
        if not os.path.exists(split_file):
            os.makedirs(split_file)
        split = data.get_split(method=cfg['split'], seed=seed)
        out_dict = {}
        out_dict['kfold'] = [[], [], []]
        for i, item in enumerate(split):
            out_dict[item] = [split[item]['Drug'].tolist(), split[item]['Target'].tolist(), split[item]['Y'].tolist()]
            if item in ['train', 'valid']:
                for i in range(3):
                    out_dict['kfold'][i].extend(out_dict[item][i])
    return split_file, out_dict


def interaction_dataset_load(name, path):
    file = os.path.join(path, name)
    df = pd.read_csv(file + '.csv')
    if name == 'Human':
        entity1 = df['compound_iso_smiles']
        entity2 = df['target_sequence']
        raw_y = df['affinity']
        entity1_idx, unique1 = pd.factorize(entity1)
        entity2_idx, unique2 = pd.factorize(entity2)
    elif name == 'C.elegans':
        entity1 = df['smile']
        entity2 = df['protein']
        raw_y = df['affinity']
        entity1_idx, unique1 = pd.factorize(entity1)
        entity2_idx, unique2 = pd.factorize(entity2)
    elif name == 'Drugbank':
        entity1 = df['smile']
        entity2 = df['protein']
        raw_y = df['affinity']
        entity1_idx = df['drug_idx']
        entity2_idx = df['protein_idx']
    return entity1, entity2, raw_y.values, entity1_idx, entity2_idx


class DTI_dataset(bi_pred_dataset.DataLoader):
    def __init__(self, name='Human', path='data'):
        entity1, entity2, raw_y, entity1_idx, entity2_idx = interaction_dataset_load(name, path)
        self.entity1 = entity1
        self.entity2 = entity2
        self.raw_y = raw_y
        self.entity1_idx = entity1_idx
        self.entity2_idx = entity2_idx
        self.entity1_name = "Drug"
        self.entity2_name = "Target"
        self.label_name = "Y"
        self.name = name
        self.aux_column = None
        self.y = raw_y
        self.path = path
        self.log_flag = False
        self.two_types = False


def cuda(obj, *args, **kwargs):
    """
    Transfer any nested container of tensors to CUDA.
    """
    if hasattr(obj, "cuda"):
        return obj.cuda(*args, **kwargs)
    elif isinstance(obj, (str, bytes)):
        return obj
    elif isinstance(obj, dict):
        return type(obj)({k: cuda(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return type(obj)(cuda(x, *args, **kwargs) for x in obj)

    raise TypeError("Can't transfer object type `%s`" % type(obj))


def loss_cal(loss_fn, output, target, type):
    if type == 'regression':
        return loss_fn(output.squeeze(), target)
    elif type == 'classification':
        return loss_fn(output, target.long())  # )   #


def eval_func(eval_fn, target, output, type):
    if type == 'regression':
        return eval_fn(target, output.squeeze())
    elif type == 'classification':
        return eval_fn(target, torch.argmax(output, dim=1))


def model_size_in_bytes(model, only_trainable=False):
    """
    计算 PyTorch 模型的大小（以字节为单位）。
    
    参数:
    model (torch.nn.Module): 要计算大小的模型。
    only_trainable (bool): 如果为 True，则只计算可训练参数的大小。
    
    返回:
    int: 模型大小（字节）。
    """
    # param_size = 0
    # for param in model.parameters():
    #     if only_trainable:
    #         if param.requires_grad:
    #             param_size += param.nelement() * param.element_size()
    #     else:
    #         param_size += param.nelement() * param.element_size()
    # return param_size
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size)
    return all_size


def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)


def dump_list(the_list, filename):
    with open(filename, 'wb') as f:
        pickle.dump(the_list, f)


class Corpus(object):
    """ An iteratable for training seq2vec models. """

    def __init__(self, prot, ngram, pad=False):
        self.prot = prot
        self.ngram = ngram
        self.pad = pad

    def __iter__(self):
        for sentence in self.prot:
            N = len(sentence)
            yield [sentence[i:i + self.ngram] for i in range(N - self.ngram + 1)]
        if self.pad:
            yield ['<pad>']


def w2v_train(saved_path, sent_list, ngram, padding_train=False):
    if os.path.exists(saved_path):
        model = Word2Vec.load(saved_path)
    else:
        sent_corpus = Corpus(prot=sent_list, ngram=ngram, pad=padding_train)
        model = Word2Vec(window=5, min_count=1, workers=6)
        model.build_vocab(sent_corpus)
        model.train(sent_corpus, epochs=30, total_examples=model.corpus_count)
        model.save(saved_path)
    return model


def get_memory_usage(gpu, print_info=False):
    """Get accurate gpu memory usage by querying torch runtime"""
    allocated = torch.cuda.memory_allocated(gpu)
    reserved = torch.cuda.memory_reserved(gpu)
    if print_info:
        print("allocated: %.2f MB" % (allocated / 1024 / 1024), flush=True)
        print("reserved:  %.2f MB" % (reserved / 1024 / 1024), flush=True)
    return allocated


def compute_tensor_bytes(tensors):
    """Compute the bytes used by a list of tensors"""
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]

    ret = 0
    for x in tensors:
        if x.dtype in [torch.int64, torch.long]:
            ret += np.prod(x.size()) * 8
        if x.dtype in [torch.float32, torch.int, torch.int32]:
            ret += np.prod(x.size()) * 4
        elif x.dtype in [torch.bfloat16, torch.float16, torch.int16]:
            ret += np.prod(x.size()) * 2
        elif x.dtype in [torch.int8]:
            ret += np.prod(x.size())
        else:
            print(x.dtype)
            raise ValueError()
    return ret
