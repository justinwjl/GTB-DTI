import warnings

warnings.filterwarnings("ignore")
import numpy as np
from rdkit import Chem
import pandas as pd


def read_data(filename):
    df = pd.read_csv('data/' + filename)
    drugs, prots, Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['affinity'])
    return drugs, prots, Y


def str2int(input_str, input_dict, max_len):
    x = np.zeros(max_len)
    for i, ch in enumerate(input_str[: max_len]):
        x[i] = input_dict[ch]
    return x


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("the_input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def normalize_smile(smile):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smile), isomericSmiles=True)


def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


def encoding_unk(x, allowable_set):
    enlist = [False for i in range(len(allowable_set))]
    i = 0
    for atom in x:
        if atom in allowable_set:
            enlist[allowable_set.index(atom)] = True
            i += 1
    if i != len(x):
        enlist[-1] = True
    return enlist

def get_padding(input, max_len):
    vec = np.zeros(max_len)
    mask = np.zeros(max_len)
    for i in range(len(input)):
        if i < max_len:
            vec[i] = input[i]
            mask[i] = 1
    return vec, mask

def padding_dim1(input, max_len):

    pad_row = max_len - len(input)
    mask = np.ones(max_len)
    mask[max_len-pad_row:] = 0.0
    x_padding = np.pad(input, ((0, pad_row), (0, 0)), 
                mode='constant', constant_values=0)
    return x_padding, mask
    
def padding_dim2(input, max_len):

    pad_len = max_len - len(input)
    return np.pad(input, ((0, pad_len), (0, pad_len)), mode='constant', constant_values=0)

def seq_to_kmers(seq, ngram):
    """ Divide a string into a list of kmers strings.

    Parameters:
        seq (string)
    Returns:
        List containing a list of kmers.
    """
    N = len(seq)
    return [seq[i: i+ngram] for i in range(N - ngram + 1)]


def pad_or_truncate(input, max_len, pad_2d=False, return_mask=False):
    mask = None

    if input.ndim == 1:
        # Handle 1D array
        if len(input) > max_len:
            # Truncate the input
            input = input[:max_len]
            if return_mask:
                mask = np.ones(max_len)
        else:
            # Pad the input
            pad_len = max_len - len(input)
            if return_mask:
                mask = np.ones(max_len)
                mask[-pad_len:] = 0
            input = np.pad(input, (0, pad_len), mode='constant', constant_values=0)
    elif input.ndim == 2:
        # Handle 2D array
        if input.shape[0] > max_len:
            if pad_2d is False:
                input = input[:max_len, :]
                if return_mask:
                    mask = np.ones(max_len)
            else:
                if max_len < input.shape[1]:
                    input = input[: max_len, :max_len]
                else:
                    input = np.pad(input[: max_len, :], ((0, 0), (0, max_len-input.shape[1])), mode='constant', constant_values=0)
                if return_mask:
                    mask = np.ones((max_len, max_len))
                    if max_len > input.shape[1]:
                        mask[:, input.shape[1]-max_len] = 0
        else:
            # Pad in the first dimension
            pad_len_row = max_len - input.shape[0]
            if pad_2d is False:
                pad_width = ((0, pad_len_row), (0, 0))
                input = np.pad(input, pad_width, mode='constant', constant_values=0)
                if return_mask:
                    mask = np.ones(max_len)
                    mask[-pad_len_row:] = 0
            else:
                pad_len_col = max_len - input.shape[1]

                pad_width = ((0, pad_len_row), (0, pad_len_col)) if pad_len_col >= 0 else ((0, pad_len_row), (0, 0))
                
                input = np.pad(input, pad_width, mode='constant', constant_values=0)
                if pad_len_col < 0:
                    input = input[:, max_len]
                if return_mask:
                    mask = np.ones((max_len, max_len))
                    mask[-pad_len_row:, :] = 0
                    if pad_len_col > 0:
                        mask[:, -pad_len_col:] = 0

    else:
        raise ValueError("Input must be either 1D or 2D array.")

    return (input, mask) if return_mask else input