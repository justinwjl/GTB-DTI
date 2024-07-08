from torch_geometric.data import InMemoryDataset, Batch, Data
import torch
import os
from itertools import repeat
from torch_geometric.data.in_memory_dataset import nested_iter
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class DT_dataset(InMemoryDataset):

    def __init__(self, root='data', featurizer=None, data=None, split='train', transform=None, pre_transform=None):

        super(DT_dataset, self).__init__(root, transform, pre_transform)

        self.data = data
        self.data_type_list = ['train', 'valid', 'test', 'kfold']
        self.data_type_dict = {'train': 0, 'valid': 1, 'test': 2, 'kfold': 3}
        self.featurizer = featurizer
        self.feat_name = self.featurizer.feat_name
        self.split = split
        if os.path.isfile(self.processed_paths[self.data_type_dict[split]]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[self.data_type_dict[split]]))
            self.data_list, self.slices_list = torch.load(self.processed_paths[self.data_type_dict[split]])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(
                self.processed_paths[self.data_type_dict[split]]))
            self.process()
            self.data_list, self.slices_list = torch.load(self.processed_paths[self.data_type_dict[split]])

    def __getitem__(self, idx):

        data_list = []

        # for data_item, slice_item in list(zip(self.data_list[self.split], self.slices_list[self.split])):
        for data_item, slice_item in list(zip(self.data_list, self.slices_list)):
            self.data = data_item
            self.slices = slice_item
            data = self.data.__class__()

            if hasattr(self.data, '__num_nodes__'):
                data.num_nodes = self.data.__num_nodes__[idx]

            for key in self.data.keys():
                item, slices = self.data[key], self.slices[key]
                start, end = slices[idx].item(), slices[idx + 1].item()
                if torch.is_tensor(item):
                    s = list(repeat(slice(None), item.dim()))
                    s[self.data.__cat_dim__(key, item)] = slice(start, end)
                elif start + 1 == end:
                    s = slices[start]
                else:
                    s = slice(start, end)

                data[key] = item[s]
            data_list.append(data)

        return data_list

    @property
    def raw_file_names(self):
        return ['train.csv', 'test.csv', 'val.csv']

    @property
    def processed_file_names(self):
        tmp = []
        for data_type in self.data_type_list:
            tmp.append(self.feat_name + '_' + data_type + '.pt')
        return tmp

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self):

        train_len, valid_len, test_len, kfold_len = len(self.data['train'][0]), len(self.data['valid'][0]), len(
            self.data['test'][0]), len(self.data['kfold'][0])
        # orignal data
        pd_train = pd.DataFrame(self.data['kfold'])
        pd_test = pd.DataFrame(self.data['test'])
        data_total = pd.concat([pd_train, pd_test], axis=1)
        smiles, sequences, labels = data_total.iloc[0].to_list(), data_total.iloc[1].to_list(), data_total.iloc[
            2].to_list()

        data_list = self.featurizer.data_input(smiles, sequences, labels)

        max_graphs = len(data_list[0])
        # Create a dictionary of lists of length max_graphs
        batched_data = {}
        for item in ['train', 'valid', 'test', 'kfold']:
            batched_data[item] = [[] for _ in range(max_graphs)]

        # Two layers of [], each item in data_list is an object of key-value pairs
        for idx, item in enumerate(data_list):
            for i in range(len(item)):
                if idx < train_len:
                    batched_data['train'][i].append(item[i])
                    batched_data['kfold'][i].append(item[i])
                elif idx < train_len + valid_len:
                    batched_data['valid'][i].append(item[i])
                    batched_data['kfold'][i].append(item[i])
                else:
                    batched_data['test'][i].append(item[i])

        if self.transform is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]


        collated_data = {}
        for data_type in self.data_type_list:
            collated_data[data_type] = [self.collate(sample) for sample in batched_data[data_type]]

        # Initialize the dictionary for saved data and slices
        saved_data = {}
        saved_slice = {}

        for idx, data_type in enumerate(self.data_type_list):

            # For each type of data: train/test/valid/kfold
            saved_data[data_type] = []
            saved_slice[data_type] = []

            for data_tmp, slice_tmp in collated_data[data_type]:
                saved_data[data_type].append(data_tmp)
                saved_slice[data_type].append(slice_tmp)

            torch.save((saved_data[data_type], saved_slice[data_type]), self.processed_paths[idx])

    def __repr__(self):
        return '{}()'.format(self.name)

    def len(self):
        if self.slices_list is None:
            return 1
        for _, value in nested_iter(self.slices_list[0]):
            return len(value) - 1
        return 0
