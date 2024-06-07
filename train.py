import warnings
import time
warnings.filterwarnings("ignore")
import tensorflow as tf
import torch
from evaluater import Evaluator
from models import *
import torch.nn as nn
from DT_dataset import DT_dataset
from torch_geometric.loader import DataLoader
import os
from sklearn.model_selection import KFold
from featurize import get_featurizer
from utils import *
import re, json
from collections import defaultdict
from torch.optim.optimizer import Optimizer


class Trainer:
    def __init__(self, cfg, logger, scheduler=None, model_path="", seed=None):

        self.cfg = cfg
        self.logger = logger
        self.batch_size = cfg.engine['batch_size']
        self.task = cfg.task['class']
        self.evaluater = Evaluator(task=self.task, metrics=cfg['eval_metric'])
        self.device = torch.device(f"cuda:{cfg.engine['device'][0]}" if cfg.engine['device'] else "cpu")
        self.model_type = cfg.task.model['class']
        self.seed = seed

        dataset_path, dataset = load_data(load_from_tdc=True, cfg=self.cfg.dataset, seed=seed, task=self.task)

        param_feature = cfg.param_feature if 'param_feature' in cfg else {}

        # ! Select Model
        param_feature['name'] = cfg.task.model['class']
        param_feature['root'] = dataset_path

        # ! Different featurizer for different model
        featurizer = get_featurizer(**param_feature)
        num_worker = 0

        self.test_set = DT_dataset(root=dataset_path, featurizer=featurizer, data=dataset, split='test')
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=num_worker)
        self.train_set = DT_dataset(root=dataset_path, featurizer=featurizer, data=dataset, split='train')
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=num_worker)
        # ! Get dataset here
        if cfg.task['train'] == 'kfold':
            self.kfold_set = DT_dataset(root=dataset_path, featurizer=featurizer, data=dataset, split='kfold')
        else:
            self.valid_set = DT_dataset(root=dataset_path, featurizer=featurizer, data=dataset, split='valid')
            self.val_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False,
                                         num_workers=num_worker)

        if 'param' in cfg.task.model:
            model_params = cfg.task.model.param
        else:
            model_params = {}

        if cfg.task.model['class'] in globals():
            if self.task == 'classification':
                self.loss_func = nn.CrossEntropyLoss()
                # self.loss_func = nn.BCELoss()
                model_params['n_output'] = 2
            elif self.task == 'regression':
                self.loss_func = nn.MSELoss()
                model_params['n_output'] = 1
        else:
            print('wrong model')
        model_params['feat_root'] = os.path.join(dataset_path, featurizer.feat_name)
        # data/DAVIS/classification_random_42/processed/GeNNius_train.pt
        model_params['train_data_path'] = os.path.join(dataset_path, "processed", featurizer.feat_name + "_train.pt")
        self.model_params = model_params
        self.model = globals()[cfg.task.model['class']](**model_params)
        self.model = self.model.to(self.device)

        self.logger.info("Model size: {:.2f} MB".format(model_size_in_bytes(self.model) / (1024 ** 2)))
        self.score_metric = cfg['score_metric']
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.optimizer['lr'])
        self.scheduler = scheduler
        self.epochs = cfg.train['num_epoch']
        self.model_path = model_path

    def train_epoch(self, epoch, dataloader):
        """
        One epoch training
        :param epoch:
        :return:
        """

        self.model.train()

        loss_total = 0.0
        for batch_idx, batch in enumerate(dataloader):
            if self.device.type == "cuda":
                batch = cuda(batch, device=self.device)
            self.optimizer.zero_grad()

            output = self.model(batch)
            target = batch[0].y

            loss = loss_cal(self.loss_func, output, target, type=self.task)

            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()

            if batch_idx % 20 == 0:
                self.logger.info('Train epoch: {} [{:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                                   100. * batch_idx / len(dataloader),
                                                                                   loss_total / 20))
        if self.scheduler:
            self.scheduler.step()
        return loss_total

    @torch.no_grad()
    def evaluate(self, dataloader):
        """
        Evaluate the model.

        Parameters:
            split (str): split to evaluate. Can be ``train``, ``valid`` or ``test``.
            log (bool, optional): log metrics or not

        Returns:
            dict: metrics
            :param dataloader:
        """

        model = self.model

        model.eval()
        preds = []
        targets = []
        for batch in dataloader:
            if self.device.type == "cuda":
                batch = cuda(batch, device=self.device)

            pred = model(batch)
            target = batch[0].y

            preds.append(pred)
            targets.append(target)

        pred = torch.cat(preds)
        target = torch.cat(targets)

        metric = eval_func(self.evaluater, target.cpu(), pred.cpu(), type=self.task)

        return metric

    def train_test(self):
        result = []
        if self.score_metric == 'mse':
            best_score = float('inf')
        else:
            best_score = float('-inf')

        best_epoch = -1

        for epoch in range(self.epochs):
            self.train_epoch(epoch, self.train_loader)
            metric = self.evaluate(self.test_loader)
            self.logger.info(metric)
            result.append(metric)
            if self.score_metric == 'mse':
                if best_score > metric[self.score_metric]:
                    best_epoch = epoch
                    best_score = metric[self.score_metric]
                    torch.save(self.model.state_dict(), os.path.join(self.model_path, 'best_model.pth'))
            else:
                if best_score < metric[self.score_metric]:
                    best_epoch = epoch
                    best_score = metric[self.score_metric]
                    torch.save(self.model.state_dict(), os.path.join(self.model_path, 'best_model.pth'))

        self.logger.info(result)
        self.logger.info('best epoch:{}'.format(best_epoch))
        self.logger.info('best score:{}'.format(best_score))

    def K_fold_train(self, n_splits=5):

        # data
        results = {}
        result = []
        num_worker = 0
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        loss_dict = {}
        for fold, (train_ids, val_ids) in enumerate(kfold.split(self.kfold_set)):
            if fold not in loss_dict:
                loss_dict[fold] = []

            self.model = globals()[self.cfg.task.model['class']](**self.model_params)
            self.model = self.model.to(self.device)

            if self.score_metric == 'mse':
                best_score = float('inf')
            else:
                best_score = float('-inf')

            best_epoch = -1

            self.logger.info(f'Fold {fold}')
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
            train_loader = DataLoader(self.kfold_set, batch_size=self.batch_size, sampler=train_subsampler,
                                      num_workers=num_worker)
            val_loader = DataLoader(self.kfold_set, batch_size=self.batch_size, sampler=val_subsampler,
                                    num_workers=num_worker)

            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=self.cfg.optimizer['lr'])

            # for epoch in tqdm(range(self.epochs)):
            for epoch in range(self.epochs):
                loss = self.train_epoch(epoch, train_loader)
                metric = self.evaluate(val_loader)
                self.logger.info(metric)
                result.append(metric)
                # time.sleep(0.003)
                if self.score_metric == 'mse':
                    if best_score > metric[self.score_metric]:
                        best_epoch = epoch
                        best_score = metric[self.score_metric]
                        torch.save(self.model.state_dict(),
                                   os.path.join(self.model_path, f'best_fold_{fold}_model.pth'))
                        metric_test = self.evaluate(self.test_loader)
                else:
                    if best_score < metric[self.score_metric]:
                        best_epoch = epoch
                        best_score = metric[self.score_metric]
                        torch.save(self.model.state_dict(),
                                   os.path.join(self.model_path, f'best_fold_{fold}_model.pth'))
                        metric_test = self.evaluate(self.test_loader)
                loss_dict[fold].append(loss)
            # torch.save(self.model.state_dict(), os.path.join(self.model_path, 'best_model.pth'))
            # metric = self.evaluate(self.test_loader)
            results[fold] = metric_test
            self.logger.info('Test for fold {0}:{1}'.format(fold, metric_test))
            self.logger.info('--------------------------------')
            df_loss = pd.DataFrame(loss_dict)
            df_loss.to_csv(os.path.join(self.model_path, f'loss_dict{fold}.csv'), index=False)
            
        self.logger.info(results)
        df_loss = pd.DataFrame(loss_dict)
        df_loss.to_csv(os.path.join(self.model_path, 'loss_dict.csv'), index=False)

        df_results = pd.DataFrame(results)
        if self.task == 'regression':
            df_results.iloc[-1] = df_results.iloc[-1].apply(lambda x: ''.join(re.findall(r'[0-9.]+', str(x)[:40])))
            df_results = df_results.map(lambda x: pd.to_numeric(x, errors='coerce'))
        df_results['mean'] = df_results.mean(axis=1)
        df_results.to_csv(os.path.join(self.model_path, 'results.csv'), index=False)

    def mem_speed_bench(self):
        total_epoch = 6
        start_count = 1

        torch.cuda.empty_cache()
        model_opt_usage = get_memory_usage(self.device, False)
        usage_dict = {
            "model_opt_usage": model_opt_usage / MB,
            "data_mem": [],
            "act_mem": [],
            "peak_mem": [],
            "duration": [],
        }
        self.logger.info(
            "model + optimizer only, mem: %.2f MB"
            % (usage_dict["model_opt_usage"])
        )
        torch.cuda.reset_max_memory_allocated(self.device)
        epoch_start_time = time.time()
        torch.cuda.synchronize()
        for epoch in range(total_epoch):
            for batch_idx, batch in enumerate(self.train_loader):
                torch.cuda.synchronize()
                iter_start_time = time.time()
                if self.device.type == "cuda":
                    batch = cuda(batch, device=self.device)
                init_mem = get_memory_usage(self.device, False)
                data_mem = init_mem - usage_dict["model_opt_usage"]
                if epoch >= start_count:
                    usage_dict["data_mem"].append(data_mem / MB)
                self.logger.info("data mem: %.2f MB" % (data_mem / MB))
                self.optimizer.zero_grad()
                output = self.model(batch)
                target = batch[0].y
                loss = loss_cal(self.loss_func, output, target, type=self.task)
                loss = loss.mean()
                before_backward = get_memory_usage(self.device, False)
                act_mem = before_backward - init_mem - compute_tensor_bytes([loss, output])
                if epoch >= start_count:
                    usage_dict["act_mem"].append(act_mem / MB)
                self.logger.info("act mem: %.2f MB" % (act_mem / MB))
                loss.backward()
                self.optimizer.step()
                torch.cuda.synchronize()
                iter_end_time = time.time()
                duration = iter_end_time - iter_start_time
                self.logger.info("duration: %.4f sec" % duration)
                if epoch >= start_count:
                    usage_dict["duration"].append(duration)
                peak_usage = torch.cuda.max_memory_allocated(self.device)
                if epoch >= start_count:
                    usage_dict["peak_mem"].append(peak_usage / MB)
                self.logger.info(f"peak mem usage: {peak_usage / MB}")
        usage_dict['sum_duration_epoch'] = np.array(usage_dict["duration"]).sum() / (total_epoch - start_count)
        usage_dict['peak_mem_mean'] = np.array(usage_dict["peak_mem"]).mean()
        usage_dict['data_mem_mean'] = np.array(usage_dict["data_mem"]).mean()
        usage_dict['act_mem_mean'] = np.array(usage_dict["act_mem"]).mean()
        with open(os.path.join(self.model_path, 'mem_speed_log.json'), "w") as fp:
            info_dict = {**self.cfg, **usage_dict}
            json.dump(info_dict, fp)