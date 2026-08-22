import math
import warnings
import time
warnings.filterwarnings("ignore")
# import tensorflow as tf  # unused, and importing it reserves GPU memory at startup
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
import json
from collections import defaultdict
from torch.optim.optimizer import Optimizer
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

class Trainer:
    # Keys build_optimizer interprets itself; the rest are forwarded to the optimizer constructor.
    OPTIMIZER_KEYS = {'class', 'lr', 'weight_decay', 'no_decay_on_bias',
                      'scheduler', 'step_size', 'gamma', 'warmup_steps',
                      'lookahead', 'lookahead_k', 'lookahead_alpha',
                      'clip_grad_norm'}

    def __init__(self, cfg, logger, model_path="", seed=None):

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
        self.epochs = cfg.train['num_epoch']
        self.model_path = model_path
        self.accumulation_steps = max(1, cfg.engine.get('accumulation_steps', 1))
        # 0 disables clipping, which is every recipe except BridgeDPI's
        self.clip_grad_norm = cfg.optimizer.get('clip_grad_norm', 0)
        if self.accumulation_steps > 1:
            self.logger.info("Gradient accumulation: {} x batch {} = effective batch {}".format(
                self.accumulation_steps, self.batch_size, self.accumulation_steps * self.batch_size))
        self.optimizer, self.scheduler = self.build_optimizer(self.optimizer_steps(self.train_loader))
        self.logger.info("Optimizer: {}, scheduler: {} ({})".format(
            type(self.optimizer).__name__,
            type(self.scheduler).__name__ if self.scheduler else 'none',
            'per ' + self.scheduler_interval if self.scheduler else 'no schedule'))

    def optimizer_steps(self, dataloader):
        """Optimizer steps per epoch, which is fewer than batches under gradient accumulation."""
        return math.ceil(len(dataloader) / self.accumulation_steps)

    def closes_group(self, batch_idx, n_batches):
        """Whether this batch closes its accumulation group.

        The last group of an epoch is flushed early, so it can hold fewer batches.
        """
        return (batch_idx + 1) % self.accumulation_steps == 0 or batch_idx + 1 == n_batches

    def group_sample_counts(self, dataloader):
        """Samples held by each accumulation group, indexed by group.

        Every batch but the last one is full, so the totals are known before the epoch starts
        and the divisor can be folded into the loss rather than applied to the gradients.
        """
        n_batches = len(dataloader)
        batch_size = dataloader.batch_size
        total = n_batches * batch_size if dataloader.drop_last else len(dataloader.sampler)
        span = self.accumulation_steps * batch_size
        return [min(span, total - start * batch_size)
                for start in range(0, n_batches, self.accumulation_steps)]

    def optimizer_step(self):
        """Clip, step, and advance a step-interval scheduler; shared by training and the bench."""
        if self.clip_grad_norm:
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     max_norm=self.clip_grad_norm, norm_type=2)
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.scheduler_interval == 'step':
            self.scheduler.step()

    def parameter_groups(self):
        """Split parameters so weight decay skips biases, keyed on the name as upstream does."""
        optimizer_config = self.cfg.optimizer
        weight_decay = optimizer_config.get('weight_decay', 0)
        trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        if not optimizer_config.get('no_decay_on_bias', False):
            return [{'params': [p for _, p in trainable], 'weight_decay': weight_decay}]

        decayed = [p for n, p in trainable if 'bias' not in n]
        undecayed = [p for n, p in trainable if 'bias' in n]
        self.logger.info("weight_decay={} on {} tensors, 0 on {} bias tensors".format(
            weight_decay, len(decayed), len(undecayed)))
        return [{'params': decayed, 'weight_decay': weight_decay},
                {'params': undecayed, 'weight_decay': 0.0}]

    def build_scheduler(self, optimizer, steps_per_epoch):
        """Build cfg.optimizer.scheduler, returning (scheduler, interval).

        The interval is the clock the schedule is written against: 'epoch' for StepLR, whose
        step_size counts epochs, and 'step' for the cosine schedule, whose warmup and total
        count optimizer steps. Under gradient accumulation a step spans several batches.
        """
        optimizer_config = self.cfg.optimizer
        kind = optimizer_config.get('scheduler', False)
        if not kind:
            return None, None
        if kind is True or kind == 'step':
            return lr_scheduler.StepLR(
                optimizer,
                step_size=optimizer_config.get('step_size', 10),
                gamma=optimizer_config.get('gamma', 0.5),
            ), 'epoch'
        if kind == 'cosine_warmup':
            warmup = optimizer_config.get('warmup_steps', 10)
            total = max(1, steps_per_epoch * self.cfg.train['num_epoch'])

            def lr_scale(step):
                if step < warmup:
                    return step / max(1, warmup)
                progress = (step - warmup) / max(1, total - warmup)
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

            return lr_scheduler.LambdaLR(optimizer, lr_scale), 'step'
        raise ValueError(f"Unknown optimizer.scheduler {kind!r} (expected false, 'step' or 'cosine_warmup')")

    def build_optimizer(self, steps_per_epoch=1):
        """Build a fresh optimizer and scheduler bound to the current self.model."""
        optimizer_config = self.cfg.optimizer
        optimizer_cls = getattr(torch.optim, optimizer_config.get('class', 'Adam'))
        extra = {k: v for k, v in optimizer_config.items() if k not in self.OPTIMIZER_KEYS}
        optimizer = optimizer_cls(self.parameter_groups(), lr=optimizer_config['lr'], **extra)
        if optimizer_config.get('lookahead', False):
            optimizer = Lookahead(optimizer,
                                  k=optimizer_config.get('lookahead_k', 5),
                                  alpha=optimizer_config.get('lookahead_alpha', 0.5))
        scheduler, self.scheduler_interval = self.build_scheduler(optimizer, steps_per_epoch)
        return optimizer, scheduler

    def train_epoch(self, epoch, dataloader):
        """
        One epoch training
        :param epoch:
        :return:
        """

        self.model.train()

        loss_total = 0.0
        epoch_samples = 0
        n_batches = len(dataloader)
        pbar = tqdm(total=n_batches, desc=f'Epoch {epoch}/{self.epochs}', unit='b')
        group_samples = self.group_sample_counts(dataloader)
        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            if self.device.type == "cuda":
                batch = cuda(batch, device=self.device)

            output = self.model(batch)
            target = batch[0].y

            loss = loss_cal(self.loss_func, output, target, type=self.task)
            n_samples = target.numel()
            # A full group divides by exactly accumulation_steps, so only a partial trailing batch sees a different weight than before
            (loss / (group_samples[batch_idx // self.accumulation_steps] / n_samples)).backward()
            if self.closes_group(batch_idx, n_batches):
                self.optimizer_step()
            loss_total += loss.item() * n_samples
            epoch_samples += n_samples

            pbar.set_postfix(loss=loss.item())
            pbar.update(1)
        pbar.close()

        if self.scheduler_interval == 'epoch':
            self.scheduler.step()
        return loss_total / epoch_samples

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

    def K_fold_train(self, n_splits=5, early_stop_patience=50):

        # data
        results = {}
        result = []
        exit_epoch_info = {}  # epoch at which each fold stopped
        best_epoch_info = {}
        num_worker = 0
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        loss_dict = {}
        lower_is_better = self.score_metric in ('mse', 'rmse', 'mae')
        for fold, (train_ids, val_ids) in enumerate(kfold.split(self.kfold_set)):
            if fold not in loss_dict:
                loss_dict[fold] = []

            # Released before the next model is built, otherwise the optimizer keeps the
            # previous fold's parameters and state alive alongside it
            self.optimizer = self.scheduler = None
            self.model = globals()[self.cfg.task.model['class']](**self.model_params)
            self.model = self.model.to(self.device)

            best_score = float('inf') if lower_is_better else float('-inf')
            best_epoch = -1

            self.logger.info(f'Fold {fold}')
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
            train_loader = DataLoader(self.kfold_set, batch_size=self.batch_size, sampler=train_subsampler,
                                      num_workers=num_worker)
            val_loader = DataLoader(self.kfold_set, batch_size=self.batch_size, sampler=val_subsampler,
                                    num_workers=num_worker)

            self.optimizer, self.scheduler = self.build_optimizer(self.optimizer_steps(train_loader))
            epoch = -1
            metric_test = None
            for epoch in range(self.epochs):
                loss = self.train_epoch(epoch, train_loader)
                metric = self.evaluate(val_loader)
                self.logger.info(metric)
                result.append(metric)
                score = metric[self.score_metric]
                improved = (score < best_score) if lower_is_better else (score > best_score)
                if improved:
                    best_epoch = epoch
                    best_score = score
                    torch.save(self.model.state_dict(),
                               os.path.join(self.model_path, f'best_fold_{fold}_model.pth'))
                    metric_test = self.evaluate(self.test_loader)
                loss_dict[fold].append(loss)
                if epoch - best_epoch >= early_stop_patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch} with best epoch {best_epoch}.")
                    break
            # epoch holds the last executed epoch, whether the loop broke early or ran to completion
            exit_epoch_info[fold] = epoch
            best_epoch_info[fold] = best_epoch
            results[fold] = metric_test
            self.logger.info('Test for fold {0}:{1}'.format(fold, metric_test))
            self.logger.info('--------------------------------')
            df_loss = pd.DataFrame({fold: pd.Series(loss_list) for fold, loss_list in loss_dict.items()})
            df_loss.to_csv(os.path.join(self.model_path, f'loss_dict{fold}.csv'), index=False)
            
        self.logger.info(results)
        df_loss = pd.DataFrame({fold: pd.Series(loss_list) for fold, loss_list in loss_dict.items()})
        df_loss.to_csv(os.path.join(self.model_path, 'loss_dict.csv'), index=False)

        df_results = pd.DataFrame(results).apply(pd.to_numeric, errors='coerce')
        df_results.loc['exit_epoch'] = pd.Series(exit_epoch_info)
        df_results.loc['best_epoch'] = pd.Series(best_epoch_info)

        fold_columns = list(df_results.columns)
        df_results['mean'] = df_results[fold_columns].mean(axis=1)
        df_results['std'] = df_results[fold_columns].std(axis=1, ddof=1)
        # The index holds the metric names, so it has to be written out
        df_results.to_csv(os.path.join(self.model_path, 'results.csv'), index=True)

    def warmup_optimizer_state(self):
        """Run one full step so the optimizer's lazily allocated state exists before measuring.

        Adam-family state and Lookahead's slow weights are only created on the first step,
        so model_opt_usage would otherwise report the parameters alone.
        """
        batch = next(iter(self.train_loader))
        if self.device.type == "cuda":
            batch = cuda(batch, device=self.device)
        output = self.model(batch)
        loss = loss_cal(self.loss_func, output, batch[0].y, type=self.task)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        del batch, output, loss

    def mem_speed_bench(self):
        total_epoch = 6
        start_count = 1


        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.warmup_optimizer_state()
        torch.cuda.empty_cache()
        model_opt_usage = get_memory_usage(self.device, False)
        usage_dict = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "model_opt_usage": model_opt_usage / MB,
            "data_mem": [],
            "act_mem": [],
            "peak_mem": [],
            "duration": [],
        }
        self.logger.info(
            "parameters: %.2f M total, %.2f M trainable"
            % (total_params / 1e6, trainable_params / 1e6)
        )
        self.logger.info(
            "model + optimizer only, mem: %.2f MB"
            % (usage_dict["model_opt_usage"])
        )
        epoch_start_time = time.time()
        torch.cuda.synchronize()
        n_batches = len(self.train_loader)
        # Mirror train_epoch so the reported duration covers the recipe actually trained with
        group_samples = self.group_sample_counts(self.train_loader)
        self.optimizer.zero_grad()
        for epoch in range(total_epoch):
            for batch_idx, batch in enumerate(self.train_loader):
                torch.cuda.synchronize()
                # Without a reset per iteration, max_memory_allocated stays a running maximum
                # and its mean would drift with total_epoch
                torch.cuda.reset_max_memory_allocated(self.device)
                iter_start_time = time.time()
                mem_before_batch = get_memory_usage(self.device, False)
                if self.device.type == "cuda":
                    batch = cuda(batch, device=self.device)
                init_mem = get_memory_usage(self.device, False)
                # Spans the transfer alone, so lingering gradients stay out of it
                data_mem = init_mem - mem_before_batch
                if epoch >= start_count:
                    usage_dict["data_mem"].append(data_mem / MB)
                self.logger.info("data mem: %.2f MB" % (data_mem / MB))
                output = self.model(batch)
                target = batch[0].y
                loss = loss_cal(self.loss_func, output, target, type=self.task)
                loss = loss.mean()
                before_backward = get_memory_usage(self.device, False)
                act_mem = before_backward - init_mem - compute_tensor_bytes([loss, output])
                if epoch >= start_count:
                    usage_dict["act_mem"].append(act_mem / MB)
                self.logger.info("act mem: %.2f MB" % (act_mem / MB))
                n_samples = target.numel()
                (loss / (group_samples[batch_idx // self.accumulation_steps] / n_samples)).backward()
                if self.closes_group(batch_idx, n_batches):
                    self.optimizer_step()
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
            if self.scheduler_interval == 'epoch':
                self.scheduler.step()
        usage_dict['sum_duration_epoch'] = np.array(usage_dict["duration"]).sum() / (total_epoch - start_count)
        usage_dict['peak_mem_mean'] = np.array(usage_dict["peak_mem"]).mean()
        usage_dict['peak_mem_max'] = np.array(usage_dict["peak_mem"]).max()
        usage_dict['data_mem_mean'] = np.array(usage_dict["data_mem"]).mean()
        usage_dict['act_mem_mean'] = np.array(usage_dict["act_mem"]).mean()
        with open(os.path.join(self.model_path, 'mem_speed_log.json'), "w") as fp:
            info_dict = {**self.cfg, **usage_dict}
            json.dump(info_dict, fp)