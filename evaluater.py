from metrics import *

def range_logAUC(true_y, predicted_score, FPR_range=(0.001, 0.1)):
    """
    Author: Yunchao "Lance" Liu (lanceknight26@gmail.com)
    Calculate logAUC in a certain FPR range (default range: [0.001, 0.1]).
    This was used by previous methods [1] and the reason is that only a
    small percentage of samples can be selected for experimental tests in
    consideration of cost. This means only molecules with very high
    predicted score can be worth testing, i.e., the decision
    threshold is high. And the high decision threshold corresponds to the
    left side of the ROC curve, i.e., those FPRs with small values. Also,
    because the threshold cannot be predetermined, the area under the curve
    is used to consolidate all possible thresholds within a certain FPR
    range. Finally, the logarithm is used to bias smaller FPRs. The higher
    the logAUC[0.001, 0.1], the better the performance.

    A perfect classifer gets a logAUC[0.001, 0.1] ) of 1, while a random
    classifer gets a logAUC[0.001, 0.1] ) of around 0.0215 (See [2])

    References:
    [1] Mysinger, M.M. and B.K. Shoichet, Rapid Context-Dependent Ligand
    Desolvation in Molecular Docking. Journal of Chemical Information and
    Modeling, 2010. 50(9): p. 1561-1573.
    [2] Liu, Yunchao, et al. "Interpretable Chirality-Aware Graph Neural
    Network for Quantitative Structure Activity Relationship Modeling in
    Drug Discovery." bioRxiv (2022).
    :param true_y: numpy array of the ground truth. Values are either 0
    (inactive) or 1(active).
    :param predicted_score: numpy array of the predicted score (The
    score does not have to be between 0 and 1)
    :param FPR_range: the range for calculating the logAUC formated in
    (x, y) with x being the lower bound and y being the upper bound
    :return: a numpy array of logAUC of size [1,1]
    """

    # FPR range validity check
    if FPR_range == None:
        raise Exception("FPR range cannot be None")
    lower_bound = FPR_range[0]
    upper_bound = FPR_range[1]
    if lower_bound >= upper_bound:
        raise Exception("FPR upper_bound must be greater than lower_bound")

    fpr, tpr, thresholds = roc_curve(true_y, predicted_score, pos_label=1)

    tpr = np.append(tpr, np.interp([lower_bound, upper_bound], fpr, tpr))
    fpr = np.append(fpr, [lower_bound, upper_bound])

    # Sort both x-, y-coordinates array
    tpr = np.sort(tpr)
    fpr = np.sort(fpr)

    # Get the data points' coordinates. log_fpr is the x coordinate, tpr is the y coordinate.
    log_fpr = np.log10(fpr)
    x = log_fpr
    y = tpr
    lower_bound = np.log10(lower_bound)
    upper_bound = np.log10(upper_bound)

    # Get the index of the lower and upper bounds
    lower_bound_idx = np.where(x == lower_bound)[-1][-1]
    upper_bound_idx = np.where(x == upper_bound)[-1][-1]

    # Create a new array trimmed at the lower and upper bound
    trim_x = x[lower_bound_idx: upper_bound_idx + 1]
    trim_y = y[lower_bound_idx: upper_bound_idx + 1]

    area = auc(trim_x, trim_y) / (upper_bound - lower_bound)
    return area


class Evaluator:
    """evaluator to evaluate predictions

    Args:
            name (str): the name of the evaluator function
    """

    def __init__(self, task='regression', metrics=['mse']):
        """create an evaluate object"""
        self.task = task
        self.reg_metric = ['mse', 'rmse', 'mae', 'r2', 'pcc', 'spearman', 'ci']
        self.cla_metric = ['roc-auc', 'pr-auc', 'range_logAUC', 'accuracy', 'precision',
                           'recall', 'f1', 'rp@k', 'pr@k']
        if self.task == 'regression':
            assert all(item in self.reg_metric for item in metrics), "Not all metrics exist"
        elif self.task == 'classification':
            assert all(item in self.cla_metric for item in metrics), "Not all metrics exist"
        self.eval_funcs = []
        self.metrics = metrics
        for metric in self.metrics:
            self.eval_funcs.append(self.assign_evaluator(metric))

    def assign_evaluator(self, metric_name):
        """obtain evaluator function given the evaluator name"""
        if metric_name == "roc-auc":
            return roc_auc_score
        elif metric_name == "f1":
            return f1_score
        elif metric_name == "pr-auc":
            return average_precision_score
        elif metric_name == "rp@k":
            return recall_at_precision_k
        elif metric_name == "pr@k":
            return precision_at_recall_k
        elif metric_name == "precision":
            return precision_score
        elif metric_name == "recall":
            return recall_score
        elif metric_name == "accuracy":
            return accuracy_score
        elif metric_name == "mse":
            return mean_squared_error
        elif metric_name == "rmse":
            return rmse
        elif metric_name == "mae":
            return mean_absolute_error
        elif metric_name == "r2":
            return r2_score
        elif metric_name == "pcc":
            return pcc
        elif metric_name == 'ci':
            return get_cindex
        elif metric_name == "spearman":
            try:
                from scipy import stats
            except:
                ImportError("Please install scipy by 'pip install scipy'! ")
            return stats.spearmanr
        elif metric_name == "range_logAUC":
            return range_logAUC

    def __call__(self, *args, **kwargs):
        """call the evaluator function on targets and predictions

        Args:
            *args: targets, predictions, and other information
            **kwargs: other auxilliary inputs for some evaluators

        Returns:
            dict: the evaluator output
        """

        y_true = kwargs["y_true"] if "y_true" in kwargs else args[0]
        y_pred = kwargs["y_pred"] if "y_pred" in kwargs else args[1]
        if len(args) <= 2 and "threshold" not in kwargs:
            threshold = 0.5
        else:
            threshold = kwargs["threshold"] if "threshold" in kwargs else args[2]
        out_dict = {}
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        for metric_name, eval_func in list(zip(self.metrics, self.eval_funcs)):
            if metric_name in ["precision", "recall", "f1", "accuracy"]:
                y_pred = [1 if i > threshold else 0 for i in y_pred]
            elif metric_name in ["rp@k", "pr@k"]:
                out_dict[metric_name] = eval_func(y_true, y_pred, threshold=threshold)
                continue
            elif metric_name == "Spearman":
                out_dict[metric_name] = eval_func(y_true, y_pred)[0]
                continue
            out_dict[metric_name] = eval_func(y_true, y_pred)
        return out_dict


if __name__ == '__main__':
    evaluator = Evaluator(task='regression', metrics=['mse', 'rmse', 'mae', 'r2', 'pcc', 'spearman', 'ci'])
    y_pred = [0.1, 0.2, 0.4, 0.7, 0.8]
    y_true = [0.1, 0.24, 0.5, 0.66, 0.89]
    score = evaluator(y_true, y_pred)
    print(score)
