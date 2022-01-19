import numpy as np
import os
import torch


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()
        self.loss_history = []

    def reset(self):
        self.val = None
        self.avg = 0
        self.loss_history = []

    def update(self, val,  accumulate=True, smooth=True):
        """

        :param val:
        :param accumulate: weather to add to loss history
        :param smooth: if True, adds smoothed value on loss history otherwise adds val
        :return:
        """
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

        if(accumulate):
            self.loss_history.append(self.avg) if smooth else self.loss_history.append(val)


    def save_csv(self, file='loss_history.csv'):
        np.savetxt(file, self.loss_history, delimiter=',')


def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z


def save_model(model, dir, prefix, epoch=None):
    # save the models
    suffix = 'epoch'+str(epoch) if epoch is not  None else ''

    model_file = prefix +suffix+ '.pt'
    if (not os.path.exists(dir)):
        os.mkdir(dir)
    model_file = os.path.join(dir, model_file)
    print('saving model: ', model_file)
    torch.save(model.state_dict(), model_file)
    return model_file



from sklearn.model_selection import GroupShuffleSplit

def Splitrandom (ratios, seed=None, group_func=None):

    def get_group(data):
        if (group_func is not None):
            groups = [group_func(d) for d in data]
        else:
            groups = np.arange(len(data))
        return groups

    def slice(data, idx):
        return [data[id] for id in idx]


    def f(data):

        def group_two(data, ratio_pair):
            groups = get_group(data)
            gss = GroupShuffleSplit(n_splits=1, train_size=ratio_pair[0], test_size=ratio_pair[1], random_state=seed)
            train_idx, test_idx  = next(gss.split(data, groups=groups))
            return slice(data,train_idx) , slice(data,test_idx)

        ratio_new = (ratios[0], np.sum(ratios[1:]))
        train, valtest = group_two(data, ratio_new)
        ratio_new =(ratios[1]/np.sum(ratios[1:]),ratios[2]/np.sum(ratios[1:]))
        val, test = group_two(valtest, ratio_new)
        return train, val, test


    return f



def select_modalities(combs, modalities_flag):
    """

    :param combs:  example [[1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    :param modalities_flag: e.g [1,1,0]
    :return: e.g [ [1, 1, 0], [1, 0, 0], [0, 1, 0]]
    """
    combs = [[a * b for a, b in zip(c, modalities_flag)] for c in combs]
    # remove duplicate combinations
    combs = list(map(list, set(map(tuple, combs))))
    combs = [c for c in combs if np.sum(c) > 0]
    combs = sorted(combs, key=lambda x: np.sum(x), reverse=True)
    return combs


def subsample(xx, cc):

    assert len(xx) == len(cc), 'input list  size does not match with the flag  length'
    f = lambda x, c: x if c == 1 else None
    return [f(xi, ci) for xi, ci in zip(xx, cc)]

