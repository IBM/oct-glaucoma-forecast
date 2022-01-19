import numpy as np
import torch


def binary_cross_entropy_with_logits(input, target):
    """Sigmoid Activation + Binary Cross Entropy

    @param input: torch.Tensor (size N)
    @param target: torch.Tensor (size N)
    @return loss: torch.Tensor (size N)
    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(
            target.size(), input.size()))

    return (torch.clamp(input, 0) - input * target
            + torch.log(1 + torch.exp(-torch.abs(input))))


def elbo_general(preds, gts, mu, logvar, annealing_factor=1., loss_type='ce'):
    """

    :param preds: list each element ais an  ndarry (N, n_t, c, H, W)
    :param gts: list  each element ais an  ndarry (N, n_t, c, H, W)
    :param mu:
    :param logvar:
    :param annealing_factor:
    :return:
    """

    n_modalities = len(preds)
    assert len(preds) == len(gts), 'Prediction list and ground truth list length does not match'
    assert loss_type in ['ce', 'se']

    MSE = 0
    for pred, gt in zip(preds, gts):

        assert all([pred is not None, gt is not None]) or all(
            [pred is None, gt is None]), ' either both gt and pred should be None or both of them shoud be provided'
        if (pred is not None and gt is not None):
            # reconstruction_loss = squared_error(gt.view(gt.shape[0], -1),
            #                                    pred.view(pred.shape[0], -1))

            if (loss_type == 'ce'):
                reconstruction_loss = torch.sum(
                    binary_cross_entropy_with_logits(pred.view(pred.shape[0], -1), gt.view(gt.shape[0], -1)), dim=1)
            else:
                pred = torch.sigmoid(pred) #from logit to sigmoid
                reconstruction_loss = torch.sum(squared_error(gt.view(gt.shape[0], -1),
                                                    pred.view(pred.shape[0], -1)), axis=1)

            MSE += reconstruction_loss

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    # NOTE: we use lambda_i = 1 for all i since each modality is roughly equal
    # ELBO = torch.mean(MSE / float(n_modalities) + annealing_factor * KLD)
    ELBO = torch.mean(MSE + annealing_factor * KLD)
    return ELBO, torch.mean(KLD)


def elbo_general_timeseries(preds, gts, mu, logvar, annealing_factor=1., loss_type='ce'):
    """

    :param preds: list each element ais an  ndarry (N, n_t, c, H, W)
    :param gts: list  each element ais an  ndarry (N, n_t, c, H, W)
    :param mu:
    :param logvar:
    :param annealing_factor:
    :return:
    """

    n_modalities = len(preds)

    assert len(preds) == len(gts), 'Prediction list and ground truth list length does not match'
    assert loss_type in ['ce', 'se']

    MSE = 0
    for pred, gt in zip(preds, gts):

        assert all([pred is not None, gt is not None]) or all(
            [pred is None, gt is None]), ' either both gt and pred should be None or both of them shoud be provided'
        if (pred is not None and gt is not None):
            # reconstruction_loss = squared_error(gt.view(gt.shape[0], -1),
            #                                    pred.view(pred.shape[0], -1))

            reconstruction_loss = mean_loss_across_time(gt, pred, loss_type=loss_type)

            MSE += reconstruction_loss

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    # NOTE: we use lambda_i = 1 for all i since each modality is roughly equal
    # ELBO = torch.mean(MSE / float(n_modalities) + annealing_factor * KLD)
    ELBO = torch.mean(MSE + annealing_factor * KLD)
    return ELBO, torch.mean(KLD)


def mean_loss_across_time(gt_ts, pred_ts, loss_type):
    N, N_t, c, H, W = gt_ts.shape
    assert gt_ts.shape == pred_ts.shape, 'shape and gt and prediction should be same'

    if (loss_type == 'ce'):
        if (len(gt_ts.shape) == 5):
            reconstruction_loss = 0
            for t in range(gt_ts.size(1)):
                gt = gt_ts[:, t, :, :, :]
                pred = pred_ts[:, t, :, :, :]
                reconstruction_loss += torch.sum(
                    binary_cross_entropy_with_logits(pred.view(pred.shape[0], -1), gt.view(gt.shape[0], -1)), dim=1)

    else:
        if (len(gt_ts.shape) == 5):
            reconstruction_loss = 0
            pred_ts = torch.sigmoid(pred_ts)

            for t in range(gt_ts.size(1)):
                gt = gt_ts[:, t, :, :, :]
                pred = pred_ts[:, t, :, :, :]
                reconstruction_loss += torch.sum(squared_error(gt.view(gt.shape[0], -1),
                                                              pred.view(pred.shape[0], -1)), dim=1)

    return reconstruction_loss# / N_t


def squared_error(x, y):
    """
    Squared error, element wise
    :param x: (B,d)
    :param y:  (B,d)
    :return: (B,)
    """
    d = x - y
    z = d * d

    z = z.view(z.shape[0], -1)
    return z


def mse_loss(x, y):
    """
    Mean squared error
    :param x: (B,d)
    :param y:  (B,d)
    :return: (B,)
    """
    d = x - y
    z = d * d

    z = z.view(z.shape[0], -1)
    return z.mean(dim=1)


def mse_loss(x, y):
    """
    Mean squared error
    :param x: (B,d)
    :param y:  (B,d)
    :return: (B,)
    """
    d = x - y
    z = d * d

    z = z.view(z.shape[0], -1)
    return z.mean(dim=1)


def mse_loss_masked(x, y, mask):
    """

    :param x: (B,d)
    :param y:  (B,d)
    :return: (B,)
    """
    d = x - y
    z = d * d

    return torch.sum(z * mask.float(), dim=1) / torch.sum(mask, dim=1)


def masked_mean(z, mask, dim):
    """

    :param z: (N, c,H,W)
    :param mask: (N,c,H,W)
    :param dim:
    :return:
    """

    return torch.sum(z * mask.float(), dim=dim) / torch.sum(mask, dim=dim)


def mae_loss(x, y, mask=None):
    """
    Mean absolute errors
    :param x: (B,d)
    :param y:  (B,d)
    :return: (B,)
    """

    x = x.view(x.shape[0], -1)
    y = y.view(y.shape[0], -1)

    d = torch.abs(x - y)
    d = d.view(d.shape[0], -1)

    if (mask is not None):
        mask = mask.view(mask.shape[0], -1)
        return masked_mean(d, mask, dim=1)
    else:
        return d.mean(dim=1)


def mae_globalmean(x, y, mask=None):
    """
    Mean absolute errors
    :param x: (B,d)
    :param y:  (B,d)
    :param mask: (B,d)
    :return: a scalar denoting the mae over all samples
    """
    x = x.view(x.shape[0], -1)
    y = y.view(y.shape[0], -1)

    if (mask is not None):
        mask = mask.view(mask.shape[0], -1)
        x = masked_mean(x, mask, dim=1)
        y = masked_mean(y, mask, dim=1)
    else:
        x = torch.mean(x, dim=1)
        y = torch.mean(y, dim=1)

    d = torch.abs(x - y)
    return d


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl
