def denormalize_range(Y, source_range, target_range):
    xmin = source_range[0]
    xmax = source_range[1]
    ratio = (target_range[1] - target_range[0]) * 1.0 / (xmax - xmin)
    # print 'ratio ', ratio
    Y = source_range[0] + (1/ratio)  * (Y - target_range[0])
    return Y


def normalize_range(X, source_range, target_range):
    xmin = source_range[0]
    xmax = source_range[1]
    ratio = (target_range[1] - target_range[0]) * 1.0 / (xmax - xmin)
    # print 'ratio ', ratio
    X = target_range[0] + ratio * (X - xmin)
    return X
