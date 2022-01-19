# pixelwise linear regression

import numpy as np
from sklearn.linear_model import LinearRegression


def pwlr(x, T):
    """
    Create a linear model per pixel returns the interpolated and extraploated values
    :param x: (N, t, 1,H,W)
    :param t: (N, T) # T>=t the extra time points are where extrapolation will be done
    :return out: (N,T,1,H,W)
    """

    nt = x.shape[1]
    t = T[:, :nt]  # remaining is used for forecasting/extrapolation

    N, H, W = x.shape[0], x.shape[3], x.shape[4]
    slopes = np.zeros((N, H, W, 1))
    intercepts = np.zeros((N, H, W, 1))

    for n in range(N):
        for h in range(H):
            for w in range(W):
                y_ = np.expand_dims(x[n, :, 0, h, w], -1)
                x_ = np.expand_dims(t[n, :], -1)
                reg = LinearRegression()
                reg.fit(x_, y_)
                slopes[n, h, w] = reg.coef_[0][0]
                intercepts[n, h, w] = reg.intercept_[0]

    slopes.repeat(T.shape[1], axis=3)  # (n,h,w,T,1)

    T = T[:, np.newaxis, np.newaxis, :]  # (N,1,1,T)
    T = T.repeat(H, axis=1).repeat(W, axis=2)  # (N,h,w,T)
    out = slopes * T + intercepts
    out = out.transpose([0, 3, 1, 2])  # (N,T,H,W)
    out = out[:, :, np.newaxis, :, :]
    return out


if (__name__ == '__main__'):
    x = np.random.random((10, 3, 1, 13, 13))
    T = np.random.random((10, 5))
    out = pwlr(x, T)
    print(out.shape)
