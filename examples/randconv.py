import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from pycapreg import CAPRegressor, ClCAPRegressor, CONVEX, CONCAVE


class RandConvFn:
    def __init__(self, n_features, n_planes, scale=10):
        self.matrix = (np.random.random((n_planes, n_features))-0.5) *2 * scale
        self.intercepts = np.random.random((n_planes, 1)) * scale * 1000
        self.n_features = n_features

    def __call__(self, x):
        nx = np.atleast_2d(x)

        return np.max(self.matrix @ nx.T + self.intercepts @ np.ones((1, nx.shape[0])), axis=0)


if __name__ == '__main__':
    N = 2
    NP = 600
    np.random.seed(3)
    rfn = RandConvFn(n_features=N, n_planes=10)
    x = np.random.random((N, ))
    print(rfn(x))
    x = np.random.random((1, N))
    print(rfn(x))
    x = np.random.random((5, N))
    print(rfn(x))

    x = (np.random.random((NP, N))-0.5) * 100
    response = rfn(x) + np.sin(x[:, 0]/200) * np.random.random((NP, )) * 1000

    ct = CAPRegressor(
        concavity=CONVEX,
        min_leaf_samples=30)

    ct.fit(x, response)


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(x[:, 0], x[:, 1], response, s=10)
    ax.scatter(x[:, 0], x[:, 1], ct.predict(x), c='k', s=10)
    print(ct.score(x, response))
    plt.show()