import pandas as pd
import numpy as np
from pycapreg import LTCAPRegressor, CONCAVE, CONVEX, ClCAPRegressor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')


COLORSb = [
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 0.5, 1.0),
    (1.0, 0.5, 0.0),
    (0.5, 0.75, 0.5),
    (0.3445056890983381, 0.011246870211062077, 0.6452094673566421),
    (0.7220997725975299, 0.03589937925492903, 0.03551588534926442),
    (0.035871216259758754, 0.5021293889615723, 0.1526019021074423),
    (0.8764249389476501, 0.48812441441940735, 0.9397856728680032),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.5),
    (0.746572771969184, 0.3112294420637951, 0.4389879794212187),
    (0.4957102361679141, 0.8415658535173653, 0.9911801387199278),
    (0.9864842563915523, 0.7862763215829006, 0.5042293023473167),
    (0.0, 0.0, 1.0),
    (0.5552304623917604, 0.8419507520356129, 0.002345510345173607),
    (0.05721718446118107, 0.3381575410624065, 0.5931169500656306),
    (0.33162972048034833, 0.193840635461354, 0.23686418258155062),
    (0.43110461721107773, 0.25551910805159705, 0.9985970660059255),
    (0.9722980140288845, 0.003922671061356908, 0.4841659187012356),
    (0.5, 0.5, 0.0),
    (0.06269545627148132, 0.7087744887281715, 0.6772378483486524),
    (0.5, 0.5, 0.75),
    (0.6954852803552696, 0.9900903796365177, 0.6733376813402882),
    (0.3627108474860462, 0.9811274376973242, 0.2971223071336244),
    (0.34957071777079884, 0.48111721147535813, 0.3945935943969948),
    (0.029258770427280534, 0.000665132769293697, 0.403683243090455),
    (0.9885254324652987, 0.2422093227322858, 0.739264468716409),
    (0.7779070665112466, 0.6740627125379937, 0.21100584586071047)]

if __name__ == '__main__':
    data = pd.read_csv(r"data_1029.csv")

    # X = data[["Education", "Experience"]].to_numpy().astype(float)
    data.drop(columns=["Unnamed: 0", "Black", "SMSA", "Region"], inplace=True)
    mndata = data.groupby(["Education", "Experience"], as_index=False).median()
    mncount = data.groupby(["Education", "Experience"], as_index=False).count()
    X = mndata[["Education", "Experience"]].to_numpy()
    y = np.squeeze(mndata[["Wage"]].to_numpy())
    yc = np.squeeze(mncount[["Wage"]].to_numpy())
    Xd = X.copy()
    Xd = Xd.astype(float)
    Xd[:, 0] = np.power(np.ones(X[:, 0].shape)*1.2, X[:, 0])

    N = X.shape[0]

    # fit CAPRegressor
    min_leaf = np.ceil(N / len(COLORSb))
    ct = ClCAPRegressor(
        concavity=CONCAVE,
        min_leaf_samples=min_leaf)

    ct.fit(Xd, y)
    print(ct.score(Xd, y))

    # create figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set(xlim=[0, 20], ylim=[0, 80], zlim=[0.0, 1200.0])
    ax.set_xlabel('Education')
    ax.set_ylabel('Experience')
    ax.set_zlabel('Wage')

    # partition the data according to the hyperplanes
    part_x, part_y, _ = ct.partition_data(Xd, y)

    ax.scatter(X[:, 0], X[:, 1], y, color='k', s=yc/5)
    # scatter the estimated output and show
    for pno, part in part_x.items():
        ax.scatter(np.emath.logn(1.2, part[:, 0]), part[:, 1], ct.predict(part), color=COLORSb.pop(), s=10)
    plt.show()