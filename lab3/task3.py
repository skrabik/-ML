import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def calc_pred_proba(w, X):
    m = X.shape[0]

    y_predicted = np.zeros(m)

    A = np.squeeze(sigmoid(np.dot(X, w)))

    # За порог отнесения к тому или иному классу примем вероятность 0.5
    for i in range(A.shape[0]):
        if (A[i] > 0.5):
            y_predicted[i] = 1
        elif (A[i] <= 0.5):
            y_predicted[i] = 0

    return y_predicted

def standard_scale(x):
    res = (x - x.mean()) / x.std()
    return res

X = np.array([ [   1,    1,  500,    1],
               [   1,    1,  700,    1],
               [   1,    2,  750,    2],
               [   1,    5,  600,    1],
               [   1,    3, 1450,    2],
               [   1,    0,  800,    1],
               [   1,    5, 1500,    3],
               [   1,   10, 2000,    3],
               [   1,    1,  450,    1],
               [   1,    2, 1000,    2]], dtype=np.float64)

X_st = X.copy()
X_st[:, 2] = standard_scale(X[:, 2])
W = [ 0.47998993, -0.20238516,  0.64624195,  1.49797551]


print(calc_pred_proba(W, X_st))