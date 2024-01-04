import numpy as np

def calc_logloss(y, y_pred):
    err = - np.mean(y * np.log(y_pred) + (1.0 - y) * np.log(1.0 - y_pred))
    return err

def sigmoid(z):
    res = 1 / (1 + np.exp(-z))
    return res

def eval_model(X, y, iterations, eta=1e-4):
    np.random.seed(42)
    W = np.random.randn(X.shape[1])
    n = X.shape[0]

    for i in range(iterations):
        z = np.dot(X, W)
        y_pred = sigmoid(z)
        err = calc_logloss(y, y_pred)

        dQ = 1 / n * X.T @ (y_pred - y)
        W -= eta * dQ
        if i % (iterations / 10) == 0:
            print(i, W, err)
    return W

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
# def calc_logloss(y, y_pred):
#     dim = y.shape[0]
#     err = 0
#     for i in range(dim):
#         if y_pred[i] != 1.0 and y_pred[i] != 0:
#             err -= y[i] * np.log(y_pred[i]) + (1.0 - y[i]) * np.log(1.0 - y_pred[i])
#     rez = err / dim
#     return rez

y = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 1], dtype=np.float64)

X_st = X.copy()
X_st[:, 2] = standard_scale(X[:, 2])
print(X_st)
W = eval_model(X_st, y, iterations=5000, eta=1e-4)