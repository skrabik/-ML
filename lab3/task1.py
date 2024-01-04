import numpy as np

def calc_logloss(y, y_pred):
    dim = y.shape[0]
    err = 0
    for i in range(dim):
        if y_pred[i] != 1.0 and y_pred[i] != 0:
            err -= y[i] * np.log(y_pred[i]) + (1.0 - y[i]) * np.log(1.0 - y_pred[i])
    # print(err)
    rez = err / dim
    return rez

# y1 = np.array([1, 0])
# y_pred1 = np.array([0.999, 0.01])
# print(calc_logloss(y1, y_pred1))

y1 = np.array([1, 0])
y_pred1 = np.array([1, 0.2])
print(calc_logloss(y1, y_pred1))