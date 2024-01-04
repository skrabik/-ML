import numpy as np

# для вычисления ошибки
def calc_mse(y, y_pred):
    err = np.mean((y - y_pred)**2)
    return err

def gradient_descent_reg_l1(X, y, iterations, eta=1e-4, reg=1e-30):
    # выбираем случайные стартовые веса
    W = np.random.randn(X.shape[1])
    n = X.shape[0]

    # запускаем цикл длинной в количество итераций
    for i in range(0, iterations):
        y_pred = np.dot(X, W)
        err = calc_mse(y, y_pred)

        dQ = 2 / n * X.T @ (y_pred - y)  # градиент функции ошибки
        dReg = reg * np.sign(W)  # градиент регуляризации L1

        W -= eta * (dQ - dReg)

        if i % (iterations / 10) == 0:
            print(f'Iter: {i}, weights: {W}, error {err}')

    print(f'Final MSE: {calc_mse(y, np.dot(X, W))}')
    return W

X = np.array([[1, 1, 500, 1],
              [1, 1, 700, 1],
              [1, 2, 750, 2],
              [1, 5, 600, 1],
              [1, 3, 1450, 2],
              [1, 0, 800, 1],
              [1, 5, 1500, 3],
              [1, 10, 2000, 3],
              [1, 1, 450, 1],
              [1, 2, 1000, 2]])

y = [45, 55, 50, 55, 60, 35, 75, 80, 50, 60]


def standard_scale(X):
    mean = X.mean()
    std = X.std()
    return (X - mean) / std

X_st = X.copy().astype(np.float64)
X_st[:, 1] = standard_scale(X_st[:, 1])
X_st[:, 2] = standard_scale(X_st[:, 2])
X_st[:, 3] = standard_scale(X_st[:, 3])


gradient_descent_reg_l1(X_st, y, iterations=5000, eta=1e-2)

