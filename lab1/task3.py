import numpy as np

X = np.array([[ 1,  1],
              [ 1,  1],
              [ 1,  2],
              [ 1,  5],
              [ 1,  3],
              [ 1,  0],
              [ 1,  5],
              [ 1, 10],
              [ 1,  1],
              [ 1,  2]])

y = [45, 55, 50, 55, 60, 35, 75, 80, 50, 60]

def calc_mse(y, y_pred):
    err = np.mean((y - y_pred)**2)
    return err

n = X.shape[0]

eta = 1e-2
epsilon = float(input("Введите значение epsilon: "))

W = np.array([1, 0.5])

print(f'Number of objects = {n} \
       \nLearning rate = {eta} \
       \nInitial weights = {W} \n')
i = 1
while True:
    y_pred = np.dot(X, W)
    err = calc_mse(y, y_pred)
#     for k in range(W.shape[0]):
#         W[k] -= eta * (1/n * 2 * X[:, k] @ (y_pred - y))
    # ИЗМЕНЕНИЯ
    # print(eta *( 2 / n * X.T @ ((X @ W) - y)))
    W -= eta * ( 2 / n * X.T @ ((X @ W) - y))
    if np.linalg.norm((2 / n * X.T @ ((X @ W) - y))) < epsilon:
        break
    # ИЗМЕНЕНИЯ
    #
    if i % 10 == 0:
        print(f'Iteration #{i}: W_new = {W}, MSE = {round(err,2)}')
    i += 1