from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt

# функция для вычисления ошибки
def calc_mse(y, y_pred):
    err = np.mean((y - y_pred)**2)
    return err

# будем использовать функции градиентного и стохастического градиентного спуска
def gradient_descent(X, y, iterations, eta=1e-2):
    W = np.random.randn(X.shape[1])
    n = X.shape[0]

    # массив для хранения ошибок
    errors = []

    for i in range(0, iterations):
        y_pred = np.dot(X, W)
        err = calc_mse(y, y_pred)
        errors.append(err)
        dQ = 2 / n * X.T @ (y_pred - y)  # градиент функции ошибки
        W -= (eta * dQ)

    return W, errors


def stohastic_gradient_descent(X, y, iterations, size, eta=1e-2):
    W = np.random.randn(X.shape[1])
    n = X.shape[0]

    # создадим массив для хранения ошибок
    errors = []

    for i in range(0, iterations):
        inds = np.random.randint(n, size=size)

        X_tmp = X[inds,]
        y_tmp = np.array(y)[inds]

        y_pred_tmp = np.dot(X_tmp, W)
        dQ = 2 / len(y_tmp) * X_tmp.T @ (y_pred_tmp - y_tmp)  # градиент функции ошибки
        W -= (eta * dQ)

        err = calc_mse(y, np.dot(X, W))
        errors.append(err)
    return W, errors


# сгенерируем данные для модели
#
# модель для обучения будет следующая:
# количество экземпляров (n_samples) == 1000
# количество свойств (n_features) == 9
# bias - добавим немного шума
# coef == True, коэффициенты при линейных уравнениях, чтобы можно было сравнить с результатом
X, y, c = make_regression(n_samples=1000, n_features=9, bias=0.5, coef=True)

# количество итераций
iterations = 250

grad, gradErr = gradient_descent(X, y, iterations=iterations)
gradSt, gradStErr = stohastic_gradient_descent(X, y, iterations=iterations, size=9)

print(f'Результат градиентного спуска:\n {grad} \n')

print(f'Результат стохастического градиентного спуска:\n {gradSt} \n')

print(f'Ответы :)\n {c}')

plt.figure()
plt.plot([i for i in range(len(gradErr))], gradErr, label='Обычный градиентный спуск')
plt.plot([i for i in range(len(gradStErr))], gradStErr, label='Стохастический градиентный спуск')
plt.legend()
plt.xlabel('Количество итераций')
plt.ylabel('Значение ошибки')
plt.show()