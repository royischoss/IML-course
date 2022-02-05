import numpy as np
import matplotlib.pyplot as plt
import ex4_tools
import adaboost


# Q 13:
def q13(X_train, y_train, X_test, y_test):
    m = X_train.shape[0]
    D = np.ones(m) / m
    model = adaboost.AdaBoost(ex4_tools.DecisionStump, T=500)
    model.train(X_train, y_train)
    error_train = []
    error_test = []
    for i in range(1, 500):
        error_train.append(model.error(X_train, y_train, i))
        error_test.append(model.error(X_test, y_test, i))
    fig = plt.figure()
    plt.plot(np.arange(1, 500), error_train, label='Train error')
    plt.plot(np.arange(1, 500), error_test, label='Test error')
    plt.title('Error as function of T')
    plt.grid()
    plt.legend()
    plt.show()


# Q 14:
def q14(X_train, y_train, X_test, y_test):
    T = [5, 10, 50, 100, 200, 500]
    i = 0
    for t in T:
        model = adaboost.AdaBoost(ex4_tools.DecisionStump, T=t)
        model.train(X_train, y_train)
        plt.subplot(2, 3, i + 1)
        ex4_tools.decision_boundaries(model, X_test, y_test, num_classifiers=t)
        i += 1
    plt.show()


def q15(X_train, y_train, X_test, y_test):
    X_train, y_train = ex4_tools.generate_data(num_samples=5000, noise_ratio=0)
    T = [5, 10, 50, 100, 200, 500]
    error = []
    model = adaboost.AdaBoost(ex4_tools.DecisionStump, T=500)
    model.train(X_train, y_train)
    for t in range(1,500):
        error.append(model.error(X_test, y_test, t))
    error = np.array(error)
    arg_min = np.argmin(error)
    ex4_tools.decision_boundaries(model, X_test, y_test, num_classifiers=arg_min + 1)
    print("Min Error = " + str(error[arg_min]) + "for T = " + str(arg_min + 1))
    plt.show()


def q16(X_train, y_train):
    model = adaboost.AdaBoost(ex4_tools.DecisionStump, T=500)
    D = model.train(X_train, y_train)
    D = D / np.max(D) * 10
    ex4_tools.decision_boundaries(model, X_train, y_train, num_classifiers=500, weights=D)
    plt.show()


if __name__ == '__main__':
    X_train, y_train = ex4_tools.generate_data(num_samples=5000, noise_ratio=0.4)
    X_test, y_test = ex4_tools.generate_data(num_samples=200, noise_ratio=0.4)
    q13(X_train, y_train, X_test, y_test)
