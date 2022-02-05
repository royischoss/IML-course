import models
import numpy as np
import matplotlib.pyplot as plt


def draw_points(m):
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]
    shape = (m,)
    vec = np.array([0.3, -0.5]).T
    x = np.random.multivariate_normal(mean, cov, shape)
    f_x = np.sign(np.inner(vec, x) + 0.1)
    y = f_x
    return x, y.T


def plot():
    x_axis = np.arange(-5, 5, 1)
    p = models.Perceptron()
    s = models.SVM()
    for m in [5, 10, 15, 25, 70]:
        x, y = draw_points(m)
        f_x = 0.6 * x_axis + 0.2
        y_pos_indices = np.argwhere(y == 1)
        y_neg_indices = np.argwhere(y == -1)
        x_pos_true = x[y_pos_indices.T[0], :]
        x_neg_true = x[y_neg_indices.T[0], :]
        fig = plt.figure()

        plt.plot(x_pos_true[:, 0], x_pos_true[:, 1], 'o', color='b', label='Positive')
        plt.plot(x_neg_true[:, 0], x_neg_true[:, 1], 'o', color='orange', label='Negative')
        plt.plot(x_axis, f_x, color='black', label="True")

        p.fit(x, y)
        per_hyp = p.model[0] / -p.model[1] * x_axis + p.model[2] / -p.model[1]
        plt.plot(x_axis, per_hyp, color='red', label='Perceptron')

        s.fit(x, y)
        svm_hyp = s.model.coef_[0][0] / -s.model.coef_[0][1] * x_axis + s.model.intercept_[0] / -s.model.coef_[0][1]
        plt.plot(x_axis, svm_hyp, color='green', label='SVM')

        plt.title("SVM vs Perceptron vs Truth m = " + str(m))
        plt.grid()
        plt.legend()
        plt.show()


def test(m, k):
    x, y_real = draw_points(m)
    z, y_test = draw_points(k)

    while y_real.max() != 1 or y_real.min() != -1:
        x, y_real = draw_points(m)

    p = models.Perceptron()
    p.fit(x, y_real)
    score_per = p.score(z, y_test)

    s = models.SVM()
    s.fit(x, y_real)
    score_svm = s.score(z, y_test)

    lda = models.LDA()
    lda.fit(x, y_real)
    score_lda = lda.score(z, y_test)
    return score_per["accuracy"], score_svm["accuracy"], score_lda["accuracy"]


def avg_accuracy():
    k = 10000
    mean_acc = np.zeros((5, 3))
    svm_accuracy = []
    lda_accuracy = []
    per_accuracy = []
    j = 0
    m_s = [5, 10, 15, 25, 70]
    for m in m_s:
        for i in range(500):
            acc = test(m, k)
            per_accuracy.append(acc[0])
            svm_accuracy.append(acc[1])
            lda_accuracy.append(acc[2])
        mean_acc[j, 0] = np.mean(per_accuracy)
        mean_acc[j, 1] = np.mean(svm_accuracy)
        mean_acc[j, 2] = np.mean(lda_accuracy)
        svm_accuracy = []
        lda_accuracy = []
        per_accuracy = []
        j += 1
    fig = plt.figure()
    plt.plot(m_s, mean_acc[:, 0], color='red', label='Perceptron')
    plt.plot(m_s, mean_acc[:, 1], color='green', label='SVM')
    plt.plot(m_s, mean_acc[:, 2], color='blue', label='LDA')
    plt.title("SVM vs Perceptron vs LDA accuracy depends on m:")
    plt.grid()
    plt.legend()
    plt.show()

