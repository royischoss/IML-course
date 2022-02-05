import numpy as np
import mnist as mn
import matplotlib.pyplot as plt
import models
import time

mnist = mn.MNIST("C:\\Users\\Roy\\PycharmProjects\\IMLEx3\\dataset\\MNIST")
x_train, y_train = mnist.load_training()
x_test, y_test = mnist.load_testing()

x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.int32)
x_test = np.asarray(x_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.int32)

train_images = np.logical_or((y_train == 0), (y_train == 1))
test_images = np.logical_or((y_test == 0), (y_test == 1))
x_train, y_train = x_train[train_images], y_train[train_images]
x_test, y_test = x_test[test_images], y_test[test_images]

y_train = (y_train * 2) - 1
y_test = (y_test * 2) - 1


def draw_images():
    counter_0 = 0
    counter_1 = 0
    i = 0
    while (counter_0 < 3 or counter_1 < 3) and i < x_train.shape[0]:
        if y_train[i] == -1 and counter_0 < 3:
            counter_0 += 1
            plt.imshow(x_train[i].reshape(28, 28), cmap=plt.cm.gray)
            plt.show()
        elif y_train[i] == 1 and counter_1 < 3:
            counter_1 += 1
            plt.imshow(x_train[i].reshape(28, 28), cmap=plt.cm.gray)
            plt.show()
        i += 1


def rearrange_data(x):
    return x.reshape((x.shape[0], x.shape[1] * x.shape[2]))


def draw_points(x, y, m):
    number_of_rows = x.shape[0]
    random_indices = np.random.choice(number_of_rows, size=m, replace=False)
    return x[random_indices], y[random_indices]


def test(m, k):
    x, y_1 = draw_points(x_train, y_train, m)
    z, y_2 = draw_points(x_test, y_test, k)

    while y_1.max() != 1 or y_1.min() != -1:
        draw_points(x_train, y_train, m)

    l = models.Logistic()
    l.fit(x, y_1)
    score_log = l.score(z, y_2)

    dt = models.DecisionTree()
    dt.fit(x, y_1)
    score_dt = dt.score(z, y_2)

    knn = models.KNN()
    knn.fit(x, y_1)
    score_knn = knn.score(z, y_2)

    svm = models.SVM()
    svm.fit(x, y_1)
    score_svm = svm.score(z, y_2)
    return score_log["accuracy"], score_dt["accuracy"], score_knn["accuracy"], score_svm["accuracy"]


def compare():
    k = x_test.shape[0]
    mean_acc = np.zeros((4, 4))
    dt_accuracy = []
    knn_accuracy = []
    log_accuracy = []
    svm_accuracy = []
    j = 0
    # timer = []
    # timer_avg = []
    m_s = [50, 100, 300, 500]
    for m in m_s:
        for i in range(50):
            # timing disabled
            # start = time.time()
            # print("hello")
            acc = test(m, k)
            log_accuracy.append(acc[0])
            # end = time.time()
            # print(end - start)
            # timer.append(end - start)
            dt_accuracy.append(acc[1])
            knn_accuracy.append(acc[2])
            svm_accuracy.append((acc[3]))
        # timer_avg.append(np.mean(timer))
        mean_acc[j, 0] = np.mean(log_accuracy)
        mean_acc[j, 1] = np.mean(dt_accuracy)
        mean_acc[j, 2] = np.mean(knn_accuracy)
        mean_acc[j, 3] = np.mean(svm_accuracy)

        dt_accuracy = []
        knn_accuracy = []
        log_accuracy = []
        svm_accuracy = []
        # timer = []
        j += 1
    fig = plt.figure()
    plt.plot(m_s, mean_acc[:, 0], color='red', label='Logistic')
    # plt.plot(m_s, timer_avg, label='SVM time')
    plt.plot(m_s, mean_acc[:, 1], color='green', label='Decision Tree')
    plt.plot(m_s, mean_acc[:, 2], color='blue', label='KNN')
    plt.plot(m_s, mean_acc[:, 3], color='black', label='SVM')
    plt.title("Accuracy mean depends on m:")
    plt.ylabel("mean accuracy")
    plt.xlabel("m")
    plt.grid()
    plt.legend()
    plt.show()
