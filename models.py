import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def score(y, y_hat):
    if y_hat is not None:
        diff = y - y_hat.T
        where = np.argwhere(diff != 0)
    else:
        return None
    num_samples = y.shape[0]
    error = where.shape[0] / num_samples
    accuracy = 1 - error
    fpr = np.argwhere(diff == -2).shape[0] / np.argwhere(y == -1).shape[0]
    tpr = np.argwhere(y_hat.T + y == 2).shape[0] / np.argwhere(y == 1).shape[0]
    if tpr == 0 and fpr == 0:
        precision = 0
    else:
        precision = tpr / (tpr + fpr)
    tn = np.argwhere(y_hat.T + y == -2).shape[0]
    specificity = tn / (tn + fpr * num_samples)
    return {"error": error, "num_samples": num_samples, "accuracy": accuracy, "FPR": fpr, "TPR": tpr,
            "Precision": precision, "Specificity": specificity}


class Perceptron:
    def __init__(self):
        self.model = None

    def fit(self, x, y):
        w = np.zeros((x.shape[1] + 1,))
        x = np.concatenate((x, np.ones(x.shape[0]).reshape(x.shape[0], 1)), axis=1)
        indices = np.where((y.T * np.inner(w, x)) <= 0)
        while len(indices[0]) >= 1:
            w += y[indices[0][0]] * x[indices[0][0], :]
            indices = np.where((y.T * np.inner(w, x)) <= 0)
        self.model = w.reshape(w.shape[0], 1)

    def predict(self, x):
        if self.model is not None:
            z = np.ones(x.shape[0])
            h = np.ones(x.shape[0]) * -1
            x = np.concatenate((x, np.ones(x.shape[0]).reshape(x.shape[0], 1)), axis=1)
            return np.where(np.dot(x, self.model).T > 0, z, h).T
        else:
            return None

    def score(self, x, y):
        y_hat = self.predict(x)
        return score(y, y_hat)


class LDA:
    def __init__(self):
        self.model = None
        self.positive = 1
        self.negative = -1

    def fit(self, x, y):
        indices_1 = np.argwhere(y == 1)
        indices_2 = np.argwhere(y == -1)
        mue_1 = np.mean(x[y == 1], axis=0)
        mue_2 = np.mean(x[y == -1], axis=0)
        pr_y_1 = indices_1.size / y.shape[0]
        pr_y_2 = indices_2.size / y.shape[0]

        sigma = ((x[y == -1] - mue_2).T @ (x[y == -1] - mue_2) + (x[y == 1] - mue_1).T @ (x[y == 1] - mue_1)) / \
                x.shape[0]
        inv_sigma = np.linalg.inv(sigma)
        self.model = {"inverse sigma": inv_sigma, "mue_1": mue_1, "mue_2": mue_2, "Prob(y_1)": pr_y_1,
                      "Prob(y_2)": pr_y_2}

    def predict(self, x):
        delta_1 = x @ self.model["inverse sigma"] @ self.model["mue_1"].T \
                  - 0.5 * self.model["mue_1"].T @ self.model["inverse sigma"] @ self.model["mue_1"] + np.log(
            self.model["Prob(y_1)"])
        delta_2 = x @ self.model["inverse sigma"] @ self.model["mue_2"] \
                  - 0.5 * self.model["mue_2"].T @ self.model["inverse sigma"] @ self.model["mue_2"] + np.log(
            self.model["Prob(y_2)"])
        z = np.ones(x.shape[0])
        h = np.ones(x.shape[0]) * -1
        y_hat = np.where(delta_1 > delta_2, z, h).T
        return y_hat

    def score(self, x, y):
        y_hat = self.predict(x)
        return score(y, y_hat)


class SVM:
    def __init__(self):
        self.model = SVC(C=1e10, kernel='linear')

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x, y):
        y_hat = self.predict(x)
        return score(y, y_hat)


class Logistic:
    def __init__(self):
        self.model = LogisticRegression(solver="liblinear")

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x, y):
        y_hat = self.predict(x)
        return score(y, y_hat)


class DecisionTree:

    def __init__(self):
        self.model = DecisionTreeClassifier()

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x, y):
        y_hat = self.predict(x)
        return score(y, y_hat)


class KNN:
    def __init__(self):
        self.model = None

    def fit(self, x, y):
        self.model = KNeighborsClassifier(n_neighbors=2)
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x, y):
        y_hat = self.predict(x)
        return score(y, y_hat)

