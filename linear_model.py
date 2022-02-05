import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# q 9:
def fit_linear_regression(x_matrix, y):
    svd = np.linalg.svd(x_matrix, compute_uv=False)
    x_dagger = np.linalg.pinv(x_matrix)
    return np.dot(x_dagger, y), svd


# q 10:
def predict(x_matrix, w):
    return np.dot(x_matrix, w)


# q 11:
def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)


# q 12:
def load_data(path):
    data = pd.read_csv(path, na_values=['no info', '.'])
    data = data.dropna()
    data['date'] = data['date'].str.slice(0, 4)
    data['zipcode'] = pd.get_dummies(data['zipcode'])
    data_array = data.to_numpy().astype(np.float64)

    shape = data_array.shape[0], data_array.shape[1] - 2
    data_array = np.delete(data_array, [17, 18], axis=1).reshape(shape)
    data_array[:, 0] = np.ones(data_array[:, 0].size)
    to_delete = np.argwhere(data_array < 0)
    deleted_lines = to_delete.shape[0]
    data_array = np.delete(data_array, to_delete[:, 0], axis=0).reshape(shape[0] - deleted_lines, shape[1])
    y = data_array[:, 2][:]
    data_array = np.delete(data_array, [2], axis=1).reshape((shape[0] - deleted_lines, shape[1] - 1))
    return data_array, y


# q 14 :
def plot_singular_values(values):
    x = np.arange(1, values.size + 1)
    y = np.sort(values)[::-1]
    fig = plt.figure()
    plt.title("Singular Values")
    plt.ylabel("Values")
    plt.xlabel("Indices")
    plt.plot(x, y)
    plt.show()


# q 16 :
def split_data(x, y):
    """Return test data, test target, train data, train target"""
    data = np.concatenate((x, y.reshape(y.shape[0], 1)), axis=1)
    np.random.shuffle(data)
    test_lim = int(x.shape[0] / 4)
    return data[0:test_lim, :-1], data[0:test_lim, -1], data[test_lim:-1, :-1], data[test_lim:-1, -1]


def plot_mse(test_set, y_test, train_set, y_train):
    mse_arr = np.zeros(100)
    for i in range(1,101):
        index = int((i / 100) * train_set.shape[0])
        fit, s = fit_linear_regression(train_set[0:index, :], y_train[0:index])
        predicted = predict(test_set, fit)
        mse_arr[i - 1] = mse(y_test, predicted.T)
    fig = plt.figure()
    plt.title("MSE vs % train")
    plt.ylabel("MSE")
    plt.xlabel("Percentages")
    plt.plot(range(1, 101), mse_arr)
    plt.show()


# q 17:
def feature_evaluation(data_frame, x, y):
    fig = plt.figure()
    x_axis = np.arange(x.shape[0])
    headers = data_frame.columns
    headers = np.delete(headers.to_numpy(), [2, 17, 18], axis=0)
    for i in range(1, x.shape[1]):
        pearson_cor = np.cov(x[:, i], y)[0][1] / (np.std(x[:, i]) * np.std(y))
        plt.title("Correlation between\n" + headers[i] + ' to price \n pearson correlation = \n' + str(pearson_cor))
        plt.xlabel(headers[i])
        plt.ylabel('correspond vector')
        plt.plot(x[:, i], y, 'o', color='blue', label='correspond vector')
        plt.legend()
        plt.show()


# q 15:
if __name__ == '__main__':
    path = 'C:\\Users\\Roy\\PycharmProjects\\IMLEx2\\kc_house_data.csv'
    data_frame = pd.read_csv(path, na_values=['no info', '.'])
    data_frame = data_frame.dropna()
    x, y = load_data(path)
    w, singular = fit_linear_regression(x, y)
    plot_singular_values(singular)
    test_set, y_test, train_set, y_train = split_data(x, y)
    plot_mse(test_set, y_test, train_set, y_train)
    feature_evaluation(data_frame, x, y)





