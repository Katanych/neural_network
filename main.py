import numpy as np

ih_wgt = np.array([
    [0.1, 0.2, -0.1],
    [-0.1, 0.1, 0.9],
    [0.1, 0.4, 0.1]]).T

hp_wgt = np.array([
    [0.3, 1.1, -0.3],
    [0.1, 0.2, 0.0],
    [0.0, 1.3, 0.1]]).T

WEIGHTS = [ih_wgt, hp_wgt]


def neural_network(inp, weights):
    """Функция нейронной сети.

    :param inp: вектор, в котором первый элемент - текущее число игр, второй - процент побед, третий - болельщиков.
    :param weights: весовые коэффициенты каждого элемента входного вектора.
    :return: предсказание.

    """

    hid = inp.dot(weights[0])
    prediction = hid.dot(weights[1])
    return prediction


def main():
    """Основная функция"""

    toes = np.array([8.5, 9.5, 9.9, 9.0])
    wl_rec = np.array([0.65, 0.8, 0.8, 0.9])
    num_fans = np.array([1.2, 1.3, 0.5, 1.0])

    inp = np.array([toes[0], wl_rec[0], num_fans[0]])
    prediction = neural_network(inp, WEIGHTS)
    print(prediction)


if __name__ == "__main__":
    main()

