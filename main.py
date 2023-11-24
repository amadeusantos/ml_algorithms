import numpy as np
import pandas
import numpy
from sklearn.preprocessing import normalize

from pandas import Series


def main():
    data_frame_train = pandas.read_csv("resources/iris.data")
    data_frame_test = pandas.read_csv("resources/bezdekIris.data")
    # data_frame_train, data_frame_test = train_test_split(data_frame, test_size=0.2, random_state=32, shuffle=True)

    class_data_frame_train: Series = data_frame_train.pop("Class")

    class_data_frame_test: Series = data_frame_test.pop("Class")

    data_frame_train = normalize(data_frame_train)
    data_frame_test = normalize(data_frame_test)

    k = 3
    acc = 0
    print(data_frame_train)
    for index, test in enumerate(data_frame_test):
        next = elements_next(test, data_frame_train, class_data_frame_train, amount=k)
        if most_common(next) == class_data_frame_test.values[index]:
            acc += 1

    print("Acc: ", acc, "Rate: ", acc / np.shape(class_data_frame_test)[0])


def elements_next(element: numpy.ndarray, data_frame_train, class_data: Series, amount: int = 3):
    distances = list(map(lambda x: calculate_distance(element, x), data_frame_train))
    class_distances = [(value, class_data.values[index]) for index, value in enumerate(distances)]
    class_distances.sort(key=tuple[0])
    return class_distances[:amount]


def most_common(arrayElements):
    return max(set(arrayElements), key=arrayElements.count)[1]


def calculate_distance(element_1: numpy.ndarray, element_2: numpy.ndarray) -> float:
    distance = 0
    for index, coordinate in enumerate(element_1):
        distance += (coordinate - element_2[index]) ** 2
    return distance ** 0.5


main()
